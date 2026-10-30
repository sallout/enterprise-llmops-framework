"""LangGraph preparation with disk-backed validation, grouping, selection, and audit exports."""
from collections import Counter
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
from typing import TypedDict
from pydantic import ValidationError
from langgraph.graph import StateGraph, START, END
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.data_io import atomic_text, digest, dump_json
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash, write_jsonl, now
from .adapters import adapt, InvalidExample
from policy_dissertation.data.adapters import adapt_native
from .deduplication import deduplicate
from .quality import scan_sensitive, validate_tokens
from .schemas import VERSION, UnifiedRecord
from .selection import PIPELINES, SPLITS, ratios, mix, split
from .storage import PreparationStore
from .tokenization import load_tokenizer


class PreparationState(TypedDict, total=False):
    stage: str
    run_directory: str
    manifest_path: str
    report_path: str


def strict_json(text):
    def invalid(value):
        raise ValueError('Nonfinite JSON value')
    return json.loads(text, parse_constant=invalid)


def bounded_lines(path, maximum):
    with path.open('rb') as stream:
        number = 0
        while True:
            data = stream.readline(maximum+1)
            if not data:break
            number += 1
            if len(data)>maximum:
                h=hashlib.sha256(data)
                while not data.endswith(b'\n'):
                    data=stream.readline(maximum+1)
                    if not data:break
                    h.update(data)
                yield number,None,h.hexdigest()
            else:
                yield number,data,hashlib.sha256(data).hexdigest()


class PreparationPipeline:
    def __init__(self, source, output, settings, maximum, seed, max_examples, partial, tokenizer, progress, report_path, configured_sources=None):
        self.source,self.output,self.settings=source,output,settings
        self.configured_sources = configured_sources
        self.maximum,self.seed,self.max_examples,self.partial=maximum,seed,max_examples,partial
        self.tokenizer,self.progress,self.report_path=tokenizer,progress,report_path
        self.store,self.work,self.inputs=None,None,[]
        self.mix_weights=ratios(settings.pipeline4.mix_ratios, tuple(s.name for s in configured_sources) if configured_sources else PIPELINES)
        self.split_weights=ratios(settings.pipeline4.split_ratios,SPLITS)
        if self.split_weights['train']<=0:
            raise DissertationError('Training split must have positive weight.')
        if settings.pipeline4.near_remove_threshold < settings.pipeline4.near_threshold:
            raise DissertationError('Near-removal threshold must be at least the grouping threshold.')
        self.report={'schema_version':VERSION,'status':'RUNNING','training_ready':False,'run_id':'not_discovered',
            'started_at':now(),'input_directory':str(source),'output_directory':str(output),
            'max_tokens':maximum,'seed':seed,'max_examples':max_examples,'allow_partial_sources':partial,
            'errors':[],'input_issues':[],'warnings':[],'inputs':[],
            'input_records':0,'validated_examples':0,'rejected_examples':0,'duplicates_removed':0,
            'not_selected_examples':0,'accepted_examples':0,'splits':{},'resumed_stages':[],
            'tokenizer':{'validated':False},'llm_calls':0,'tests_executed_in_pipeline4':0,
            'versions':{p:version(p) for p in ('transformers','tokenizers','langgraph','pydantic')},
            'validation_scope':'deterministic preparation; no semantic certification or reexecution of upstream tests'}

    def discover(self, state):
        self.progress('[1/7] Inspecting configured datasets and source hashes' if self.configured_sources else '[1/7] Inspecting all three upstream datasets and export hashes')
        inputs=[]
        for pipeline in (() if self.configured_sources else PIPELINES):
            folder=self.source/pipeline
            path,report_path=folder/'training.jsonl',folder/'report.json'
            entry={'pipeline':pipeline,'training_path':str(path),'readable':False}
            if not path.is_file():
                self.report['input_issues'].append(pipeline+':missing_training_jsonl')
            elif not report_path.is_file():
                self.report['input_issues'].append(pipeline+':missing_upstream_report')
            else:
                try:
                    upstream=strict_json(report_path.read_text(encoding='utf-8'))
                    actual=file_hash(path)
                    entry.update(training_hash=actual,report_hash=file_hash(report_path),
                        upstream_status=upstream.get('status'),upstream_accepted=upstream.get('accepted_examples'))
                    if upstream.get('output_hashes',{}).get('training.jsonl') != actual:
                        self.report['input_issues'].append(pipeline+':training_checksum_mismatch')
                    else:
                        entry['readable']=True
                    if upstream.get('status') not in ('COMPLETED','COMPLETED_WITH_WARNINGS') or upstream.get('errors'):
                        self.report['input_issues'].append(pipeline+':upstream_run_failed_or_incomplete')
                except (OSError,ValueError,TypeError,AttributeError):
                    self.report['input_issues'].append(pipeline+':invalid_upstream_report')
            inputs.append(entry)
        if self.configured_sources:
            from policy_dissertation.data.sources import materialize
            for specification in self.configured_sources:
                entry = materialize(specification, self.output, self.settings)
                inputs.append(entry)
                if (entry.get('producer_run') or {}).get('accepted_subset_only'):
                    self.report['warnings'].append(specification.name+':accepted_subset_from_incomplete_producer')
        self.inputs=self.report['inputs']=inputs
        if self.report['input_issues'] and not self.partial:
            self.report['errors'].extend(self.report['input_issues'])
            raise DissertationError('Mandatory upstream datasets are missing, failed, or inconsistent; inspect input_issues.')
        if self.report['input_issues']:
            self.report['warnings'].append('partial_source_diagnostic_only')
        self.progress('[2/7] Loading actual pinned Qwen3-14B tokenizer assets (no model weights)')
        token=self.settings.hf_token.get_secret_value() if self.settings.hf_token else None
        self.tokenizer=self.tokenizer or load_tokenizer(self.output,self.settings.pipeline4,token,self.progress)
        self.report['tokenizer']=self.tokenizer.info
        sources={str(p.relative_to(Path(__file__).parents[2])):file_hash(p) for p in Path(__file__).parent.glob('*.py')}
        key=digest({'version':VERSION,'inputs':[{k:v for k,v in i.items() if k!='training_path'} for i in inputs],
            'settings':self.settings.pipeline4.model_dump(),'limit':self.maximum,'seed':self.seed,
            'max_examples':self.max_examples,'partial':self.partial,'tokenizer':self.tokenizer.info,
            'code':sources,'shared_code': {p.name:file_hash(p) for p in (Path(__file__).parents[2]/'data').glob('*.py')}, 'versions':self.report['versions'],
            'source_schema_hashes':{n:file_hash(Path(__file__).parents[1]/n) for n in
                ('external_datasets/schemas.py','repository_data/schemas.py','policy_transformation/data_schemas.py')}})
        self.report['run_id']=key
        self.work=self.output/'_runs'/key
        self.work.mkdir(parents=True,exist_ok=True)
        self.store=PreparationStore(self.work/'state.sqlite3')
        return {'stage':'discovered','run_directory':str(self.work)}

    def load(self, state):
        self.progress('[3/7] Adapting and validating conversations, references, secrets, and exact token limits')
        if self.store.complete('load'):
            self.report['resumed_stages'].append('load')
            return {'stage':'loaded'}
        secrets=[v.get_secret_value() for v in (self.settings.model.api_key,self.settings.hf_token) if v]
        for entry in self.inputs:
            if not entry['readable']:continue
            pipeline=entry['pipeline']
            for line,data,raw_hash in bounded_lines(Path(entry['training_path']),self.settings.pipeline4.max_line_bytes):
                if self.store.db.execute('SELECT 1 FROM records WHERE pipeline=? AND line=?',(pipeline,line)).fetchone():continue
                record,reason,tokens=None,None,None
                ref={'pipeline':pipeline,'file':pipeline+'/training.jsonl','file_hash':entry['training_hash'],'line':line,'record_hash':raw_hash}
                try:
                    if data is None:raise InvalidExample('oversized_jsonl_record')
                    if not data.strip():raise InvalidExample('empty_jsonl_record')
                    text=data.decode('utf-8')
                    scan_sensitive(text,self.settings.pipeline4,secrets)
                    raw=strict_json(text)
                    if not isinstance(raw,dict):raise InvalidExample('unsupported_record_structure')
                    raw=adapt_native(raw,entry,line-1,self.settings)
                    record=adapt(raw,pipeline,ref)
                    record['metadata']['dataset_source'] = entry.get('source_spec', {'name': pipeline, 'type': 'local'})
                    if entry.get('resolved_revision'):
                        record['metadata']['dataset_source']['resolved_revision'] = entry['resolved_revision']
                    if entry.get('split') in ('test', 'validation'):
                        record['metadata']['official_holdout'] = entry['split']
                    tokens=validate_tokens(record,self.tokenizer,self.maximum,self.settings.pipeline4)
                except InvalidExample as exc:
                    reason=str(exc)
                except (ValueError,TypeError,KeyError,AttributeError,ValidationError):
                    reason='invalid_json_or_source_schema'
                priority=4 if record and record['verification']['status']=='upstream_local_pass' else 3 if record and record['metadata']['pipeline']=='pipeline1' else 2 if record and record['metadata']['pipeline']=='pipeline2' else 1
                with self.store.db:
                    self.store.db.execute('INSERT INTO records (pipeline,line,raw_hash,payload,reason,priority,domain,source,status,tokens,rank) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                        (pipeline,line,raw_hash,json.dumps(record,ensure_ascii=False,allow_nan=False) if record and not reason else None,
                         reason,priority,record['domain'] if record else None,record['source'] if record else None,
                         record['verification']['status'] if record else None,tokens,
                         digest([self.seed,record['id']]) if record else None))
                if line==1 or line%100==0:self.progress(f'  {pipeline}: checked {line} rows')
        self.store.finish('load')
        return {'stage':'loaded'}

    def dedup(self, state):
        self.progress('[4/7] Deduplicating and grouping related source examples')
        if self.store.complete('deduplication'):self.report['resumed_stages'].append('deduplication')
        deduplicate(self.store,self.settings.pipeline4,self.progress)
        return {'stage':'deduplicated'}

    def mixing(self, state):
        self.progress('[5/7] Balancing source contributions without replacement')
        self.report['mixing']=mix(self.store,self.mix_weights,self.max_examples,self.seed)
        available=self.report['mixing']['available']
        for p,w in self.mix_weights.items():
            if w>0 and not available.get(p):
                if self.partial:self.report['warnings'].append(p+':no_eligible_examples')
                else:self.report['errors'].append(p+':mandatory_source_no_eligible_examples')
        return {'stage':'mixed'}

    def splitting(self, state):
        self.progress('[6/7] Assigning component-disjoint splits and auditing leakage')
        self.report['splitting']=split(self.store,self.split_weights,self.seed)
        self.report['errors'].extend(self.report['splitting']['errors'])
        return {'stage':'split'}

    def export(self, state=None):
        self.progress('[7/7] Exporting datasets, rejection audit, manifest, and research report')
        r=self.report
        counts=Counter()
        self.output.mkdir(parents=True,exist_ok=True)
        def exported(name):
            if not self.store:return
            for row in self.store.rows('selected=1 AND split=?',(name,)):
                data=json.loads(row['payload'])
                data['metadata']['split']=name
                data['metadata']['split_group']=row['root']
                duplicates=[]
                duplicate_count=self.store.count('duplicate_of=?',(row['rid'],))
                for other in self.store.db.execute('SELECT * FROM records WHERE duplicate_of=? ORDER BY rid LIMIT 100',(row['rid'],)):
                    if self.store.canonical(other['rid'])==row['rid']:
                        original=json.loads(other['payload'])
                        duplicates.append({'input_reference':original['metadata']['input_reference'],
                            'upstream_example_id':original['metadata']['upstream_example_id'],
                            'source':original['source'],'verification_status':original['verification']['status'],
                            'official_holdout':original['metadata'].get('official_holdout')})
                if duplicates:data['metadata']['duplicate_references']=duplicates
                if duplicate_count:
                    data['metadata']['duplicate_reference_count']=duplicate_count
                    data['metadata']['complete_duplicate_audit']='rejected.jsonl; duplicate_of='+data['id']
                UnifiedRecord.model_validate(data)
                counts[name]+=1
                yield data
        for name in SPLITS:write_jsonl(self.output/(name+'.jsonl'),exported(name))
        def rejected():
            if not self.store:return
            for row in self.store.rows('selected=0'):
                reason=row['reason'] or 'not_selected_by_mixing'
                entry=next(i for i in self.inputs if i['pipeline']==row['pipeline'])
                item={'pipeline':row['pipeline'],'input_file':row['pipeline']+'/training.jsonl',
                      'input_file_hash':entry['training_hash'],'line':row['line'],'record_hash':row['raw_hash'],
                      'disposition':'duplicate' if row['duplicate_of'] else 'rejected' if row['reason'] else 'not_selected',
                      'reasons':[reason]}
                if row['payload']:
                    original=json.loads(row['payload'])
                    item.update(id=original['id'],source=original['source'],upstream_example_id=original['metadata']['upstream_example_id'],
                        verification_status=original['verification']['status'],official_holdout=original['metadata'].get('official_holdout'))
                if row['duplicate_of']:
                    winner=self.store.db.execute('SELECT payload FROM records WHERE rid=?',(self.store.canonical(row['rid']),)).fetchone()[0]
                    item['duplicate_of']=json.loads(winner)['id']
                yield item
        write_jsonl(self.output/'rejected.jsonl',rejected())
        if self.store:
            r['input_records']=self.store.count()
            r['validated_examples']=self.store.count('payload IS NOT NULL')
            r['rejected_examples']=self.store.count('payload IS NULL')
            r['duplicates_removed']=self.store.count('duplicate_of IS NOT NULL')
            r['not_selected_examples']=self.store.count('payload IS NOT NULL AND reason IS NULL AND selected=0')
            r['rejection_reasons']=self.store.counts('reason','reason IS NOT NULL')
            r['deduplication']={'exact_removed':self.store.count("reason='exact_duplicate'"),
                'near_format_removed':self.store.count("reason='near_format_duplicate'"),
                'near_pairs_detected':self.store.db.execute('SELECT count(*) FROM near_pairs').fetchone()[0],
                'near_method':'exact prefix-filtered SHA256 word-5-gram set Jaccard; full intersection verification',
                'near_threshold':self.settings.pipeline4.near_threshold,
                'removal_policy':'format equivalence only; retain content differences in same split group'}
            r['distribution']={f:self.store.counts(f,'selected=1') for f in ('pipeline','domain','source','status')}
            row=self.store.db.execute('SELECT min(tokens),max(tokens),avg(tokens),sum(tokens) FROM records WHERE selected=1').fetchone()
            r['token_statistics']=dict(zip(('min','max','mean','total'),row))
        r['splits']={s:counts[s] for s in SPLITS}
        r['accepted_examples']=sum(counts.values())
        if not r['accepted_examples']:r['errors'].append('no_exported_examples')
        # Upstream counts are checked even when quality rejection reduced the eligible cohort.
        if self.store:
            for entry in self.inputs:
                if entry['readable'] and entry.get('upstream_accepted') is not None and entry.get('upstream_accepted') != self.store.count('pipeline=?',(entry['pipeline'],)):
                    if self.partial:r['warnings'].append(entry['pipeline']+':upstream_count_mismatch')
                    else:r['errors'].append(entry['pipeline']+':upstream_count_mismatch')
        for entry in self.inputs:
            if entry['readable']:
                try:
                    if file_hash(Path(entry['training_path'])) != entry['training_hash'] or ('report_hash' in entry and file_hash(Path(entry.get('report_path', str(Path(entry['training_path']).parent/'report.json')))) != entry['report_hash']):
                        r['errors'].append(entry['pipeline']+':input_changed_during_preparation')
                except OSError:
                    r['errors'].append(entry['pipeline']+':input_changed_during_preparation')
        r['errors']=list(dict.fromkeys(r['errors']))
        r['training_ready']=bool(not r['errors'] and not r['input_issues'] and not self.partial and r['tokenizer'].get('validated') and r['accepted_examples'])
        if not r['tokenizer'].get('validated'):r['warnings'].append('actual_tokenizer_not_validated')
        r['status']='FAILED' if r['errors'] else 'COMPLETED_WITH_WARNINGS' if (r['warnings'] or r['rejected_examples'] or r['duplicates_removed'] or r['not_selected_examples'] or not r['training_ready']) else 'COMPLETED'
        r['finished_at']=now()
        manifest={'schema_version':VERSION,'run_id':r['run_id'],'training_ready':r['training_ready'],'status':r['status'],'exports_valid':True,
            'inputs':self.inputs,'tokenizer':r['tokenizer'],'settings':self.settings.pipeline4.model_dump(),
            'max_tokens':self.maximum,'seed':self.seed,'mixing':r.get('mixing'), 'splitting':r.get('splitting'),
            'outputs':{n+'.jsonl':file_hash(self.output/(n+'.jsonl')) for n in (*SPLITS,'rejected')},
            'consumer_contract':'Apply the pinned original chat template to messages and optional tools only; add_generation_prompt=False; enable_thinking=False; never truncate. Metadata/verification are auxiliary; assistant-only loss masking is a downstream trainer requirement.'}
        dump_json(self.output/'manifest.json',manifest)
        r['output_hashes']=manifest['outputs']
        self.update_report()
        manifest.update(training_ready=r['training_ready'],status=r['status'])
        dump_json(self.output/'manifest.json',manifest)
        dump_json(self.output/'report.json',r)
        if self.work:dump_json(self.work/'last_execution_report.json',r)
        return {'stage':'exported','manifest_path':str(self.output/'manifest.json'),'report_path':str(self.output/'report.json')}

    def update_report(self):
        r=self.report
        # Failed pre-discovery runs use an input-specific marker rather than a shared generic marker.
        marker=r['run_id'] if r['run_id']!='not_discovered' else digest([str(self.source),r['input_issues'],self.partial])
        start,end=f'<!-- pipeline4-run:{marker} -->',f'<!-- /pipeline4-run:{marker} -->'
        block=f'''{start}
## Pipeline 4 execution — {r['finished_at']}

- Status: **{r['status']}**; preparation training-ready: **{r['training_ready']}** (not semantic certification).
- Input records: {r['input_records']}; structurally/token valid: {r['validated_examples']}; rejected invalid: {r['rejected_examples']}.
- Duplicates removed: {r['duplicates_removed']}; excluded by mixing: {r['not_selected_examples']}; exported: {r['accepted_examples']}.
- Split counts: {json.dumps(r['splits'])}; distribution: {json.dumps(r.get('distribution',{}))}.
- Tokenizer: {json.dumps(r['tokenizer'])}; selected token statistics: {json.dumps(r.get('token_statistics',{}))}.
- Source issues: {json.dumps(r['input_issues'])}; errors: {json.dumps(r['errors'])}.
- Resumed stages: {json.dumps(r['resumed_stages'])}; no model training, LLM calls, or reexecution of source tests.
- Report: `{self.output}/report.json`. Held-out tests are excluded from training and hyperparameter-selection inputs.
{end}'''
        try:
            text=self.report_path.read_text() if self.report_path.exists() else '# Dissertation Project Report\n'
            if start in text and end in text:
                a,b=text.index(start),text.index(end)+len(end);text=text[:a]+block+text[b:]
            else:text=text.rstrip()+'\n\n'+block+'\n'
            atomic_text(self.report_path,text)
            r['research_report_updated']=True
        except OSError:
            r['errors'].append('research_report_update_failed');r['status']='FAILED';r['training_ready']=False

    def graph(self):
        graph=StateGraph(PreparationState)
        nodes=('discover','load','dedup','mixing','splitting','export')
        for name in nodes:graph.add_node(name,getattr(self,name))
        for a,b in zip((START,*nodes),(*nodes,END)):graph.add_edge(a,b)
        return graph.compile()


def run_pipeline4(input_directory,output_directory,settings,max_tokens=4096,*,seed=42,max_examples=None,
                  allow_partial_sources=False,tokenizer=None,progress=lambda text:None,research_report=None, sources=None, overwrite=False):
    source,output=Path(input_directory).resolve(),Path(output_directory).resolve()
    if max_tokens<1 or (max_examples is not None and max_examples<1):raise DissertationError('Limits must be positive.')
    source_folders=[(source/p).resolve() for p in PIPELINES]
    input_files=[(source/p/name).resolve() for p in PIPELINES for name in ('training.jsonl','report.json')]
    if output==source or any(output==folder or output.is_relative_to(folder) for folder in source_folders) or any(path.is_relative_to(output) for path in input_files):
        raise DissertationError('Output must not overwrite upstream source directories.')
    if sources:
        for specification in sources:
            if specification.type == 'local':
                path = Path(specification.path).resolve()
                if output == path or output.is_relative_to(path) or path.is_relative_to(output):
                    raise DissertationError('Output must not overlap configured local inputs.')
    with output_lock(output):
        from policy_dissertation.data.hub import protect_exports
        protect_exports(output, overwrite)
        engine=PreparationPipeline(source,output,settings,max_tokens,seed,max_examples,allow_partial_sources,
                                   tokenizer,progress,research_report or Path('report.md').resolve(), sources)
        dump_json(output/'report.json',engine.report)
        try:engine.graph().invoke({'stage':'starting'})
        except (Exception,KeyboardInterrupt) as exc:
            category=str(exc) if isinstance(exc,DissertationError) else 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'preparation_failed:'+type(exc).__name__
            engine.report['errors'].append(category)
            try:engine.export()
            except Exception:
                engine.report.update(status='FAILED',training_ready=False,finished_at=now(),exports_valid=False)
                engine.report['errors'].append('export_or_checkpoint_validation_failed')
                dump_json(output/'manifest.json',{'run_id':engine.report['run_id'],'status':'FAILED','training_ready':False,'exports_valid':False})
                dump_json(output/'report.json',engine.report)
        finally:
            if engine.store:engine.store.close()
        return engine.report
