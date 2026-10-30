"""Prepare native source selections with exact tokens, leakage gates and recovery.

Only source coordinates, hashes and quality decisions are persisted; record bodies
stay native on the Hub. This graph never loads model weights or starts training.
"""
from collections import Counter
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from policy_dissertation.config import Pipeline3Settings, Pipeline4Settings
from policy_dissertation.data.adapters import InvalidExample
from policy_dissertation.data.training_inputs import TrainingInputGate, index_enterprise, validate_native_record, pinned_enterprise_settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
from policy_dissertation.pipelines.policy_transformation.data_io import digest, dump_json, atomic_text, save_cache, load_cache
from policy_dissertation.pipelines.dataset_preparation.tokenization import load_tokenizer
from policy_dissertation.pipelines.external_datasets.pipeline import allocations, file_hash, now
from policy_dissertation.pipelines.external_datasets.normalization import RejectedRecord
from policy_dissertation.pipelines.external_datasets.sampling import sampled
from policy_dissertation.pipelines.external_datasets.sources import HubProvider, DATASETS, inspected_source, retry, error_category

VERSION='native-nvidia-selection-v1'


class NativeState(TypedDict,total=False):
    stage:str
    run_directory:str


def _save(db,key,value):
    db.execute('INSERT OR REPLACE INTO native_meta VALUES (?,?)',(key,json.dumps(value,ensure_ascii=False)))


def _load(db,key,default=None):
    row=db.execute('SELECT value FROM native_meta WHERE key=?',(key,)).fetchone()
    return json.loads(row[0]) if row else default


def decision_digest(db):
    h=hashlib.sha256()
    for row in db.execute('SELECT source,position,data FROM native_decisions ORDER BY source,position'):
        h.update(json.dumps(list(row),ensure_ascii=False,separators=(',',':')).encode())
        h.update(b'\n')
    return h.hexdigest()


class NativePipeline:
    def __init__(self,enterprise,output,settings,maximum,selected,seed,allocation,cap,configs,revisions,tokenizer,provider,progress):
        self.enterprise,self.output=enterprise,output
        self.settings=pinned_enterprise_settings(enterprise,settings)
        try:
            self.previous_report=json.loads((output/'report.json').read_text()) if (output/'report.json').exists() else {}
            self.previous_plan=json.loads((output/'source_plan.json').read_text()) if (output/'source_plan.json').exists() else {}
        except (OSError,ValueError):raise DissertationError('Existing native plan/report is unreadable; use a new output directory.') from None
        self.maximum,self.selected,self.seed=maximum,selected,seed
        self.quotas=allocations(selected,maximum,allocation or {})
        self.cap,self.configs,self.revisions=cap,configs or {},revisions or {}
        self.tokenizer,self.provider,self.progress=tokenizer,provider,progress
        self.gate,self.work,self.sources=None,None,[]
        self.report={'schema_version':VERSION,'status':'RUNNING','started_at':now(),'run_id':None,
            'max_samples':maximum,'records_processed':0,'accepted_examples':0,'rejected_examples':0,
            'decoded_rows_read_this_run':0,'decoded_bytes_read_this_run':0,'replayed_rows_this_run':0,
            'resumed_decisions':0,'stream_failures_this_run':0,'errors':[],'warnings':[],
            'training_started':False,'llm_calls':0,'locally_executed_external_tests':0,
            'data_prepared':False,'semantic_status':'not_semantically_certified',
            'sampling':'seeded bounded prefix buffer; not global-uniform sampling',
            'network_bytes':'not measured; row/byte counters count records delivered to the sampler, excluding internal reader/Arrow/network read-ahead',
            'versions':{p:version(p) for p in ('datasets','pyarrow','pydantic','transformers','tokenizers','langgraph')}}

    def discover(self,state):
        self.progress('[1/4] Inspecting native dataset schemas/licenses/configurations and pinning revisions')
        token=self.settings.hf_token.get_secret_value() if self.settings.hf_token else None
        self.provider=self.provider or HubProvider(self.settings.pipeline3,token)
        for alias in self.selected:
            key=digest([alias,self.revisions.get(alias)])
            cache=self.output/'_metadata'/(key+'.json')
            info=load_cache(cache,key) if cache.exists() else None
            if cache.exists() and not info:raise DissertationError('Native metadata cache checksum mismatch.')
            if not info:
                info=retry(lambda alias=alias:self.provider.inspect(alias,self.revisions.get(alias)),self.settings.pipeline3,self.progress)
                save_cache(cache,key,info)
            source=inspected_source(info,self.configs.get(alias) or ('chat' if alias=='nemotron' else 'train'),'train')
            source['quota']=self.quotas[alias]
            source['scan_limit']=self.cap or max(1,source['quota']*self.settings.pipeline3.scan_multiplier)
            self.sources.append(source)
        self.progress('[2/4] Loading pinned Qwen tokenizer and validating enterprise split fingerprints')
        self.tokenizer=self.tokenizer or load_tokenizer(self.output,self.settings.pipeline4,token,self.progress)
        split_hashes={s:file_hash(self.enterprise/(s+'.jsonl')) for s in ('train','validation','test')}
        code={str(p.relative_to(Path(__file__).parents[2])):file_hash(p) for directory in
            (Path(__file__).parent,Path(__file__).parents[2]/'data',Path(__file__).parents[1]/'external_datasets',Path(__file__).parents[1]/'dataset_preparation') for p in directory.glob('*.py')}
        fingerprint=digest({'version':VERSION,'sources':self.sources,'enterprise':split_hashes,'seed':self.seed,
            'settings3':self.settings.pipeline3.model_dump(),'settings4':self.settings.pipeline4.model_dump(),
            'max_tokens':self.max_tokens,'tokenizer':self.tokenizer.info,'code':code,'versions':self.report['versions']})
        self.code=code
        self.report.update(run_id=fingerprint,tokenizer=self.tokenizer.info,enterprise_split_hashes=split_hashes,max_tokens=self.max_tokens)
        self.work=self.output/'_runs'/fingerprint;self.work.mkdir(parents=True,exist_ok=True)
        prior_db=self.work/'fingerprints.sqlite3'
        if (self.previous_plan.get('run_id')==fingerprint and self.previous_report.get('status')!='RUNNING' and
            self.previous_plan.get('decision_database_hash') and prior_db.exists() and
            file_hash(prior_db)!=self.previous_plan['decision_database_hash']):
            raise DissertationError('Native checkpoint file changed after its finalized audit; resume refused.')
        self.gate=TrainingInputGate(self.work/'fingerprints.sqlite3',self.settings.pipeline4.near_threshold,
                                    resume=True,autocommit=False)
        db=self.gate.db
        db.executescript('''
          CREATE TABLE IF NOT EXISTS native_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS native_decisions(source TEXT,position INTEGER,data TEXT NOT NULL,checksum TEXT NOT NULL,
            PRIMARY KEY(source,position));
        ''')
        positions=Counter()
        for row in db.execute('SELECT source,position,data,checksum FROM native_decisions ORDER BY source,position'):
            if row['position'] != positions[row['source']]:raise DissertationError('Native checkpoint positions are not contiguous.')
            positions[row['source']]+=1
            if digest(json.loads(row['data']))!=row['checksum']:raise DissertationError('Native decision checksum mismatch.')
        self.report['resumed_decisions']=db.execute('SELECT count(*) FROM native_decisions').fetchone()[0]
        if not _load(db,'enterprise_indexed'):
            with db:
                index_enterprise(self.enterprise,self.settings,self.tokenizer,self.max_tokens,self.gate)
                _save(db,'enterprise_indexed',True)
                _save(db,'enterprise_info',self.gate.sources)
                _save(db,'counts',dict(self.gate.counts))
        else:
            self.gate.sources=_load(db,'enterprise_info',[])
            self.gate.counts=Counter(_load(db,'counts',{}))
        self.gate.seal_holdouts()
        if any(not x.get('source_run_complete',True) for x in self.gate.sources):
            self.report['warnings'].append('enterprise_accepted_subset_from_incomplete_producer')
        return {'stage':'discovered','run_directory':str(self.work)}

    def read(self,source):
        def on_read(size,raw):
            self.report['decoded_rows_read_this_run']+=1
            self.report['decoded_bytes_read_this_run']+=size
        return sampled(self.provider.stream(source),self.seed,self.settings.pipeline3,on_read)

    def process_source(self,source):
        alias=source['alias'];db=self.gate.db
        processed=accepted=0
        for row in db.execute('SELECT data FROM native_decisions WHERE source=? ORDER BY position',(alias,)):
            processed+=1;accepted+=int(json.loads(row[0])['accepted'])
        if accepted>=source['quota'] or _load(db,'exhausted:'+alias):return
        stream=None;failures=0
        try:
            while accepted<source['quota'] and processed<source['scan_limit']:
                try:
                    if stream is None:
                        stream=self.read(source)
                        for previous in db.execute('SELECT data FROM native_decisions WHERE source=? ORDER BY position',(alias,)):
                            expected=json.loads(previous[0])
                            try:replayed=next(stream)
                            except StopIteration:raise DissertationError('Pinned native stream ended before committed replay rows.') from None
                            self.report['replayed_rows_this_run']+=1
                            if replayed['ordinal']!=expected['ordinal'] or replayed['raw_hash']!=expected['raw_hash']:
                                raise DissertationError('Pinned native stream replay changed; resume refused.')
                    item=next(stream)
                except StopIteration:
                    # Exhaustion during replay indicates changed/incomplete pinned source content.
                    if stream is None:raise DissertationError('Native stream replay failed.')
                    with db:_save(db,'exhausted:'+alias,True)
                    break
                except (Exception,KeyboardInterrupt) as exc:
                    if stream is not None and hasattr(stream,'close'):stream.close()
                    stream=None
                    if isinstance(exc,(DissertationError,KeyboardInterrupt)):raise
                    failures+=1;self.report['stream_failures_this_run']+=1
                    self.progress(f'  {alias}: stream retry {failures}/{self.settings.pipeline3.max_attempts}: {error_category(exc)}')
                    if failures>=self.settings.pipeline3.max_attempts:raise DissertationError(alias+':stream_retries_exhausted') from None
                    time.sleep(min(.25*2**(failures-1),1));continue
                failures=0
                entry={'ordinal':item['ordinal'],'raw_hash':item['raw_hash'],'accepted':False,'reason':None,
                       'id':None,'token_count':None,'rendered_token_hash':None,'transformations':[]}
                with db:
                    try:
                        if item['rejection']:raise InvalidExample(item['rejection'])
                        record=validate_native_record(item['raw'],source,item['ordinal'],self.settings,self.tokenizer,self.max_tokens)
                        rid=self.gate.accept_training(record)
                        entry.update(accepted=True,id=record['id'],token_count=record['metadata']['preparation']['token_count'],
                            rendered_token_hash=record['metadata']['preparation'].get('rendered_token_hash'),gate_rid=rid,
                            transformations=record['metadata']['provenance'].get('transformations',[]))
                    except (InvalidExample,RejectedRecord) as exc:entry['reason']=str(exc)
                    except (ValueError,TypeError,KeyError):entry['reason']='invalid_native_schema'
                    data=json.dumps(entry,ensure_ascii=False)
                    db.execute('INSERT INTO native_decisions VALUES (?,?,?,?)',(alias,processed,data,digest(entry)))
                    _save(db,'counts',dict(self.gate.counts))
                processed+=1;accepted+=int(entry['accepted'])
                if processed==1 or processed%100==0 or accepted==source['quota']:
                    self.progress(f'  {alias}/{source["config"]}: examined {processed}; accepted {accepted}/{source["quota"]}')
        finally:
            if stream is not None and hasattr(stream,'close'):stream.close()

    def process(self,state):
        self.progress('[3/4] Streaming, adapting, filtering and checkpointing native rows')
        for source in self.sources:
            try:self.process_source(source)
            except Exception as exc:
                self.gate.counts=Counter(_load(self.gate.db,'counts',{}))
                self.report['errors'].append(str(exc) if isinstance(exc,DissertationError) else source['alias']+':source_failed:'+type(exc).__name__)
        return {'stage':'processed'}

    def finalize(self,state=None):
        self.progress('[4/4] Writing reproducible source plan and actual quality statistics (no NVIDIA JSONL copy)')
        r=self.report;stats=[];reasons=Counter();actions=Counter()
        if self.gate:
            for source in self.sources:
                n=good=0
                for row in self.gate.db.execute('SELECT data FROM native_decisions WHERE source=? ORDER BY position',(source['alias'],)):
                    e=json.loads(row[0]);n+=1;good+=int(e['accepted'])
                    if e['reason']:reasons[e['reason']]+=1
                    actions.update(t['action'] for t in e['transformations'])
                stop='target_met' if good>=source['quota'] else 'source_exhausted' if _load(self.gate.db,'exhausted:'+source['alias']) else 'candidate_limit' if n>=source['scan_limit'] else 'failed_or_interrupted'
                stats.append({k:source[k] for k in ('alias','dataset','config','split','revision','license','license_url','quota','scan_limit')} |
                    {'processed':n,'accepted':good,'rejected':n-good,'stop_reason':stop})
                if source['quota'] and not good:r['errors'].append(source['alias']+':no_accepted_examples')
                elif good<source['quota']:r['warnings'].append(source['alias']+':accepted_target_shortfall')
            r.update(records_processed=sum(s['processed'] for s in stats),accepted_examples=sum(s['accepted'] for s in stats),
                     rejected_examples=sum(s['rejected'] for s in stats),enterprise=self.gate.sources,
                     indexed_counts=_load(self.gate.db,'counts',{}))
        if not r['accepted_examples']:r['errors'].append('no_accepted_external_examples')
        if self.gate:
            # Verify immutable enterprise split bytes again before publishing a usable plan.
            for name,h in r.get('enterprise_split_hashes',{}).items():
                try:unchanged=file_hash(self.enterprise/(name+'.jsonl'))==h
                except OSError:unchanged=False
                if not unchanged:r['errors'].append('enterprise_changed_during_native_preparation')
        r.update(source_statistics=stats,rejection_reasons=dict(reasons),normalization_actions=dict(actions),finished_at=now())
        r['errors']=list(dict.fromkeys(r['errors']));r['warnings']=list(dict.fromkeys(r['warnings']))
        r['data_prepared']=bool(not r['errors'] and r['accepted_examples'] and self.tokenizer and self.tokenizer.info.get('validated'))
        if self.tokenizer and not self.tokenizer.info.get('validated'):r['warnings'].append('actual_tokenizer_not_validated')
        r['status']='FAILED' if r['errors'] else 'COMPLETED_WITH_WARNINGS' if reasons or actions or r['warnings'] else 'COMPLETED'
        plan={'schema_version':VERSION,'run_id':r['run_id'],'status':r['status'],'data_prepared':r['data_prepared'],
            'max_tokens':self.max_tokens,'seed':self.seed,'sources':self.sources,'adapter_code_hashes':getattr(self,'code',{}),'versions':r['versions'],
            'enterprise_split_hashes':r.get('enterprise_split_hashes',{}),'tokenizer':self.tokenizer.info if self.tokenizer else None,
            'normalization':self.settings.pipeline3.model_dump(),'quality':self.settings.pipeline4.model_dump(),
            'accepted_examples':r['accepted_examples'],'selection_scope':'only committed accepted source coordinates; not entire upstream datasets',
            'semantic_status':'not_semantically_certified','training_started':False}
        if self.gate:
            plan['decisions_digest']=decision_digest(self.gate.db)
            self.gate.close();self.gate=None
            plan['decision_database']=str((self.work/'fingerprints.sqlite3').relative_to(self.output))
            plan['decision_database_hash']=file_hash(self.work/'fingerprints.sqlite3')
        dump_json(self.output/'source_plan.json',plan)
        dump_json(self.output/'report.json',r)
        return {'stage':'reported'}


def run_native_datasets(enterprise,output,settings,*,max_samples=10000,max_tokens=4096,selected=('nemotron','opencode'),
                        seed=42,allocation=None,max_records_per_dataset=None,configs=None,revisions=None,
                        tokenizer=None,provider=None,progress=lambda text:None,research_report=None):
    selected=tuple(selected);enterprise=Path(enterprise).resolve();output=Path(output).resolve()
    if not selected or len(set(selected))!=len(selected) or any(a not in DATASETS for a in selected):raise DissertationError('Select unique supported NVIDIA datasets.')
    if (type(max_samples) is not int or type(max_tokens) is not int or max_samples<1 or max_tokens<1 or
        (max_records_per_dataset is not None and (type(max_records_per_dataset) is not int or max_records_per_dataset<1))):
        raise DissertationError('Native preparation limits must be positive integers.')
    if type(seed) is not int or any(type(n) is not int for n in (allocation or {}).values()):
        raise DissertationError('Native seed/allocations must be integers.')
    if any(a not in selected for mapping in (allocation or {},configs or {},revisions or {}) for a in mapping):raise DissertationError('Native settings refer to an unselected dataset.')
    if output==enterprise or output.is_relative_to(enterprise) or enterprise.is_relative_to(output):raise DissertationError('Native output must not overlap enterprise data.')
    if (output/'report.json').exists():
        try:existing=json.loads((output/'report.json').read_text())
        except (OSError,ValueError):raise DissertationError('Existing native output report is unreadable.') from None
        if existing.get('schema_version') != VERSION:raise DissertationError('Native output must be a new or existing native-plan directory.')
    if any((output/name).exists() for name in ('training.jsonl','train.jsonl','validation.jsonl','test.jsonl')):
        raise DissertationError('Native output cannot replace existing dataset files.')
    with output_lock(output):
        engine=NativePipeline(enterprise,output,settings,max_samples,selected,seed,allocation,max_records_per_dataset,configs,revisions,tokenizer,provider,progress)
        engine.max_tokens=max_tokens
        dump_json(output/'report.json',engine.report)
        graph=StateGraph(NativeState)
        for name in ('discover','process','finalize'):graph.add_node(name,getattr(engine,name))
        for a,b in zip((START,'discover','process','finalize'),('discover','process','finalize',END)):graph.add_edge(a,b)
        try:graph.compile().invoke({'stage':'starting'})
        except (Exception,KeyboardInterrupt) as exc:
            engine.report['errors'].append(str(exc) if isinstance(exc,DissertationError) else 'native_stage_failed:'+type(exc).__name__)
            try:engine.finalize()
            except Exception:
                if engine.gate:engine.gate.close();engine.gate=None
                engine.report.update(status='FAILED',data_prepared=False,finished_at=now())
                engine.report['errors'].append('native_report_or_checkpoint_failed')
                dump_json(output/'report.json',engine.report)
                dump_json(output/'source_plan.json',{'schema_version':VERSION,'status':'FAILED','data_prepared':False})
        if research_report:
            path=Path(research_report)
            r=engine.report;marker=r['run_id'] or 'failed-discovery';start,end=f'<!-- native-nvidia:{marker} -->',f'<!-- /native-nvidia:{marker} -->'
            block=f"{start}\n## Native NVIDIA preparation execution — {r['finished_at']}\n\nStatus: {r['status']}; examined {r['records_processed']}; accepted {r['accepted_examples']}; rejected {r['rejected_examples']}; decoded rows this invocation {r['decoded_rows_read_this_run']}; resumed decisions {r['resumed_decisions']}. No model training or external test execution. Sources: {json.dumps(r['source_statistics'])}. Quality reasons: {json.dumps(r['rejection_reasons'])}. This is selection-scoped structural/token/leakage evidence, not semantic certification.\n{end}"
            try:
                text=path.read_text() if path.exists() else '# Research Report\n'
                if start in text and end in text:
                    a,b=text.index(start),text.index(end)+len(end);text=text[:a]+block+text[b:]
                else:text=text.rstrip()+'\n\n'+block+'\n'
                atomic_text(path,text)
            except OSError:
                engine.report.update(status='FAILED',data_prepared=False)
                engine.report['errors'].append('research_report_update_failed')
                dump_json(output/'report.json',engine.report)
                plan=json.loads((output/'source_plan.json').read_text())
                plan.update(status='FAILED',data_prepared=False)
                dump_json(output/'source_plan.json',plan)
        return engine.report
