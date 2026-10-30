"""LangGraph streaming loop with transactional checkpoints, atomic exports, and reports."""
from collections import Counter
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import time
from typing import TypedDict
from zoneinfo import ZoneInfo
from langgraph.graph import StateGraph, START, END
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.data_io import atomic_text, digest, dump_json, load_cache, save_cache
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
from .normalization import normalize, RejectedRecord
from .sampling import sampled
from .schemas import VERSION, ExternalTrainingRecord
from .sources import DATASETS, DEFAULT_CONFIGS, HubProvider, inspected_source, retry, error_category
from .storage import Store


class DatasetState(TypedDict, total=False):
    stage: str
    run_directory: str
    report_path: str


def now():
    return datetime.now(ZoneInfo('Asia/Dubai')).isoformat()


def allocations(selected, maximum, explicit):
    if any(a not in selected for a in explicit) or any(n < 0 for n in explicit.values()):
        raise DissertationError('Invalid dataset allocation.')
    remaining = maximum - sum(explicit.values())
    missing = [a for a in selected if a not in explicit]
    if remaining < 0 or (not missing and remaining != 0):
        raise DissertationError('Allocations must sum to --max-samples.')
    result = dict(explicit)
    if missing:
        for i, a in enumerate(missing):
            result[a] = remaining // len(missing) + (i < remaining % len(missing))
    return result


def write_jsonl(path, records):
    """Stream exports to a same-directory temporary file, then publish atomically."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def file_hash(path):
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


class DatasetPipeline:
    def __init__(self, output, settings, selected, maximum, seed, allocation, configs, splits, revisions,
                 refresh, provider, progress, report_path):
        self.output, self.settings, self.selected = output, settings, selected
        self.maximum, self.seed, self.allocation = maximum, seed, allocation
        self.configs, self.splits, self.revisions = configs, splits, revisions
        self.refresh, self.provider, self.progress, self.report_path = refresh, provider, progress, report_path
        self.store, self.work, self.sources, self.iterators, self.failed = None, None, [], {}, set()
        self.current, self.normalized, self.rejection = None, None, None
        self.report = {'schema_version': VERSION, 'run_id': 'not_discovered', 'status': 'RUNNING',
            'started_at': now(), 'output_directory': str(output), 'max_samples': maximum,
            'seed': seed, 'allocation': allocation, 'datasets_selected': list(selected), 'filters': settings.model_dump(), 'errors': [],
            'sources': [], 'records_read_this_run': 0, 'decoded_source_bytes_this_run': 0,
            'sampling_replay_decisions_this_run': 0, 'network_failures_this_run': 0,
            'accepted_examples': 0, 'rejected_examples': 0, 'records_processed': 0,
            'resumed_decisions': 0, 'normalization_actions': {}, 'rejection_reasons': {},
            'observed_fields': {}, 'llm_calls': 0, 'locally_executed_tests': 0,
            'downloaded_network_bytes': None,
            'measurement_note': 'records_read counts decoded stream rows including sampler prefetch and retry replay; network/library read-ahead bytes are not measured.',
            'sampling_method': 'seeded_bounded_prefix_buffer_v1; not whole-corpus uniform',
            'validation_scope': 'structural/source checks; external answers/tests not locally verified',
            'versions': {p: version(p) for p in ('datasets', 'huggingface-hub', 'pyarrow', 'langgraph')}}

    def discover(self, state):
        self.progress('[1/6] Inspecting dataset configurations, splits, licenses, and pinned revisions')
        request = digest({'selected': self.selected, 'revisions': self.revisions})
        cache = self.output/'_metadata'/f'{request}.json'
        infos = {} if self.refresh else (load_cache(cache, request) or {})
        for alias in self.selected:
            if alias in infos:
                continue
            self.progress(f'  Inspecting {DATASETS[alias]}')
            try:
                infos[alias] = retry(lambda: self.provider.inspect(alias, self.revisions.get(alias)),
                                     self.settings, self.progress)
                save_cache(cache, request, infos)
            except DissertationError as exc:
                self.report['errors'].append(f'{alias}: {exc}')
        sources = []
        for alias in self.selected:
            if alias not in infos:
                continue
            info = infos[alias]
            if info['dataset'] != DATASETS[alias] or not info.get('revision'):
                raise DissertationError('Invalid dataset inspection metadata.')
            wanted = self.configs.get(alias, DEFAULT_CONFIGS[alias])
            if wanted == ['all']:
                wanted = [c['config_name'] for c in info['configs']]
            if not wanted or len(set(wanted)) != len(wanted):
                raise DissertationError('Configuration selection must be nonempty and unique.')
            for i, config in enumerate(wanted):
                source = inspected_source(info, config, self.splits.get(alias, 'train'))
                source['quota'] = self.allocation[alias] // len(wanted) + (i < self.allocation[alias] % len(wanted))
                source['scan_limit'] = source['quota'] * self.settings.scan_multiplier
                source['key'] = digest(source)
                sources.append(source)
        code = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
        shared = Path(__file__).parents[1]/'policy_transformation/data_schemas.py'
        key = digest({'version': VERSION, 'sources': sources, 'seed': self.seed,
                      'settings': self.settings.model_dump(), 'code': code, 'shared': file_hash(shared),
                      'versions': self.report['versions'],
                      'unified_contract': {p.name:file_hash(p) for p in (Path(__file__).parents[2]/'data').glob('*.py')}})
        self.work, self.sources = self.output/'_runs'/key, sources
        self.work.mkdir(parents=True, exist_ok=True)
        self.report['run_id'] = key
        self.report['dataset_metadata'] = infos
        self.report['sources'] = sources
        dump_json(self.work/'manifest.json', {'sources': sources, 'metadata': infos})
        for alias, info in infos.items():
            atomic_text(self.work/'metadata'/alias/'README.md', info['card_text'])
        self.store = Store(self.work/'decisions.sqlite3')
        previous_report = self.work/'last_execution_report.json'
        if previous_report.exists():
            try:
                previous = json.loads(previous_report.read_text())
                if previous.get('run_id') == key:
                    self.report['observed_fields'] = previous.get('observed_fields', {})
            except (OSError, ValueError):
                pass
        self.report['resumed_decisions'] = sum(self.store.counts(s['key'])[0] for s in sources)
        self.progress(f"  Pinned {len(infos)} datasets; accepted-example cap: {self.maximum}")
        return {'stage': 'discovered', 'run_directory': str(self.work)}

    def make_iterator(self, source):
        key = source['key']
        def on_read(size, raw):
            self.report['records_read_this_run'] += 1
            self.report['decoded_source_bytes_this_run'] += size
            if key not in self.report['observed_fields']:
                self.report['observed_fields'][key] = {k: type(v).__name__ for k, v in raw.items()} if isinstance(raw, dict) else {'record': type(raw).__name__}
        iterator = sampled(self.provider.stream(source), int(digest([self.seed, key])[:16], 16), self.settings, on_read)
        processed, _ = self.store.counts(key)
        for position in range(processed):
            try:
                item = next(iterator)
            except StopIteration:
                raise DissertationError('Pinned source stream changed during checkpoint replay.') from None
            if item['raw_hash'] != self.store.replay_hash(key, position):
                raise DissertationError('Pinned source stream changed during checkpoint replay.')
            self.report['sampling_replay_decisions_this_run'] += 1
        return iterator

    def ingest(self, state):
        self.current = None
        for source in self.sources:
            key = source['key']
            processed, accepted = self.store.counts(key)
            if key in self.failed or accepted >= source['quota'] or processed >= source['scan_limit'] or self.store.exhausted(key):
                continue
            for attempt in range(self.settings.max_attempts):
                try:
                    if key not in self.iterators:
                        self.progress(f"[2/6] Streaming {source['alias']}/{source['config']} ({accepted}/{source['quota']} accepted)")
                        self.iterators[key] = self.make_iterator(source)
                    item = next(self.iterators[key])
                    self.current = (source, processed, item)
                    return {'stage': 'ingested'}
                except StopIteration:
                    self.store.finish(key)
                    self.iterators.pop(key, None)
                    break
                except Exception as exc:
                    self.report['network_failures_this_run'] += 1
                    self.iterators.pop(key, None)
                    category = error_category(exc)
                    self.progress(f"  Stream attempt {attempt+1}/{self.settings.max_attempts}: {category}")
                    if attempt + 1 == self.settings.max_attempts:
                        self.failed.add(key)
                        self.report['errors'].append(f"{source['alias']}/{source['config']}: {category}")
                    else:
                        time.sleep(min(.25*2**attempt, 1))
        return {'stage': 'ingestion_complete'}

    def normalize(self, state):
        source, position, item = self.current
        self.normalized, self.rejection = None, item['rejection']
        if not self.rejection:
            try:
                self.normalized = normalize(item['raw'], source, item['ordinal'], self.settings)
            except RejectedRecord as exc:
                self.rejection = str(exc)
        return {'stage': 'normalized'}

    def filter(self, state):
        source, position, item = self.current
        if self.normalized and not self.rejection:
            from policy_dissertation.data.adapters import adapt, InvalidExample
            try:
                adapt(self.normalized, 'pipeline3', {'stage': 'shared_schema_validation'})
            except (InvalidExample, ValueError, TypeError, KeyError):
                self.normalized = None
                self.rejection = 'incompatible_unified_schema'
        rejection = {'kind': 'external_record', 'dataset': source['dataset'], 'config': source['config'],
                     'split': source['split'], 'revision': source['revision'], 'stream_ordinal': item['ordinal'],
                     'record_id': (str(item['raw']['id']) if isinstance(item['raw'], dict) and
                                   isinstance(item['raw'].get('id'), (str, int)) else f"row:{item['ordinal']}"),
                     'content_hash': item['raw_hash'], 'source_bytes': item['size'],
                     'reasons': [self.rejection] if self.rejection else []}
        content_key = digest(self.normalized['messages']) if self.normalized else None
        self.store.commit(source['key'], position, item, self.normalized, rejection, content_key)
        processed, accepted = self.store.counts(source['key'])
        if processed == 1 or processed % 100 == 0 or accepted == source['quota']:
            self.progress(f"[3–4/6] {source['alias']}/{source['config']}: examined {processed}; accepted {accepted}/{source['quota']}")
        # Release large transient records; graph state contains only stage/path references.
        self.current, self.normalized = None, None
        return {'stage': 'filtered'}

    def export(self, state=None):
        self.progress('[5/6] Exporting canonical conversations and rejected-record references')
        accepted = self.store.rows('accepted') if self.store else iter(())
        rejected = self.store.rows('rejected') if self.store else iter(())
        def checked():
            for record in accepted:
                ExternalTrainingRecord.model_validate(record)
                self.report['accepted_examples'] += 1
                from policy_dissertation.data.adapters import adapt
                yield adapt(record, 'pipeline3', {'pipeline': 'pipeline3', 'stage': 'validated_export'})
        self.report['training_schema_version'] = 'unified-qwen-sft-v1'
        self.report['accepted_examples'] = self.report['rejected_examples'] = 0
        write_jsonl(self.output/'training.jsonl', checked())
        reasons, actions = Counter(), Counter()
        def rejection_rows():
            for record in rejected:
                self.report['rejected_examples'] += 1
                reasons.update(record['reasons'])
                yield record
        write_jsonl(self.output/'rejected.jsonl', rejection_rows())
        if self.store:
            for record in self.store.rows('accepted'):
                actions.update(t['action'] for t in record['metadata']['transformations'])
        self.report['normalization_actions'], self.report['rejection_reasons'] = dict(actions), dict(reasons)
        self.report['records_processed'] = self.report['accepted_examples'] + self.report['rejected_examples']
        counts = []
        for source in (self.sources if self.store else []):
            processed, accepted_count = self.store.counts(source['key'])
            reason = ('target_met' if accepted_count >= source['quota'] else 'stream_error' if source['key'] in self.failed else
                      'source_exhausted' if self.store.exhausted(source['key']) else
                      'candidate_limit' if processed >= source['scan_limit'] else 'interrupted')
            counts.append({k: source[k] for k in ('alias', 'dataset', 'config', 'split', 'revision', 'quota', 'scan_limit')} |
                          {'processed': processed, 'accepted': accepted_count, 'rejected': processed-accepted_count, 'stop_reason': reason})
        self.report['source_statistics'] = counts
        if not self.report['accepted_examples']:
            self.report['errors'].append('no_accepted_examples')
        self.report['status'] = 'FAILED' if self.report['errors'] else ('COMPLETED_WITH_WARNINGS' if
            self.report['rejected_examples'] or actions or self.report['accepted_examples'] < self.maximum else 'COMPLETED')
        self.report['finished_at'] = now()
        self.report['output_hashes'] = {f: file_hash(self.output/f) for f in ('training.jsonl', 'rejected.jsonl')}
        return {'stage': 'exported', 'report_path': str(self.output/'report.json')}

    def report_results(self, state=None):
        self.progress('[6/6] Recording actual processing evidence')
        r = self.report
        start, end = f"<!-- pipeline3-run:{r['run_id']} -->", f"<!-- /pipeline3-run:{r['run_id']} -->"
        block = f'''{start}
## Pipeline 3 execution — {r['finished_at']}

- Status: **{r['status']}**; accepted target/cap: {r['max_samples']}.
- Unique records processed: {r['records_processed']}; accepted: {r['accepted_examples']}; rejected: {r['rejected_examples']}.
- Decoded rows read this invocation (including prefetch/replay): {r['records_read_this_run']}; resumed decisions: {r['resumed_decisions']}.
- Rejection reasons: {json.dumps(r['rejection_reasons'])}; accepted-record transformations: {json.dumps(r['normalization_actions'])}.
- Source statistics: {json.dumps(r.get('source_statistics', []))}.
- Seed: {r['seed']}; bounded buffer: {self.settings.shuffle_buffer}. Revisions and card license snapshots are in report.json and the internal manifest.
- No LLM calls, local code tests, model downloads, dataset mixing, or fine-tuning. External test metadata is not local verification. Network byte totals are not measured.
- Report: `{self.output}/report.json`; errors: {json.dumps(r['errors'])}.
{end}'''
        try:
            text = self.report_path.read_text() if self.report_path.exists() else '# Dissertation Project Report\n'
            if start in text and end in text:
                a, b = text.index(start), text.index(end)+len(end)
                text = text[:a]+block+text[b:]
            else:
                text = text.rstrip()+'\n\n'+block+'\n'
            atomic_text(self.report_path, text)
            r['research_report_updated'] = True
        except OSError:
            r['errors'].append('research_report_update_failed')
            r['status'], r['research_report_updated'] = 'FAILED', False
        dump_json(self.output/'report.json', r)
        if self.work:
            dump_json(self.work/'last_execution_report.json', r)
        return {'stage': 'reported'}

    def graph(self):
        graph = StateGraph(DatasetState)
        for name in ('discover', 'ingest', 'normalize', 'filter', 'export', 'report_results'):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, 'discover')
        graph.add_edge('discover', 'ingest')
        graph.add_conditional_edges('ingest', lambda state: 'normalize' if state['stage'] == 'ingested' else 'export')
        graph.add_edge('normalize', 'filter')
        graph.add_edge('filter', 'ingest')
        graph.add_edge('export', 'report_results')
        graph.add_edge('report_results', END)
        return graph.compile()


def run_pipeline3(output_directory, settings, max_samples=10000, *, selected=('nemotron', 'opencode'),
                  seed=42, allocation=None, configs=None, splits=None, revisions=None, refresh=False,
                  provider=None, progress=lambda message: None, research_report=None, overwrite=False):
    selected = tuple(selected)
    if max_samples < 1 or not selected or len(set(selected)) != len(selected) or any(a not in DATASETS for a in selected):
        raise DissertationError('Select supported datasets and a positive accepted-sample limit.')
    if any(a not in selected for m in (configs or {}, splits or {}, revisions or {}) for a in m):
        raise DissertationError('Configuration options refer to an unselected dataset.')
    allocation = allocations(selected, max_samples, allocation or {})
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    output = Path(output_directory).resolve()
    with output_lock(output):
        from policy_dissertation.data.hub import protect_exports
        protect_exports(output, overwrite)
        engine = DatasetPipeline(output, settings.pipeline3, selected, max_samples, seed, allocation,
            configs or {}, splits or {}, revisions or {}, refresh, provider or HubProvider(settings.pipeline3, token),
            progress, research_report or Path('report.md').resolve())
        dump_json(output/'report.json', engine.report)
        try:
            engine.graph().invoke({'stage': 'starting'}, config={'recursion_limit': max_samples*settings.pipeline3.scan_multiplier*4+100})
        except (Exception, KeyboardInterrupt) as exc:
            category = error_category(exc) if not isinstance(exc, KeyboardInterrupt) else 'interrupted'
            engine.report['errors'].append(category)
            try:
                engine.export()
            except Exception:
                engine.report.update(status='FAILED', finished_at=now(), exports_valid=False)
                engine.report['errors'].append('export_or_checkpoint_validation_failed')
            engine.report_results()
        finally:
            for iterator in engine.iterators.values():
                iterator.close()
            if engine.store:
                engine.store.close()
        return engine.report
