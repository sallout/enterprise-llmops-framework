"""LangGraph repository workflow; state carries paths, never a repository checkout."""
from collections import Counter
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
from typing import TypedDict
from zoneinfo import ZoneInfo

from langgraph.graph import END, START, StateGraph

from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.data_io import atomic_text, digest, dump_json, dump_jsonl, load_cache, save_cache
from policy_dissertation.pipelines.policy_transformation.generation import ModelCallError
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
from policy_dissertation.pipelines.policy_transformation.training_validation import duplicate_key
from .discovery import index_repositories, sha
from .llm import CodeGateway, GENERATE, REVIEW, CORRECT
from .sandbox import DockerSandbox
from .schemas import VERSION, TASK_TYPES, CodeEnvelope, CodeTrainingRecord, CodeUnit
from .validation import model_context, validate_and_verify


class CodeState(TypedDict, total=False):
    stage: str
    manifest_path: str
    report_path: str


def update_report(path: Path, report: dict):
    start, end = f"<!-- pipeline2-run:{report['run_id']} -->", f"<!-- /pipeline2-run:{report['run_id']} -->"
    text = path.read_text(encoding='utf-8') if path.exists() else '# Dissertation Project Report\n'
    block = f'''{start}
## Pipeline 2 execution — {report['finished_at']}

- Status: **{report['status']}**; repositories indexed: {report['repositories_discovered']}; selected units: {report['units_selected']}.
- Failed units: {report['failed_units']}; pending units: {report['pending_units']}.
- Generated examples: {report['examples_generated']}; accepted: {report['accepted_examples']}; rejected/quarantined: {report['rejected_examples']}.
- Accepted runtime-verified: {report['accepted_verified']}; accepted nonexecuting knowledge tasks: {report['accepted_unverified_knowledge']}.
- Rejected/quarantined verification statuses: {json.dumps(report['rejected_by_verification_status'])}.
- Source languages: {json.dumps(report['languages'])}; generated task types: {json.dumps(report['generated_by_type'])}.
- Sandbox: {report['sandbox']['reason']}; runtime test invocations confirmed this run: {report['execution_attempts_this_run']}.
- LLM calls this run: {report['llm_calls_this_run']}; resumed generation/validation units: {report['resumed_generation_units']}/{report['resumed_validation_units']}.
- Report: `{report['output_directory']}/report.json`.
- Syntax checks and model review do not establish runtime correctness. Unexecuted coding examples are quarantined; nonexecuting knowledge tasks are explicitly unverified for runtime.
- No repository or generated code was executed directly on the host. Errors: {json.dumps(report['errors'])}.

No Pipeline 3 or model fine-tuning was performed.
{end}'''
    if start in text and end in text:
        a, b = text.index(start), text.index(end)+len(end)
        text = text[:a]+block+text[b:]
    else:
        text = text.rstrip()+'\n\n'+block+'\n'
    atomic_text(path, text)


class RepositoryPipeline:
    def __init__(self, source, output, settings, max_tokens, limit_units, task_types, model, sandbox, progress, report_path):
        self.source, self.output, self.settings = source, output, settings
        self.max_tokens, self.limit_units, self.task_types = max_tokens, limit_units, task_types
        self.model, self.progress, self.report_path = model, progress, report_path
        self.sandbox = sandbox if sandbox is not None else DockerSandbox(settings.pipeline2)
        self.gateway, self.work = None, None
        self.report = {
            'schema_version': VERSION, 'run_id': 'not_indexed', 'status': 'RUNNING',
            'input_directory': str(source), 'output_directory': str(output),
            'started_at': datetime.now(ZoneInfo('Asia/Dubai')).isoformat(),
            'repositories_discovered': 0, 'repositories_selected': 0, 'units_selected': 0,
            'examples_generated': 0, 'accepted_examples': 0, 'rejected_examples': 0,
            'accepted_verified': 0, 'accepted_unverified_knowledge': 0, 'failed_units': 0,
            'pending_units': 0,
            'resumed_generation_units': 0, 'resumed_validation_units': 0,
            'execution_attempts_this_run': 0, 'execution_requests_this_run': 0, 'tests_run_this_run': 0,
            'llm_calls_this_run': 0,
            'limit_units': limit_units, 'chunk_max_tokens': max_tokens,
            'token_method': 'utf8_bytes_upper_bound_v1', 'requested_task_types': list(task_types),
            'sandbox': {'available': self.sandbox.available, 'reason': self.sandbox.reason,
                        'image_id': self.sandbox.image_id},
            'generated_by_type': {}, 'accepted_by_type': {}, 'rejected_by_type': {},
            'rejected_by_verification_status': {}, 'languages': {}, 'source_inventory': {}, 'errors': [],
            'semantic_status': 'automated_checks_not_human_verified',
            'versions': {p: version(p) for p in ('langgraph', 'langchain-openai', 'tree-sitter', 'tree-sitter-kotlin', 'tree-sitter-typescript')},
        }

    def index(self, state):
        self.progress('[1/4] Discovering repositories and statically parsing source/context')
        secret = self.settings.model.api_key.get_secret_value() if self.settings.model.api_key else ''
        scan = self.output/'_index'
        manifest = index_repositories(self.source, scan, self.max_tokens,
                                      max_file_bytes=self.settings.pipeline2.max_file_bytes,
                                      limit_units=self.limit_units, api_key=secret)
        code_hash = digest({str(p.relative_to(Path(__file__).parents[2])): sha(p.read_bytes())
                            for p in [*Path(__file__).parent.glob('*.py'),
                                      Path(__file__).parents[1]/'policy_transformation/generation.py']})
        key = digest({'version': VERSION, 'inventory': manifest, 'source': str(self.source),
                      'config': self.settings.pipeline2.model_dump(), 'model': self.settings.model.model_name,
                      'endpoint_hash': digest(self.settings.model.base_url), 'task_types': self.task_types,
                      'code_hash': code_hash, 'prompt_hash': digest([GENERATE, REVIEW, CORRECT]),
                      'sandbox': self.report['sandbox'], 'versions': self.report['versions']})
        self.report['run_id'] = key
        self.work = self.output/'_runs'/key
        self.work.mkdir(parents=True, exist_ok=True)
        atomic_text(self.work/'index/units.jsonl', (scan/'units.jsonl').read_text(encoding='utf-8'))
        dump_json(self.work/'index/manifest.json', manifest)
        self.report['repositories_discovered'] = len(manifest['repositories'])
        self.report['repositories_selected'] = len({unit.repository_id for unit in self.units()})
        self.report['units_selected'] = manifest['counts']['selected_units']
        self.report['source_inventory'] = manifest
        self.report['languages'] = dict(Counter(unit.language for unit in self.units()))
        if manifest['counts'].get('unreadable_files') or manifest['counts'].get('unreadable_directories'):
            self.report['errors'].append('source_discovery_incomplete')
        if not self.report['units_selected']:
            raise DissertationError('No eligible source units within the chunk limit were found.')
        self.progress(f"  Indexed {self.report['repositories_discovered']} repositories; selected {self.report['units_selected']} units")
        return {'stage': 'indexed', 'manifest_path': str(self.work/'index/manifest.json')}

    def units(self):
        with (self.work/'index/units.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                yield CodeUnit.model_validate_json(line)

    def task(self, index):
        return self.task_types[index % len(self.task_types)]

    def key(self, unit, task):
        return digest([self.report['run_id'], unit.model_dump(), task])

    def generate(self, state):
        self.progress('[2/4] Generating source-grounded software-engineering tasks')
        self.gateway = CodeGateway(self.settings, self.model)
        self.gateway.on_retry = self.progress
        endpoint_failed = False
        for index, unit in enumerate(self.units()):
            task = self.task(index)
            path = self.work/'generation'/f'{unit.unit_id}.json'
            key = self.key(unit, task)
            cached = load_cache(path, key)
            if cached and cached.get('success'):
                self.report['resumed_generation_units'] += 1
                self.verify_unit(index, unit)
                continue
            self.progress(f'  Unit {index+1}/{self.report["units_selected"]}: {unit.language}, {task}')
            data = {'success': False, 'error': 'not_attempted_after_endpoint_failure', 'task_type': task}
            if not endpoint_failed:
                try:
                    envelope = self.gateway.generate_code({'requested_task_type': task, 'context': model_context(unit)})
                    data = {'success': True, 'task_type': task, 'envelope': envelope.model_dump()}
                except ModelCallError as exc:
                    data['error'] = exc.kind
                    data['diagnostics'] = exc.diagnostics
                    endpoint_failed = exc.kind.startswith('endpoint_')
            save_cache(path, key, data)
            if data['success']:
                self.verify_unit(index, unit)
        return {'stage': 'generated'}

    def verify(self, state):
        self.progress('[3/4] Per-unit validation checkpoints completed during generation')
        return {'stage': 'verified'}

    def verify_unit(self, index, unit):
        task = self.task(index)
        generated = load_cache(self.work/'generation'/f'{unit.unit_id}.json', self.key(unit, task))
        if not generated or not generated.get('success'):
            return
        key = digest([self.key(unit, task), generated])
        path = self.work/'validation'/f'{unit.unit_id}.json'
        cached = load_cache(path, key)
        if cached and cached.get('complete'):
            try:
                if cached.get('accepted'):
                    CodeTrainingRecord.model_validate(cached['accepted'])
                self.report['resumed_validation_units'] += 1
                return
            except ValueError:
                pass
        candidate = CodeEnvelope.model_validate(generated['envelope']).examples[0]
        self.progress(f'  Validating unit {index+1}/{self.report["units_selected"]}: {task}')
        outcome = validate_and_verify(unit, candidate, task, self.gateway, self.sandbox,
                                      self.settings.pipeline2.max_corrections)
        self.report['execution_requests_this_run'] += len(outcome['trajectory'])
        actual_runs = [step['execution'] for step in outcome['trajectory']
                       if step['execution']['executor'] == 'docker' and step['execution']['tests_run'] > 0]
        self.report['execution_attempts_this_run'] += len(actual_runs)
        self.report['tests_run_this_run'] += sum(step['tests_run'] for step in actual_runs)
        save_cache(path, key, outcome)

    def export(self, state=None):
        self.progress('[4/4] Publishing training data, rejections/quarantine, and report')
        accepted, rejected, seen = [], [], set()
        generated_counts, accepted_counts, rejected_counts, statuses = Counter(), Counter(), Counter(), Counter()
        for name in ('examples_generated', 'accepted_verified', 'accepted_unverified_knowledge', 'failed_units', 'pending_units'):
            self.report[name] = 0
        if self.work and (self.work/'index/units.jsonl').exists():
            for index, unit in enumerate(self.units()):
                task = self.task(index)
                generated = load_cache(self.work/'generation'/f'{unit.unit_id}.json', self.key(unit, task))
                if not generated or not generated.get('success'):
                    self.report['failed_units' if generated and generated.get('error') != 'not_attempted_after_endpoint_failure'
                                else 'pending_units'] += 1
                    rejected.append({'kind': 'unit_failure', 'unit_id': unit.unit_id, 'task_type': task,
                                     'repository_id': unit.repository_id, 'verification_status': 'unverified',
                                     'diagnostics': (generated or {}).get('diagnostics', []),
                                     'reasons': [(generated or {}).get('error', 'generation_incomplete')]})
                    continue
                self.report['examples_generated'] += len(generated['envelope']['examples'])
                generated_counts[task] += len(generated['envelope']['examples'])
                key = digest([self.key(unit, task), generated])
                outcome = load_cache(self.work/'validation'/f'{unit.unit_id}.json', key)
                if not outcome:
                    self.report['pending_units'] += 1
                    rejected.append({'kind': 'example', 'unit_id': unit.unit_id, 'task_type': task,
                                     'candidate': generated['envelope']['examples'][0], 'verification_status': 'unverified',
                                     'disposition': 'quarantined', 'reasons': ['validation_incomplete']})
                    rejected_counts[task] += 1
                    statuses['unverified'] += 1
                    continue
                if not outcome['complete']:
                    self.report['failed_units'] += 1
                if outcome.get('rejected'):
                    rejection = outcome['rejected']
                    rejection['task_type'] = task
                    rejected.append(rejection)
                    rejected_counts[task] += 1
                    statuses[outcome['verification_status']] += 1
                elif outcome.get('accepted'):
                    record = CodeTrainingRecord.model_validate(outcome['accepted'])
                    pair = duplicate_key(record)
                    if pair in seen:
                        rejected.append({'kind': 'example', 'unit_id': unit.unit_id, 'task_type': task,
                                         'verification_status': outcome['verification_status'], 'disposition': 'rejected',
                                         'candidate': record.model_dump(), 'reasons': ['duplicate_question_answer']})
                        rejected_counts[task] += 1
                        statuses[outcome['verification_status']] += 1
                    else:
                        seen.add(pair)
                        record.metadata['duplicate_check'] = 'passed'
                        accepted.append(record.model_dump())
                        accepted_counts[task] += 1
                        self.report['accepted_verified' if outcome['verification_status'] == 'verified'
                                    else 'accepted_unverified_knowledge'] += 1
        self.report['accepted_examples'] = len(accepted)
        self.report['rejected_examples'] = sum(r['kind'] == 'example' for r in rejected)
        self.report['rejection_record_count'] = len(rejected)
        self.report['quarantined_examples'] = sum(r.get('disposition') == 'quarantined' for r in rejected)
        self.report['generated_by_type'], self.report['accepted_by_type'] = dict(generated_counts), dict(accepted_counts)
        self.report['rejected_by_type'], self.report['rejected_by_verification_status'] = dict(rejected_counts), dict(statuses)
        self.report['corrected_accepted_examples'] = sum(len(x['metadata']['trajectory']) > 1 for x in accepted)
        self.report['llm_calls_this_run'] = self.gateway.calls if self.gateway else 0
        if self.report['pending_units']:
            self.report['errors'].append('unit_processing_pending')
        if self.report['failed_units']:
            self.report['errors'].append('mandatory_unit_processing_incomplete')
        if not accepted:
            self.report['errors'].append('no_accepted_training_examples')
        self.report['errors'] = list(dict.fromkeys(self.report['errors']))
        self.report['status'] = 'FAILED' if self.report['errors'] else (
            'COMPLETED_WITH_WARNINGS' if rejected or self.report['accepted_unverified_knowledge'] or
            not self.sandbox.available or self.report['source_inventory'].get('counts', {}).get('oversized_symbols')
            else 'COMPLETED')
        self.report['finished_at'] = datetime.now(ZoneInfo('Asia/Dubai')).isoformat()
        self.report['rejection_reasons'] = dict(Counter(reason for r in rejected for reason in r['reasons']))
        from policy_dissertation.data.adapters import adapt
        from policy_dissertation.data.schemas import VERSION as EXPORT_VERSION
        accepted = [adapt(item, 'pipeline2', {'pipeline': 'pipeline2', 'stage': 'validated_export'}) for item in accepted]
        self.report['training_schema_version'] = EXPORT_VERSION
        dump_jsonl(self.output/'training.jsonl', accepted)
        dump_jsonl(self.output/'rejected.jsonl', rejected)
        self.report['output_hashes'] = {name: sha((self.output/name).read_bytes()) for name in ('training.jsonl', 'rejected.jsonl')}
        try:
            update_report(self.report_path, self.report)
            self.report['research_report_updated'] = True
        except OSError:
            self.report['errors'].append('research_report_update_failed')
            self.report['status'] = 'FAILED'
            self.report['research_report_updated'] = False
        dump_json(self.output/'report.json', self.report)
        dump_json(self.output/'manifest.json', {'schema_version': EXPORT_VERSION, 'status': self.report['status'],
            'accepted_examples': len(accepted), 'output_hashes': self.report['output_hashes'],
            'license': 'unspecified; publisher must confirm redistribution rights', 'publication': None})
        if self.work:
            dump_json(self.work/'last_execution_report.json', self.report)
        return {'stage': 'exported', 'report_path': str(self.output/'report.json')}

    def graph(self):
        graph = StateGraph(CodeState)
        nodes = ['index', 'generate', 'verify', 'export']
        for name in nodes:
            graph.add_node(name, getattr(self, name))
        for a, b in zip([START, *nodes], [*nodes, END]):
            graph.add_edge(a, b)
        return graph.compile()


def run_pipeline2(input_directory: Path, output_directory: Path, settings, max_tokens=2048, *,
                  limit_units: int | None = None, task_types=TASK_TYPES, model=None, sandbox=None,
                  progress=lambda text: None, research_report: Path | None = None, overwrite=False) -> dict:
    source, output = input_directory.resolve(), output_directory.resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise DissertationError('Input and output directories must not overlap.')
    if max_tokens < 1 or (limit_units is not None and limit_units < 1):
        raise DissertationError('Chunk/unit limits must be positive.')
    if not task_types or any(t not in TASK_TYPES for t in task_types):
        raise DissertationError('Unsupported task types. Use: '+', '.join(TASK_TYPES))
    with output_lock(output):
        from policy_dissertation.data.hub import protect_exports
        protect_exports(output, overwrite)
        engine = RepositoryPipeline(source, output, settings, max_tokens, limit_units, tuple(task_types), model,
                                    sandbox, progress, research_report or Path('report.md').resolve())
        dump_json(output/'report.json', engine.report)
        try:
            engine.graph().invoke({'stage': 'starting'})
        except (Exception, KeyboardInterrupt) as exc:
            engine.report['errors'].append(str(exc) if isinstance(exc, DissertationError) else 'pipeline_stage_failed:'+type(exc).__name__)
            engine.export()
        return engine.report
