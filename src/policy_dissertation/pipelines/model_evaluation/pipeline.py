"""Frozen suite → sequential CUDA generation → isolated tests → auditable comparison."""
import json
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from huggingface_hub import HfApi
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
from policy_dissertation.pipelines.policy_transformation.data_io import digest, dump_json, dump_jsonl, load_cache, save_cache
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
from .suites import read_suite
from .backend import CudaGenerator
from .execution import DockerExecutor
from .scoring import score, summarize
from .contamination import audit

VERSION = 'qwen3-model-evaluation-v1'


def model_identity(source, settings):
    path = Path(source)
    if path.is_dir():
        path = path.resolve()
        if not (path/'config.json').is_file():
            raise DissertationError('Model directory must contain config.json, not the compression run root.')
        files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*')) if p.is_file()}
        if not any(name.endswith('.safetensors') for name in files):
            raise DissertationError('Local model has no safetensors weights.')
        return {'source': str(path), 'revision': None, 'files': files, 'identity': digest(files)}
    if source != 'Qwen/Qwen3-14B':
        raise DissertationError('Use a local Qwen3 checkpoint or the official Qwen/Qwen3-14B baseline.')
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    requested = settings.pipeline4.tokenizer_revision
    pin = HfApi(token=token).model_info(source, revision=requested).sha
    return {'source': source, 'revision': pin, 'identity': pin}


def compression_models(root, intermediates=False):
    root = Path(root).resolve()
    report = json.loads((root/'report.json').read_text())
    if not report.get('final_model_ready') or report.get('status') not in ('COMPLETED', 'COMPLETED_WITH_WARNINGS'):
        raise DissertationError('Compression run must have a completed final model.')
    models = {'original_bf16': 'Qwen/Qwen3-14B'}
    if intermediates:
        fingerprint = report['fingerprint']
        if not re.fullmatch('[0-9a-f]{64}', fingerprint):
            raise DissertationError('Invalid compression stage fingerprint.')
        models['pruned_bf16'] = str(root/'_runs'/fingerprint/'pruned')
        models['recovered_bf16'] = str(root/'_runs'/fingerprint/'recovery'/'merged')
    models['final_nf4'] = str(root/'model_4bit')
    return models


def run_evaluation(suite_path, output, models, settings, config, *, resume=False,
                   training_tokens=None, progress=print, backend=None, executor=None, identity_provider=None):
    suite_path = Path(suite_path).resolve(); output = Path(output).resolve()
    payload, cases = read_suite(suite_path)
    if not models or any(not re.fullmatch('[a-zA-Z0-9_-]{1,64}', label) for label in models):
        raise DissertationError('Provide unique simple model labels, such as original and final.')
    for source in models.values():
        p = Path(source).resolve()
        if Path(source).is_dir() and (output.is_relative_to(p) or p.is_relative_to(output)):
            raise DissertationError('Evaluation output must be separate from model directories.')
    if training_tokens and (output == Path(training_tokens).resolve() or Path(training_tokens).resolve().is_relative_to(output)):
        raise DissertationError('Training cache must be outside the evaluation output.')
    backend = backend or CudaGenerator(settings)
    executor = executor or (DockerExecutor(config) if config.executor == 'docker' else None)
    report = {'schema_version': VERSION, 'status': 'RUNNING', 'started_at': datetime.now(timezone.utc).isoformat(),
              'models': {}, 'errors': [], 'warnings': [], 'evaluation_complete': False,
              'suite_digest': payload['digest'], 'configuration': config.model_dump(),
              'limitations': payload['limitations'] + ['Benchmark selected after training; freeze before model answer inspection.',
                  'Small samples have wide uncertainty; do not tune repeatedly on a final test set.',
                  'Quality tests are not a security audit.'], 'semantic_enterprise_correctness_verified': False}
    if any(c.suite == 'enterprise' for c in cases): report['warnings'].append('enterprise_semantic_review_pending')
    if not training_tokens: report['warnings'].append('training_overlap_not_checked')
    if config.executor == 'disabled': report['warnings'].append('code_execution_disabled_no_pass_at_1')
    with output_lock(output):
        if (output/'report.json').exists() and not resume:
            raise DissertationError('Evaluation output exists. Use --resume for identical inputs or a new directory.')
        dump_json(output/'report.json', report)
        try:
            progress('[1/4] Freezing model identities, suite, runtime and execution environment')
            identities = {k: (identity_provider or model_identity)(v, settings) for k, v in models.items()}
            runtime = backend.preflight()
            execution = executor.preflight() if executor and any(c.suite != 'enterprise' for c in cases) else {'executor': 'disabled'}
            frozen = {'version': VERSION, 'suite_digest': payload['digest'], 'models': identities,
                'configuration': config.model_dump(), 'runtime': runtime, 'execution': execution,
                'training_tokens_hash': file_hash(training_tokens) if training_tokens else None,
                'implementation': {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}}
            fingerprint = digest(frozen); manifest = output/'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text()).get('fingerprint') != fingerprint:
                raise DissertationError('Evaluation inputs/settings/runtime changed; use a new output directory.')
            dump_json(manifest, frozen | {'fingerprint': fingerprint})
            dump_json(output/'suite.json', payload)
            report.update(fingerprint=fingerprint, runtime=runtime, execution=execution)
            selected = list(cases); all_rows = []
            if executor:
                progress('[2/4] Checking reference implementations against selected benchmark tests')
                for case in cases:
                    if case.suite == 'enterprise': continue
                    cache = output/'_checks'/(digest(case.model_dump())+'.json')
                    key = digest({'case': case.model_dump(), 'execution': execution})
                    checked = load_cache(cache, key)
                    if checked is None:
                        checked = executor.run(case.reference, case.tests)
                        if checked['status'] != 'passed':
                            raise DissertationError('Reference benchmark test failed for '+case.id+'; inspect image dependencies/adapter before scoring models.')
                        save_cache(cache, key, checked)
            for label, identity in identities.items():
                progress(f'[3/4] Loading {label}; models run sequentially')
                info = backend.load(identity['source'], identity['revision'], config)
                try:
                    if report['models']:
                        first = next(iter(report['models'].values()))['load']
                        if first.get('chat_template_hash') != info.get('chat_template_hash'):
                            raise DissertationError('Compared checkpoint chat templates differ; use a common verified template before comparing results.')
                    if len(report['models']) == 0:
                        if training_tokens:
                            audit_key = digest({'fingerprint': fingerprint, 'audit': VERSION})
                            audit_path = output/'overlap.json'
                            overlap = load_cache(audit_path, audit_key)
                            if overlap is None:
                                overlap = audit(cases, Path(training_tokens), backend.tokenizer, progress)
                                save_cache(audit_path, audit_key, overlap)
                            report['contamination'] = overlap
                            selected = [c for c in cases if c.id not in overlap['excluded']]
                            if overlap['excluded']: report['warnings'].append('suspected_training_overlap_excluded')
                        else: report['contamination'] = {'status': 'NOT_CHECKED', 'excluded': {}}
                    if not selected: raise DissertationError('No evaluation tasks remain after overlap exclusion.')
                    # One unscored warmup per model; no target answer or tests supplied.
                    backend.generate([{'role': 'user', 'content': 'Return only the word ready.'}], config.model_copy(update={'max_new_tokens': 16}))
                    rows = []
                    for i, case in enumerate(selected, 1):
                        key = digest({'run': fingerprint, 'model': label, 'case': case.model_dump()})
                        cache = output/'_checkpoints'/label/(digest(case.id)+'.json')
                        row = load_cache(cache, key)
                        if row is None:
                            generated = backend.generate(case.messages, config)
                            row = {'model': label, 'case_id': case.id, 'suite': case.suite, 'source': case.source,
                                   'generation': generated, 'score': score(case, generated, executor)}
                            if row['score']['status'] == 'unverified' and config.executor != 'disabled':
                                raise DissertationError('Code executor unavailable during evaluation; resume after repairing it.')
                            save_cache(cache, key, row)
                        rows.append(row)
                        if i == 1 or i % 10 == 0 or i == len(selected): progress(f'  {label}: {i}/{len(selected)} tasks')
                    summary = summarize(rows)
                    report['models'][label] = {'identity': identity, 'load': info, 'metrics': summary}
                    all_rows.extend(rows)
                    dump_jsonl(output/'results.jsonl', all_rows)
                    dump_jsonl(output/'enterprise_review.jsonl', [r | {'messages': next(c.messages for c in selected if c.id == r['case_id']),
                        'reference': next(c.reference for c in selected if c.id == r['case_id']),
                        'review': {'correctness': None, 'conditions_and_exceptions': None, 'unsupported_claims': None}}
                        for r in all_rows if r['suite'] == 'enterprise'])
                    dump_json(output/'report.json', report)
                finally: backend.release()
            progress('[4/4] Exporting matched comparison and review records')
            comparison = comparisons(report['models'])
            report.update(comparisons=comparison, selected_tasks=len(cases), evaluated_tasks_per_model=len(selected),
                status='COMPLETED_WITH_WARNINGS' if report['warnings'] else 'COMPLETED', evaluation_complete=config.executor != 'disabled' or not any(c.suite != 'enterprise' for c in cases))
            report['output_hashes'] = {n: file_hash(output/n) for n in ('suite.json', 'results.jsonl', 'enterprise_review.jsonl')}
            return report
        except BaseException as exc:
            report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED'
            report['errors'].append(str(exc) if isinstance(exc, DissertationError) else 'evaluation_stage_failed:'+type(exc).__name__)
            raise DissertationError('Evaluation did not complete. Inspect report.json; completed task checkpoints are preserved.') from None
        finally:
            report['finished_at'] = datetime.now(timezone.utc).isoformat()
            dump_json(output/'report.json', report)
            backend.release()


def score_saved(input_directory, output, config, *, executor=None, progress=print):
    """Score saved GPU answers on a Docker host without loading model weights."""
    root = Path(input_directory).resolve(); output = Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise DissertationError('Offline scoring output must be separate from the generation input.')
    if config.executor != 'docker':
        raise DissertationError('Offline executable scoring requires Docker.')
    executor = executor or DockerExecutor(config)
    producer = json.loads((root/'report.json').read_text())
    if producer.get('schema_version') != VERSION or producer.get('status') not in ('COMPLETED', 'COMPLETED_WITH_WARNINGS'):
        raise DissertationError('Only a completed generation export can be scored offline.')
    for name in ('suite.json', 'results.jsonl'):
        if producer.get('output_hashes', {}).get(name) != file_hash(root/name):
            raise DissertationError('Generation output checksum mismatch; offline scoring refused.')
    payload, cases = read_suite(root/'suite.json')
    by_id = {c.id: c for c in cases if c.id not in producer.get('contamination', {}).get('excluded', {})}
    rows = [json.loads(line) for line in (root/'results.jsonl').read_text().splitlines() if line.strip()]
    expected = {(label, case_id) for label in producer['models'] for case_id in by_id}
    if {(r['model'], r['case_id']) for r in rows} != expected or len(rows) != len(expected):
        raise DissertationError('Saved answers are incomplete or duplicated.')
    with output_lock(output):
        if (output/'report.json').exists():
            raise DissertationError('Offline scoring output exists; use a new directory.')
        report = {k: v for k, v in producer.items() if k not in ('output_hashes', 'errors', 'comparisons')}
        report.update(status='RUNNING', errors=[], evaluation_complete=False,
                      generation_report_hash=file_hash(root/'report.json'), scoring_configuration=config.model_dump(), scoring_started_at=datetime.now(timezone.utc).isoformat())
        try:
            execution = executor.preflight(); report['execution'] = execution
            for case in by_id.values():
                if case.suite != 'enterprise' and executor.run(case.reference, case.tests)['status'] != 'passed':
                    raise DissertationError('Reference test failed during offline scoring: '+case.id)
            for i, row in enumerate(rows, 1):
                case = by_id[row['case_id']]
                if row['suite'] != case.suite:
                    raise DissertationError('Saved answer suite mismatch.')
                row['score'] = score(case, row['generation'], executor)
                if row['score']['status'] == 'unverified':
                    raise DissertationError('Docker execution failed during offline scoring.')
                if i == 1 or i % 10 == 0 or i == len(rows): progress(f'Scored saved answers: {i}/{len(rows)}')
            for label in report['models']:
                report['models'][label]['metrics'] = summarize([r for r in rows if r['model'] == label])
            report['comparisons'] = comparisons(report['models'])
            report['warnings'] = [w for w in report['warnings'] if w != 'code_execution_disabled_no_pass_at_1']
            report.update(status='COMPLETED_WITH_WARNINGS' if report['warnings'] else 'COMPLETED', evaluation_complete=True)
            dump_json(output/'suite.json', payload); dump_jsonl(output/'results.jsonl', rows)
            # The original review sheet and generation timings remain available in the input export.
            report['enterprise_review_source'] = str(root/'enterprise_review.jsonl')
            report['output_hashes'] = {n: file_hash(output/n) for n in ('suite.json', 'results.jsonl')}
            return report
        except BaseException as exc:
            report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED'
            report['errors'].append(str(exc) if isinstance(exc, DissertationError) else 'offline_scoring_failed:'+type(exc).__name__)
            raise DissertationError('Offline scoring did not complete; inspect report.json. Generated answers are preserved.') from None
        finally:
            report['scoring_finished_at'] = datetime.now(timezone.utc).isoformat()
            dump_json(output/'report.json', report)


def comparisons(models):
    labels = list(models); baseline = labels[0]; result = {}
    for label in labels[1:]:
        result[label] = {}
        for suite in ('humaneval', 'mbpp'):
            a = models[baseline]['metrics'].get(suite, {}).get('pass_at_1')
            b = models[label]['metrics'].get(suite, {}).get('pass_at_1')
            if a is not None and b is not None:
                result[label][suite] = {'baseline': baseline, 'delta_percentage_points': 100*(b-a),
                    'note': 'Descriptive matched-selection difference, not a statistical significance claim.'}
    return result
