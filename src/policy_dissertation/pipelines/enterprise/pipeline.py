"""Compose existing enterprise branches; preserve source datasets and their evidence."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from policy_dissertation.data.adapters import adapt, InvalidExample
from policy_dissertation.data.hub import protect_exports
from policy_dissertation.data.schemas import VERSION
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.data_io import atomic_text, digest, dump_json
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock, run_pipeline
from policy_dissertation.pipelines.repository_data.pipeline import run_pipeline2
from policy_dissertation.pipelines.repository_data.schemas import TASK_TYPES
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash, write_jsonl, now
from policy_dissertation.pipelines.dataset_preparation.pipeline import bounded_lines, strict_json
from policy_dissertation.pipelines.dataset_preparation.quality import scan_sensitive


class EnterpriseState(TypedDict, total=False):
    stage: str
    report_path: str


def _overlap(a, b):
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def run_enterprise(output_directory, settings, *, documents=None, code=None,
                   documents_dataset=None, code_dataset=None, allow_incomplete_import=False,
                   max_tokens=2048, limit_documents=None, limit_units=None, task_types=TASK_TYPES,
                   overwrite=False, model=None, sandbox=None, progress=lambda text: None,
                   research_report=None):
    """Generate optional branches and/or reuse existing accepted exports read-only.

    Imports are explicit to avoid accidentally including stale datasets. A raw folder
    and a prior accepted dataset are mutually exclusive for the same branch.
    """
    if not any((documents, code, documents_dataset, code_dataset)):
        raise DissertationError('Provide --documents, --code, --documents-dataset or --code-dataset.')
    if (documents and documents_dataset) or (code and code_dataset):
        raise DissertationError('Choose a source folder or an existing dataset per branch, not both.')
    if max_tokens < 1 or any(n is not None and n < 1 for n in (limit_documents, limit_units)):
        raise DissertationError('Token and testing limits must be positive.')
    if not task_types or any(t not in TASK_TYPES for t in task_types):
        raise DissertationError('Unsupported repository task types.')
    output = Path(output_directory).resolve()
    for source in (documents, code, documents_dataset, code_dataset):
        if source:
            path = Path(source).resolve()
            if not path.exists():raise DissertationError('Enterprise input does not exist.')
            if _overlap(output, path):raise DissertationError('Enterprise output must not overlap any input.')
    # Prevent using an existing branch output as the new enterprise destination,
    # even if --overwrite was given. Never silently replace old coding datasets.
    prior = output/'report.json'
    if prior.exists():
        try:previous = strict_json(prior.read_text(encoding='utf-8'))
        except (OSError, ValueError):raise DissertationError('Existing output report is unreadable.') from None
        if previous.get('pipeline') != 'enterprise':
            raise DissertationError('Use a new enterprise output directory; existing non-enterprise exports are protected.')
    with output_lock(output):
        protect_exports(output, overwrite)
        report = {'pipeline': 'enterprise', 'schema_version': VERSION, 'status': 'RUNNING',
                  'started_at': now(), 'output_directory': str(output), 'max_tokens': max_tokens,
                  'branches': {}, 'errors': [], 'warnings': [], 'accepted_examples': 0,
                  'rejected_examples': 0, 'invalid_records': 0, 'duplicates_removed': 0, 'branch_accepted_examples': {},
                  'branch_rejected_examples': {}, 'branch_quarantined_examples': {}, 'rejected_audit_records': 0,
                  'llm_calls_this_run': 0,
                  'source_run_complete': True, 'limits': {'documents': limit_documents, 'code_units': limit_units}}
        entries = []
        def branch(name, folder, dataset):
            progress(f'Enterprise — {name}: '+('reuse accepted dataset' if dataset else 'generate and validate'))
            if dataset:
                path = Path(dataset).resolve()
                if path.is_dir():path = path/'training.jsonl'
                upstream_path = path.parent/'report.json'
                if not path.is_file() or not upstream_path.is_file():
                    raise DissertationError('Imported accepted dataset requires training JSONL and its adjacent report.json.')
                upstream = strict_json(upstream_path.read_text(encoding='utf-8'))
                entry = {'branch': name, 'mode': 'import', 'path': str(path), 'hash': file_hash(path),
                         'report_path': str(upstream_path), 'report_hash': file_hash(upstream_path)}
                if upstream.get('output_hashes', {}).get(path.name) != entry['hash']:
                    raise DissertationError('Imported accepted dataset checksum mismatch.')
                incomplete = upstream.get('status') not in ('COMPLETED','COMPLETED_WITH_WARNINGS') or bool(upstream.get('errors')) or upstream.get('source_run_complete') is False
                if incomplete and not allow_incomplete_import:
                    raise DissertationError('Imported producer is failed/incomplete; use --allow-incomplete-import to reuse its validated accepted subset explicitly.')
                if incomplete:
                    report['warnings'].append(name+':accepted_subset_from_incomplete_producer')
                    report['source_run_complete'] = False
                entry.update(expected_count=upstream.get('accepted_examples'), producer_status=upstream.get('status'),
                             producer_complete=not incomplete)
            else:
                destination = output/'_branches'/name
                kwargs = dict(overwrite=overwrite, model=model, progress=progress,
                              research_report=destination/'research.md')
                if name == 'documents':
                    upstream = run_pipeline(Path(folder), destination, settings, max_tokens,
                                            limit_documents=limit_documents, **kwargs)
                else:
                    upstream = run_pipeline2(Path(folder), destination, settings, max_tokens,
                                             limit_units=limit_units, task_types=task_types, sandbox=sandbox, **kwargs)
                path = destination/'training.jsonl'
                entry = {'branch': name, 'mode': 'generated', 'path': str(path), 'hash': file_hash(path),
                         'report_path': str(destination/'report.json'), 'report_hash': file_hash(destination/'report.json'),
                         'expected_count': upstream.get('accepted_examples'), 'producer_status': upstream.get('status'),
                         'producer_complete': upstream.get('status') in ('COMPLETED','COMPLETED_WITH_WARNINGS') and not upstream.get('errors')}
                if not entry['producer_complete']:
                    report['errors'].append(name+':mandatory_branch_failed_or_incomplete')
                    report['source_run_complete'] = False
                elif upstream.get('status') == 'COMPLETED_WITH_WARNINGS':
                    report['warnings'].append(name+':completed_with_warnings')
            entry['rejected_examples'] = upstream.get('rejected_examples', 0)
            entry['quarantined_examples'] = upstream.get('quarantined_examples', 0)
            entry['llm_calls_this_run'] = upstream.get('llm_calls_this_run', 0) if entry['mode']=='generated' else 0
            report['llm_calls_this_run'] += entry['llm_calls_this_run']
            entry['producer_metrics'] = {k: upstream[k] for k in
                ('documents_processed', 'chunks_generated', 'repositories_discovered', 'units_selected',
                 'failed_units', 'pending_units', 'accepted_verified', 'accepted_unverified_knowledge') if k in upstream}
            report['branch_rejected_examples'][name] = entry['rejected_examples']
            report['branch_quarantined_examples'][name] = entry['quarantined_examples']
            entries.append(entry)
            report['branches'][name] = entry

        def documents_node(state):
            if documents or documents_dataset:
                try:branch('documents', documents, documents_dataset)
                except (Exception, KeyboardInterrupt) as exc:
                    report['errors'].append('documents:'+ (str(exc) if isinstance(exc, DissertationError) else type(exc).__name__))
                    report['source_run_complete'] = False
                    if isinstance(exc, KeyboardInterrupt):raise
            return {'stage': 'documents_processed'}

        def code_node(state):
            if code or code_dataset:
                try:branch('code', code, code_dataset)
                except (Exception, KeyboardInterrupt) as exc:
                    report['errors'].append('code:'+ (str(exc) if isinstance(exc, DissertationError) else type(exc).__name__))
                    report['source_run_complete'] = False
                    if isinstance(exc, KeyboardInterrupt):raise
            return {'stage': 'code_processed'}

        def merge_node(state):
            progress('Enterprise — validating and combining accepted records')
            # A new per-invocation SQLite staging file prevents appending/reusing stale merges.
            db_path = output/'_merge.sqlite3'
            db = sqlite3.connect(db_path)
            db.execute('DROP TABLE IF EXISTS accepted')
            db.execute('CREATE TABLE accepted (id TEXT PRIMARY KEY, pair TEXT UNIQUE, payload TEXT)')
            reasons = Counter(); seen_counts = Counter(); duplicates = 0
            secrets = [v.get_secret_value() for v in (settings.hf_token, settings.model.api_key) if v]
            def audit():
                nonlocal duplicates
                for entry in entries:
                    expected_origin = 'pipeline1' if entry['branch']=='documents' else 'pipeline2'
                    for line, data, raw_hash in bounded_lines(Path(entry['path']), settings.pipeline4.max_line_bytes):
                        seen_counts[entry['branch']] += 1
                        reference = {'enterprise_branch': entry['branch'], 'file_hash': entry['hash'],
                                     'line': line, 'record_hash': raw_hash}
                        try:
                            if data is None:raise InvalidExample('oversized_jsonl_record')
                            scan_sensitive(data.decode('utf-8'), settings.pipeline4, secrets)
                            raw = strict_json(data.decode('utf-8'))
                            record = adapt(raw, expected_origin, reference)
                            if record['metadata']['pipeline'] != expected_origin:
                                raise InvalidExample('wrong_enterprise_branch_origin')
                            record['metadata']['enterprise_import'] = {
                                'branch': entry['branch'], 'mode': entry['mode'], 'file_hash': entry['hash'],
                                'producer_status': entry['producer_status'], 'producer_complete': entry['producer_complete']}
                            pair = digest([record['messages'], record.get('tools')])
                            existing = db.execute('SELECT id,payload FROM accepted WHERE pair=? OR id=?', (pair,record['id'])).fetchone()
                            if existing:
                                previous_record = json.loads(existing[1])
                                if digest([previous_record['messages'],previous_record.get('tools')]) != pair:
                                    raise InvalidExample('conflicting_example_id')
                                duplicates += 1
                                # Preserve duplicate provenance without duplicating model-facing examples.
                                previous_record['metadata'].setdefault('enterprise_duplicate_references', []).append(
                                    {'id': record['id'], 'input_reference': reference,
                                     'provenance': record['metadata']['provenance'], 'verification': record['verification']})
                                db.execute('UPDATE accepted SET payload=? WHERE id=?', (json.dumps(previous_record,ensure_ascii=False), existing[0]))
                                yield {'branch':entry['branch'],'disposition':'duplicate','reasons':['duplicate_conversation'],**reference}
                            else:
                                db.execute('INSERT INTO accepted VALUES (?,?,?)', (record['id'],pair,json.dumps(record,ensure_ascii=False)))
                        except (ValueError,TypeError,KeyError,AttributeError) as exc:
                            reason = str(exc) if isinstance(exc,InvalidExample) else 'invalid_record_or_provenance'
                            reasons[reason] += 1
                            yield {'branch':entry['branch'],'disposition':'rejected','reasons':[reason],**reference}
                    # Preserve rejected/quarantined branch audits separately from merge-invalid counts.
                    if entry['mode']=='generated':
                        rejected_path = Path(entry['path']).parent/'rejected.jsonl'
                        if rejected_path.exists():
                            with rejected_path.open(encoding='utf-8') as rejected_stream:
                                for line in rejected_stream:
                                    item = json.loads(line)
                                    yield {'branch':entry['branch'],'disposition':'branch_rejection','upstream':item}
            try:
                def counted_audit():
                    for item in audit():
                        report['rejected_audit_records'] += 1
                        yield item
                write_jsonl(output/'rejected.jsonl',counted_audit())
                db.commit()
                write_jsonl(output/'training.jsonl',(json.loads(r[0]) for r in db.execute('SELECT payload FROM accepted ORDER BY id')))
                report['accepted_examples'] = db.execute('SELECT count(*) FROM accepted').fetchone()[0]
                report['domain_counts'] = dict(Counter(json.loads(r[0])['domain'] for r in db.execute('SELECT payload FROM accepted')))
                report['verification_counts'] = dict(Counter(json.loads(r[0])['verification']['status'] for r in db.execute('SELECT payload FROM accepted')))
            finally:
                db.close()
            report['branch_accepted_examples'] = dict(seen_counts)
            report['duplicates_removed'] = duplicates
            report['invalid_records'] = sum(reasons.values())
            report['rejected_examples'] = report['invalid_records'] + sum(report['branch_rejected_examples'].values())
            report['rejection_reasons'] = dict(reasons)
            if reasons:report['errors'].append('accepted_export_contains_invalid_records')
            for entry in entries:
                if seen_counts[entry['branch']] != entry['expected_count']:
                    report['errors'].append(entry['branch']+':accepted_count_mismatch')
                if file_hash(Path(entry['path'])) != entry['hash'] or file_hash(Path(entry['report_path'])) != entry['report_hash']:
                    report['errors'].append(entry['branch']+':input_changed_during_merge')
            return {'stage':'merged'}

        graph = StateGraph(EnterpriseState)
        for name, action in (('documents',documents_node),('code',code_node),('merge',merge_node)):
            graph.add_node(name,action)
        for a,b in zip((START,'documents','code','merge'),('documents','code','merge',END)):
            graph.add_edge(a,b)
        try:
            graph.compile().invoke({'stage':'starting'})
        except (Exception,KeyboardInterrupt) as exc:
            report['errors'].append('enterprise_stage_failed:'+type(exc).__name__)
        if not report['accepted_examples']:report['errors'].append('no_accepted_training_examples')
        report['errors'] = list(dict.fromkeys(report['errors']))
        report['status'] = 'FAILED' if report['errors'] else 'COMPLETED_WITH_WARNINGS' if report['warnings'] or report['duplicates_removed'] else 'COMPLETED'
        report['finished_at'] = now()
        report['output_hashes'] = {n:file_hash(output/n) for n in ('training.jsonl','rejected.jsonl') if (output/n).exists()}
        manifest = {'pipeline':'enterprise','schema_version':VERSION,'status':report['status'],
                    'accepted_examples':report['accepted_examples'],'source_run_complete':report['source_run_complete'],
                    'inputs':entries,'output_hashes':report['output_hashes'],
                    'license':'Original per-record source licenses/restrictions apply; no new license inferred.'}
        report_path = Path(research_report) if research_report else Path('report.md')
        try:
            marker = digest([str(output),report['output_hashes']])
            start,end=f'<!-- enterprise-run:{marker} -->',f'<!-- /enterprise-run:{marker} -->'
            block = f"{start}\n## Enterprise dataset execution — {report['finished_at']}\n\nStatus: {report['status']}; accepted: {report['accepted_examples']}; duplicates: {report['duplicates_removed']}; invalid accepted records: {report['invalid_records']}. Source runs complete: {report['source_run_complete']}. Branches: {json.dumps({k:{f:v.get(f) for f in ('mode','expected_count','producer_status')} for k,v in report['branches'].items()})}. No model training.\n{end}"
            text = report_path.read_text() if report_path.exists() else '# Research Report\n'
            if start in text and end in text:
                a,b=text.index(start),text.index(end)+len(end);text=text[:a]+block+text[b:]
            else:text=text.rstrip()+'\n\n'+block+'\n'
            atomic_text(report_path,text)
        except OSError:
            report['errors'].append('research_report_update_failed');report['status']='FAILED'
            manifest['status']='FAILED'
        dump_json(output/'manifest.json',manifest)
        dump_json(output/'report.json',report)
        return report
