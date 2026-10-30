"""Single-command LangGraph orchestration with durable file-based stage work."""
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
from typing import TypedDict
from zoneinfo import ZoneInfo

from langgraph.graph import END, START, StateGraph

from policy_dissertation.errors import DissertationError
from .chunk_batch import chunk_documents
from .chunk_schemas import PolicyChunk
from .data_io import atomic_text, digest, dump_json, dump_jsonl, load_cache, save_cache
from .data_schemas import DATA_VERSION, GenerationEnvelope, ValidationOutcome
from .generation import GENERATION_PROMPT, VERIFICATION_PROMPT, JsonGateway, ModelCallError, passages
from .ingestion import ingest_documents
from .training_validation import duplicate_key, validate_candidates


class WorkflowState(TypedDict, total=False):
    run_directory: str
    stage: str
    report_path: str


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.pipeline1.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DissertationError('Another Pipeline 1 process is using this output directory.') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_fingerprint(source: Path, settings, max_tokens: int, limit_documents: int | None = None) -> str:
    import hashlib
    entries = []
    for path in sorted(source.rglob('*')):
        if path.is_file() and not path.is_symlink():
            try:
                content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                content_hash = 'unreadable'
            entries.append([str(path.relative_to(source)), content_hash])
    selection = {} if limit_documents is None else {'limit_documents': limit_documents}
    return digest({**selection, 'version': DATA_VERSION, 'source': str(source), 'files': entries,
                   'chunk_limit': max_tokens, 'model': settings.model.model_name,
                   'endpoint_fingerprint': digest(settings.model.base_url),
                   'generation': settings.pipeline1.model_dump(),
                   'prompt_hash': digest([GENERATION_PROMPT, VERIFICATION_PROMPT]),
                   'versions': {p: version(p) for p in ('pymupdf', 'pymupdf4llm', 'langgraph', 'langchain-openai')}})


def update_research_report(path: Path, report: dict) -> None:
    marker = f"<!-- pipeline1-run:{report['run_id']} -->"
    end = f"<!-- /pipeline1-run:{report['run_id']} -->"
    block = f'''{marker}
## Pipeline 1 automated execution — {report['finished_at']}

- Run ID: `{report['run_id']}`; status: **{report['status']}**.
- LangGraph stages: parse → normalize/chunk → generate → validate → export.
- Documents processed: {report['documents_processed']}; chunks: {report['chunks_generated']}.
- Document limit: {report.get('limit_documents')}; excluded by limit: {report.get('documents_excluded_by_limit', 0)}.
- Generated examples: {report['examples_generated']}; accepted: {report['accepted_examples']}; rejected: {report['rejected_examples']}.
- Failed chunks: {report['failed_chunks']}; LLM calls this invocation: {report['llm_calls_this_run']}.
- Validation: Pydantic schema, source identity, complete extractive passage, model-assisted grounding/conditions/scope, global exact question/answer deduplication.
- Accepted means these automated gates passed; semantic/legal correctness and downstream fine-tuning performance remain unverified.
- Output report: `{report['output_directory']}/report.json`.
- Errors: {json.dumps(report['errors'])}.

No Pipeline 2 or model fine-tuning was performed.
{end}'''
    text = path.read_text(encoding='utf-8') if path.exists() else '# Dissertation Project Report\n'
    if marker in text and end in text:
        start = text.index(marker)
        stop = text.index(end, start) + len(end)
        text = text[:start] + block + text[stop:]
    else:
        text = text.rstrip() + '\n\n' + block + '\n'
    atomic_text(path, text)


class Pipeline:
    def __init__(self, source, output, settings, max_tokens, model, progress, research_report, limit_documents=None):
        self.source, self.output, self.settings = source, output, settings
        self.max_tokens, self.model, self.progress = max_tokens, model, progress
        self.limit_documents = limit_documents
        self.research_report = research_report
        self.run_id = run_fingerprint(source, settings, max_tokens, limit_documents)
        self.work = output / '_runs' / self.run_id
        self.work.mkdir(parents=True, exist_ok=True)
        self.gateway = None
        self.report = {
            'schema_version': DATA_VERSION, 'run_id': self.run_id, 'status': 'RUNNING',
            'input_directory': str(source), 'output_directory': str(output),
            'started_at': datetime.now(ZoneInfo('Asia/Dubai')).isoformat(),
            'documents_discovered': 0, 'documents_processed': 0, 'documents_failed': 0,
            'unsupported_documents': 0, 'limit_documents': limit_documents,
            'supported_documents_discovered': 0, 'documents_selected': 0,
            'documents_excluded_by_limit': 0, 'selected_source_paths': [],
            'excluded_source_paths': [], 'chunks_generated': 0, 'examples_generated': 0,
            'accepted_examples': 0, 'rejected_examples': 0, 'failed_chunks': 0,
            'resumed_generation_chunks': 0, 'resumed_validation_chunks': 0,
            'llm_calls_this_run': 0, 'errors': [], 'quality_summary': {},
            'chunk_max_tokens': max_tokens, 'chunk_token_method': 'utf8_bytes_upper_bound_v1',
            'response_max_tokens': settings.pipeline1.response_max_tokens,
            'max_attempts_per_call': settings.pipeline1.max_attempts,
            'generation_model': settings.model.model_name,
            'semantic_validation': 'automated_extractive_and_model_assisted_not_human_verified',
            'versions': {p: version(p) for p in ('langgraph', 'langchain-openai', 'pydantic')},
        }

    def parse(self, state):
        self.progress('[1/5] Discovering and parsing policy documents')
        summary = ingest_documents(self.source, self.work / 'parsed', self.limit_documents)
        self.report.update(documents_discovered=summary.discovered, documents_processed=summary.parsed,
                           documents_failed=summary.failed, unsupported_documents=summary.unsupported,
                           supported_documents_discovered=summary.supported_discovered,
                           documents_selected=summary.selected_documents,
                           documents_excluded_by_limit=summary.excluded_by_limit,
                           selected_source_paths=summary.selected_source_paths,
                           excluded_source_paths=summary.excluded_source_paths)
        if self.limit_documents is not None:
            self.progress(f'  Selected {summary.selected_documents}/{summary.supported_discovered} supported documents; '
                          f'{summary.excluded_by_limit} excluded by limit.')
        if summary.failed or summary.discovery_warnings:
            self.report['errors'].append('document_ingestion_incomplete')
        if not summary.parsed:
            raise DissertationError('No supported documents were parsed successfully.')
        return {'stage': 'parsed'}

    def chunk(self, state):
        self.progress('[2/5] Normalizing and chunking policy content')
        summary = chunk_documents(self.work / 'parsed', self.work / 'chunks', self.max_tokens)
        self.report['chunks_generated'] = summary.chunk_count
        self.report['quality_summary'] = summary.model_dump(exclude={'documents', 'input_directory', 'output_directory'})
        if summary.documents_failed:
            self.report['errors'].append('document_chunking_incomplete')
        if not summary.chunk_count:
            raise DissertationError('No policy chunks were produced.')
        return {'stage': 'chunked'}

    def chunks(self):
        summary = json.loads((self.work / 'chunks/summary.json').read_text(encoding='utf-8'))
        for reference in summary['documents']:
            with (self.work / 'chunks' / reference['chunks_path']).open(encoding='utf-8') as stream:
                for line in stream:
                    yield PolicyChunk.model_validate_json(line)

    def contexts(self):
        context = defaultdict(lambda: {'policy_title': '', 'scope': '', 'exceptions': '', 'references': []})
        for chunk in self.chunks():
            item = context[chunk.document_id]
            item['policy_title'] = chunk.parent_sections[0].title if chunk.parent_sections else chunk.section_title
            if chunk.section_title.casefold() == 'scope' or 'exception' in chunk.section_title.casefold():
                field = 'scope' if chunk.section_title.casefold() == 'scope' else 'exceptions'
                item[field] += ('\n\n' if item[field] else '') + chunk.content
                item['references'].append({'chunk_id': chunk.chunk_id, 'section_id': chunk.section_id,
                                           'page_numbers': chunk.page_numbers, 'document_id': chunk.document_id})
        return context

    def generate(self, state):
        self.progress('[3/5] Generating source-grounded questions and answers')
        self.gateway = JsonGateway(self.settings, self.model)
        contexts = self.contexts()
        endpoint_failed = False
        for index, chunk in enumerate(self.chunks(), 1):
            path = self.work / 'generation' / f'{chunk.chunk_id}.json'
            key = digest([self.run_id, chunk.model_dump()])
            cached = load_cache(path, key)
            if cached and cached.get('success'):
                self.report['resumed_generation_chunks'] += 1
                continue
            self.progress(f"  Generating chunk {index}/{self.report['chunks_generated']}")
            data = {'success': False, 'error': None}
            if chunk.oversized or chunk.token_count > self.max_tokens:
                data['error'] = 'chunk_exceeds_limit'
            elif endpoint_failed:
                data['error'] = 'not_attempted_after_endpoint_failure'
            else:
                try:
                    envelope = self.gateway.generate({
                        'document_id': chunk.document_id, 'chunk_id': chunk.chunk_id,
                        'section_id': chunk.section_id, 'section_title': chunk.section_title,
                        'policy_context': contexts[chunk.document_id], 'passages': passages(chunk),
                    })
                    data = {'success': True, 'envelope': envelope.model_dump()}
                except ModelCallError as exc:
                    data['error'] = exc.kind
                    if exc.kind.startswith('endpoint_'):
                        endpoint_failed = True
            save_cache(path, key, data)
        return {'stage': 'generated'}

    def validate(self, state):
        self.progress('[4/5] Validating schema, evidence, references, and policy grounding')
        contexts = self.contexts()
        for index, chunk in enumerate(self.chunks(), 1):
            key = digest([self.run_id, chunk.model_dump()])
            generated = load_cache(self.work / 'generation' / f'{chunk.chunk_id}.json', key)
            if not generated or not generated.get('success'):
                continue
            path = self.work / 'validation' / f'{chunk.chunk_id}.json'
            cached = load_cache(path, key)
            if cached and cached.get('complete'):
                try:
                    ValidationOutcome.model_validate(cached)
                    self.report['resumed_validation_chunks'] += 1
                    continue
                except ValueError:
                    pass
            self.progress(f"  Validating chunk {index}/{self.report['chunks_generated']}")
            outcome = validate_candidates(chunk, GenerationEnvelope.model_validate(generated['envelope']),
                                          passages(chunk), contexts[chunk.document_id], self.gateway)
            save_cache(path, key, outcome.model_dump())
        return {'stage': 'validated'}

    def export(self, state=None):
        self.progress('[5/5] Exporting accepted records, rejections, and execution report')
        for field in ('examples_generated', 'accepted_examples', 'rejected_examples', 'failed_chunks'):
            self.report[field] = 0
        accepted, rejected, seen = [], [], set()
        # Recompute exports from current-run checkpoints; never append duplicate committed data.
        if (self.work / 'chunks/summary.json').exists():
            for chunk in self.chunks():
                key = digest([self.run_id, chunk.model_dump()])
                generated = load_cache(self.work / 'generation' / f'{chunk.chunk_id}.json', key)
                if not generated or not generated.get('success'):
                    self.report['failed_chunks'] += 1
                    rejected.append({'kind': 'chunk_failure', 'chunk_id': chunk.chunk_id,
                                     'document_id': chunk.document_id,
                                     'reasons': [(generated or {}).get('error') or 'generation_incomplete']})
                    continue
                self.report['examples_generated'] += len(generated['envelope']['examples'])
                cached = load_cache(self.work / 'validation' / f'{chunk.chunk_id}.json', key)
                if not cached:
                    self.report['failed_chunks'] += 1
                    rejected.append({'kind': 'chunk_failure', 'chunk_id': chunk.chunk_id,
                                     'document_id': chunk.document_id, 'reasons': ['validation_incomplete']})
                    for index, candidate in enumerate(generated['envelope']['examples']):
                        rejected.append({'kind': 'example', 'chunk_id': chunk.chunk_id,
                                         'document_id': chunk.document_id, 'candidate_index': index,
                                         'candidate': self.gateway.scrub(candidate) if self.gateway else candidate,
                                         'reasons': ['validation_incomplete']})
                    continue
                outcome = ValidationOutcome.model_validate(cached)
                if not outcome.complete:
                    self.report['failed_chunks'] += 1
                rejected.extend(outcome.rejected)
                for record in outcome.accepted:
                    pair = duplicate_key(record)
                    if pair in seen:
                        rejected.append({'kind': 'example', 'chunk_id': chunk.chunk_id,
                                         'document_id': chunk.document_id, 'example_id': record.example_id,
                                         'candidate': record.model_dump(), 'reasons': ['duplicate_question_answer']})
                    else:
                        seen.add(pair)
                        record.metadata['validation']['duplicate'] = 'passed'
                        accepted.append(record.model_dump())
        self.report['accepted_examples'] = len(accepted)
        self.report['rejected_examples'] = sum(item['kind'] == 'example' for item in rejected)
        self.report['rejection_record_count'] = len(rejected)
        if self.report['failed_chunks']:
            self.report['errors'].append('mandatory_chunk_processing_incomplete')
        if not accepted:
            self.report['errors'].append('no_accepted_training_examples')
        warnings = bool(self.report['rejected_examples'] or self.report['unsupported_documents']
                        or self.report['quality_summary'].get('quality_issue_count'))
        self.report['status'] = 'FAILED' if self.report['errors'] else ('COMPLETED_WITH_WARNINGS' if warnings else 'COMPLETED')
        self.report['llm_calls_this_run'] = self.gateway.calls if self.gateway else 0
        self.report['finished_at'] = datetime.now(ZoneInfo('Asia/Dubai')).isoformat()
        self.report['output_files'] = {'training': 'training.jsonl', 'rejected': 'rejected.jsonl', 'report': 'report.json'}
        self.report['rejection_reasons'] = {}
        for item in rejected:
            for reason in item['reasons']:
                self.report['rejection_reasons'][reason] = self.report['rejection_reasons'].get(reason, 0) + 1
        from policy_dissertation.data.adapters import adapt
        from policy_dissertation.data.schemas import VERSION as EXPORT_VERSION
        accepted = [adapt(item, 'pipeline1', {'pipeline': 'pipeline1', 'stage': 'validated_export'}) for item in accepted]
        self.report['training_schema_version'] = EXPORT_VERSION
        dump_jsonl(self.output / 'training.jsonl', accepted)
        dump_jsonl(self.output / 'rejected.jsonl', rejected)
        import hashlib
        self.report['output_hashes'] = {name: hashlib.sha256((self.output/name).read_bytes()).hexdigest()
                                        for name in ('training.jsonl', 'rejected.jsonl')}
        try:
            update_research_report(self.research_report, self.report)
            self.report['research_report_updated'] = True
        except OSError:
            self.report['errors'].append('research_report_update_failed')
            self.report['status'] = 'FAILED'
            self.report['research_report_updated'] = False
        dump_json(self.output / 'report.json', self.report)
        dump_json(self.output/'manifest.json', {'schema_version': EXPORT_VERSION, 'status': self.report['status'],
            'accepted_examples': len(accepted), 'output_hashes': self.report['output_hashes'],
            'license': 'unspecified; publisher must confirm redistribution rights', 'publication': None})
        return {'stage': 'exported', 'report_path': str(self.output / 'report.json')}

    def graph(self):
        graph = StateGraph(WorkflowState)
        names = ['parse', 'chunk', 'generate', 'validate', 'export']
        for name in names:
            graph.add_node(name, getattr(self, name))
        for before, after in zip([START, *names], [*names, END]):
            graph.add_edge(before, after)
        return graph.compile()


def run_pipeline(input_directory: Path, output_directory: Path, settings, max_tokens=2048,
                 *, limit_documents: int | None = None, model=None, progress=lambda text: None,
                 research_report: Path | None = None, overwrite=False) -> dict:
    source, output = input_directory.resolve(), output_directory.resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise DissertationError('Input and output directories must not overlap.')
    if max_tokens < 1:
        raise DissertationError('max_tokens must be positive.')
    if limit_documents is not None and limit_documents < 1:
        raise DissertationError('limit_documents must be positive when supplied.')
    with output_lock(output):
        from policy_dissertation.data.hub import protect_exports
        protect_exports(output, overwrite)
        engine = Pipeline(source, output, settings, max_tokens, model, progress,
                          research_report or Path('report.md').resolve(), limit_documents)
        dump_json(output / 'report.json', engine.report)
        try:
            engine.graph().invoke({'run_directory': str(engine.work), 'stage': 'starting'})
        except (Exception, KeyboardInterrupt) as exc:
            # Provider exceptions/credentials are never serialized or printed.
            engine.report['errors'].append(str(exc) if isinstance(exc, DissertationError)
                                          else f'pipeline_stage_failed:{type(exc).__name__}')
            engine.export()
        return engine.report
