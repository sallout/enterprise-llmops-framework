"""End-to-end LangGraph runs with explicit mocked LLM responses, never real API calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.pipelines.policy_transformation.data_schemas import TrainingRecord
from policy_dissertation.pipelines.policy_transformation.generation import JsonGateway
from policy_dissertation.pipelines.policy_transformation.pipeline import run_pipeline


class MockModel:
    def __init__(self, mode='valid'):
        self.mode = mode
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        # Retry feedback follows the JSON payload; decode just the JSON object.
        payload, _ = json.JSONDecoder().raw_decode(messages[1].content)
        if self.mode == 'endpoint_error':
            raise RuntimeError('test-secret must never appear in outputs')
        if self.mode == 'malformed' or (self.mode == 'retry' and self.calls == 1):
            return SimpleNamespace(content='not JSON')
        if 'passages' in payload:
            passage = payload['passages'][0]
            candidate = {'instruction': f"What requirements apply in the {payload['section_title']} section of this policy?",
                         'response': passage['text'], 'document_id': payload['document_id'],
                         'chunk_id': payload['chunk_id'], 'passage_id': passage['passage_id']}
            if self.mode == 'unsupported_answer':
                candidate['response'] = 'Employees MAY disclose information without any approval.'
            if self.mode == 'bad_reference':
                candidate['document_id'] = 'wrong-document'
            if self.mode == 'bad_schema':
                candidate['instruction'] = 123
            candidates = [candidate, dict(candidate)] if self.mode == 'duplicate' else [candidate]
            return SimpleNamespace(content=json.dumps({'examples': candidates}))
        verdict = {'grounded': True, 'answers_question': True, 'preserves_conditions': True,
                   'scope_correct': True, 'reason': 'The complete source clause supports the question.'}
        if self.mode == 'not_grounded':
            verdict['grounded'] = False
        if self.mode == 'bad_judge':
            return SimpleNamespace(content='broken verdict')
        return SimpleNamespace(content=json.dumps(verdict))


@pytest.fixture
def settings():
    return Settings(_env_file=None, model={'base_url': 'https://example.invalid/v1',
                                         'api_key': 'test-secret', 'model_name': 'mock-model'},
                    pipeline1={'max_attempts': 2, 'request_timeout': 1})


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'policies'
    path.mkdir()
    (path / 'policy.md').write_text(
        '# Test Access Policy\n\n## 1. Access\n\n'
        '- Employees MUST obtain Legal approval before disclosure unless the DPO grants an exception.\n\n'
        '## 2. Retention\n\n'
        '- Records MUST NOT be retained beyond 30 days unless Legal issues a hold.\n', encoding='utf-8')
    return path


def run(source, output, settings, model, tmp_path, **kwargs):
    return run_pipeline(source, output, settings, model=model, research_report=tmp_path/'research.md', **kwargs)


def test_full_langgraph_workflow_and_resume(source, settings, tmp_path):
    output = tmp_path / 'output'
    model = MockModel()
    original = (source/'policy.md').read_bytes()
    report = run(source, output, settings, model, tmp_path)
    assert report['status'] == 'COMPLETED'
    assert report['documents_processed'] == 1
    assert report['chunks_generated'] == report['examples_generated'] == report['accepted_examples'] == 2
    assert report['rejected_examples'] == report['failed_chunks'] == 0
    assert model.calls == 4
    records = [TrainingRecord.model_validate_json(line) for line in (output/'training.jsonl').read_text().splitlines()]
    assert len(records) == 2
    assert [m.role for m in records[0].messages] == ['system', 'user', 'assistant']
    assert 'unless the DPO grants an exception' in records[0].messages[2].content
    assert records[0].metadata['source_content_hash']
    assert records[0].metadata['source_spans']
    assert (source/'policy.md').read_bytes() == original
    first_output = (output/'training.jsonl').read_bytes()
    resumed = run(source, output, settings, MockModel('endpoint_error'), tmp_path)
    assert resumed['status'] == 'COMPLETED'
    assert resumed['llm_calls_this_run'] == 0
    assert resumed['resumed_generation_chunks'] == resumed['resumed_validation_chunks'] == 2
    assert (output/'training.jsonl').read_bytes() == first_output
    assert (tmp_path/'research.md').read_text().count('<!-- pipeline1-run:') == 1


def test_bounded_json_retry(source, settings, tmp_path):
    model = MockModel('retry')
    report = run(source, tmp_path/'out', settings, model, tmp_path)
    assert report['accepted_examples'] == 2
    assert model.calls == 5


@pytest.mark.parametrize('mode,reason', [
    ('unsupported_answer', 'answer_not_complete_verbatim_passage'),
    ('bad_reference', 'invalid_source_reference'),
    ('bad_schema', 'invalid_example_schema'),
    ('not_grounded', 'failed_grounded'),
])
def test_required_validation_gates(source, settings, tmp_path, mode, reason):
    output = tmp_path/'out'
    report = run(source, output, settings, MockModel(mode), tmp_path)
    assert report['status'] == 'FAILED'  # all records failed required gates
    assert report['accepted_examples'] == 0
    assert report['rejected_examples'] == 2
    assert report['rejection_reasons'][reason] == 2
    assert (output/'training.jsonl').read_text() == ''


def test_global_duplicate_rejection_partial_success(source, settings, tmp_path):
    report = run(source, tmp_path/'out', settings, MockModel('duplicate'), tmp_path)
    assert report['status'] == 'COMPLETED_WITH_WARNINGS'
    assert report['examples_generated'] == 4
    assert report['accepted_examples'] == report['rejected_examples'] == 2
    assert report['rejection_reasons']['duplicate_question_answer'] == 2


def test_transport_error_circuit_breaker_and_secret_suppression(source, settings, tmp_path):
    output = tmp_path/'out'
    model = MockModel('endpoint_error')
    report = run(source, output, settings, model, tmp_path)
    assert report['status'] == 'FAILED'
    assert model.calls == settings.pipeline1.max_attempts
    assert report['failed_chunks'] == 2
    assert report['examples_generated'] == report['rejected_examples'] == 0
    assert report['rejection_record_count'] == 2
    for path in output.rglob('*.json*'):
        assert 'test-secret' not in path.read_text()


def test_malformed_response_exhaustion_and_recovery(source, settings, tmp_path):
    output = tmp_path/'out'
    report = run(source, output, settings, MockModel('malformed'), tmp_path)
    assert report['status'] == 'FAILED'
    assert report['failed_chunks'] == 2
    recovered = run(source, output, settings, MockModel(), tmp_path)
    assert recovered['status'] == 'COMPLETED'
    assert recovered['accepted_examples'] == 2


def test_judge_failure_fails_closed(source, settings, tmp_path):
    report = run(source, tmp_path/'out', settings, MockModel('bad_judge'), tmp_path)
    assert report['status'] == 'FAILED'
    assert report['accepted_examples'] == 0
    assert report['failed_chunks'] == 2
    assert report['rejection_reasons']['grounding_verification_failed'] == 2


def test_changed_input_invalidates_resume(source, settings, tmp_path):
    output = tmp_path/'out'
    first = run(source, output, settings, MockModel(), tmp_path)
    with (source/'policy.md').open('a') as f:
        f.write('\nAdditional approval is required.\n')
    second = run(source, output, settings, MockModel(), tmp_path)
    assert first['run_id'] != second['run_id']
    assert second['llm_calls_this_run'] == 4
    assert len((output/'training.jsonl').read_text().splitlines()) == 2


def test_corrupt_validation_checkpoint_recomputed(source, settings, tmp_path):
    output = tmp_path/'out'
    first = run(source, output, settings, MockModel(), tmp_path)
    path = next((output/'_runs'/first['run_id']/'validation').glob('*.json'))
    path.write_text('{broken')
    second = run(source, output, settings, MockModel(), tmp_path)
    assert second['llm_calls_this_run'] == 1
    assert second['accepted_examples'] == 2


def test_source_document_failure_is_nonzero_even_with_valid_records(source, settings, tmp_path):
    (source/'bad.pdf').write_text('not a PDF')
    report = run(source, tmp_path/'out', settings, MockModel(), tmp_path)
    assert report['status'] == 'FAILED'
    assert report['documents_failed'] == 1
    assert report['accepted_examples'] == 2


def test_oversized_chunks_never_sent_or_truncated(source, settings, tmp_path):
    model = MockModel()
    report = run(source, tmp_path/'out', settings, model, tmp_path, max_tokens=10)
    assert report['status'] == 'FAILED'
    assert report['failed_chunks'] == 2
    assert report['rejection_reasons']['chunk_exceeds_limit'] == 2
    assert model.calls == 0


def test_cli_runs_every_stage(source, settings, tmp_path, monkeypatch):
    monkeypatch.setattr('policy_dissertation.cli.Settings', lambda: settings)
    monkeypatch.setattr('policy_dissertation.pipelines.policy_transformation.generation.ChatOpenAI', lambda **kwargs: MockModel())
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ['pip1_data', '--input', str(source), '--output', str(tmp_path/'out')])
    assert result.exit_code == 0, result.output
    assert 'Accepted examples: 2' in result.output
    assert 'Status: COMPLETED' in result.output
    assert '[1/5]' in result.output and '[5/5]' in result.output


def test_cli_failure_status(source, settings, tmp_path, monkeypatch):
    monkeypatch.setattr('policy_dissertation.cli.Settings', lambda: settings)
    monkeypatch.setattr('policy_dissertation.pipelines.policy_transformation.generation.ChatOpenAI', lambda **kwargs: MockModel('endpoint_error'))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ['pip1_data', '--input', str(source), '--output', str(tmp_path/'out')])
    assert result.exit_code == 1
    assert 'Status: FAILED' in result.output
    assert 'test-secret' not in result.output


def test_formatting_warnings_do_not_blanket_block_records(source, settings, tmp_path):
    file = source/'policy.md'
    file.write_text(file.read_text().replace('- Employees', 'G Employees'))
    report = run(source, tmp_path/'out', settings, MockModel(), tmp_path)
    assert report['accepted_examples'] == 2
    assert report['status'] == 'COMPLETED_WITH_WARNINGS'


def test_configured_response_budget_is_distinct(source, settings, tmp_path, monkeypatch):
    captured = {}
    def factory(**kwargs):
        captured.update(kwargs)
        return MockModel()
    monkeypatch.setattr('policy_dissertation.pipelines.policy_transformation.generation.ChatOpenAI', factory)
    run_pipeline(source, tmp_path/'out', settings, max_tokens=256, research_report=tmp_path/'research.md')
    assert captured['max_tokens'] == settings.pipeline1.response_max_tokens == 4096
    assert captured['max_retries'] == 0


def test_generator_prompt_exposes_exact_candidate_schema(settings):
    class CapturingModel(MockModel):
        def invoke(self, messages, config=None):
            assert '"instruction"' in messages[0].content
            assert '"passage_id"' in messages[0].content
            assert 'never question' in messages[0].content
            return super().invoke(messages, config)
    gateway = JsonGateway(settings, CapturingModel())
    envelope = gateway.generate({'document_id': 'doc', 'chunk_id': 'chunk', 'section_title': 'Access',
                                 'passages': [{'passage_id': 'p', 'text': 'Access MUST receive Legal approval.'}]})
    assert 'instruction' in envelope.examples[0]


def test_resume_after_interruption(source, settings, tmp_path):
    class InterruptingModel(MockModel):
        def invoke(self, messages, config=None):
            if self.calls == 1:
                raise KeyboardInterrupt()
            return super().invoke(messages, config)
    output = tmp_path/'out'
    interrupted = run(source, output, settings, InterruptingModel(), tmp_path)
    assert interrupted['status'] == 'FAILED'
    assert interrupted['examples_generated'] == interrupted['rejected_examples'] == 1
    resumed = run(source, output, settings, MockModel(), tmp_path)
    assert resumed['status'] == 'COMPLETED'
    assert resumed['resumed_generation_chunks'] == 1
    assert resumed['llm_calls_this_run'] == 3


def test_scope_and_exception_context_in_training_messages(source, settings, tmp_path):
    (source/'policy.md').write_text(
        '# Access Policy\n\n## 1. Scope\n\nThis policy applies to all enterprise employees.\n\n'
        '## 2. Controls\n\n- Employees MUST NOT disclose data unless the CISO approves.\n\n'
        '## 3. Exceptions\n\nExceptions MUST expire within 30 days and require Legal approval.')
    output = tmp_path/'out'
    report = run(source, output, settings, MockModel(), tmp_path)
    assert report['accepted_examples'] == 3
    for line in (output/'training.jsonl').read_text().splitlines():
        record = TrainingRecord.model_validate_json(line)
        assert 'all enterprise employees' in record.messages[0].content
        assert 'Exceptions MUST expire within 30 days' in record.messages[0].content
        assert {r['section_id'] for r in record.metadata['context_references']} == {'1', '3'}


def test_pdf_to_training_end_to_end(settings, tmp_path):
    import pymupdf
    source = tmp_path/'pdfs'
    source.mkdir()
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72,72), '1. Access Policy', fontsize=20)
        page.insert_text((72,120), 'Employees MUST obtain Legal approval before disclosure.', fontsize=11)
        pdf.save(source/'policy.pdf')
    report = run(source, tmp_path/'out', settings, MockModel(), tmp_path)
    assert report['documents_processed'] == 1
    assert report['accepted_examples'] == 1


def test_same_output_directory_is_locked(source, settings, tmp_path):
    from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
    from policy_dissertation.errors import DissertationError
    output = tmp_path/'out'
    with output_lock(output):
        with pytest.raises(DissertationError, match='Another Pipeline 1'):
            run(source, output, settings, MockModel(), tmp_path)


def test_recursive_multi_document_and_global_deduplication(source, settings, tmp_path):
    nested = source/'nested'
    nested.mkdir()
    (nested/'copy.md').write_bytes((source/'policy.md').read_bytes())
    (source/'unsupported.docx').write_text('unsupported')
    report = run(source, tmp_path/'out', settings, MockModel(), tmp_path)
    assert report['documents_discovered'] == 3
    assert report['documents_processed'] == 2
    assert report['unsupported_documents'] == 1
    assert report['examples_generated'] == 4
    assert report['accepted_examples'] == report['rejected_examples'] == 2


def test_document_limit_applies_before_parsing_and_llm_work(source, settings, tmp_path):
    nested = source/'nested'
    nested.mkdir()
    (nested/'second.md').write_text('# Second\n\n## 1. Rules\n\nEmployees MUST retain audit records.')
    (source/'00-unsupported.docx').write_text('unsupported files do not consume the limit')
    output = tmp_path/'out'
    model = MockModel()
    limited = run(source, output, settings, model, tmp_path, limit_documents=1)
    assert limited['status'] == 'COMPLETED_WITH_WARNINGS'
    assert limited['documents_discovered'] == 3
    assert limited['supported_documents_discovered'] == 2
    assert limited['documents_selected'] == limited['documents_processed'] == 1
    assert limited['documents_excluded_by_limit'] == 1
    assert limited['selected_source_paths'] == [str(source/'policy.md')]
    assert limited['excluded_source_paths'] == [str(nested/'second.md')]
    assert limited['chunks_generated'] == limited['accepted_examples'] == 2
    assert model.calls == 4
    repeated = run(source, output, settings, MockModel('endpoint_error'), tmp_path, limit_documents=1)
    assert repeated['run_id'] == limited['run_id']
    assert repeated['llm_calls_this_run'] == 0
    unlimited = run(source, output, settings, MockModel(), tmp_path)
    assert unlimited['documents_processed'] == 2
    assert unlimited['documents_excluded_by_limit'] == 0
    assert unlimited['run_id'] != limited['run_id']


def test_failed_supported_document_consumes_limit(source, settings, tmp_path):
    (source/'00-empty.txt').touch()
    report = run(source, tmp_path/'out', settings, MockModel(), tmp_path, limit_documents=1)
    assert report['status'] == 'FAILED'
    assert report['documents_selected'] == report['documents_failed'] == 1
    assert report['documents_processed'] == report['llm_calls_this_run'] == 0
    assert report['documents_excluded_by_limit'] == 1


def test_document_limit_above_available_count(source, settings, tmp_path):
    report = run(source, tmp_path/'out', settings, MockModel(), tmp_path, limit_documents=10)
    assert report['status'] == 'COMPLETED'
    assert report['documents_processed'] == 1
    assert report['documents_excluded_by_limit'] == 0


def test_cli_document_limit_runs_only_one_supported_document(source, settings, tmp_path, monkeypatch):
    (source/'z-second.txt').write_text('Employees MUST retain audit records.')
    monkeypatch.setattr('policy_dissertation.cli.Settings', lambda: settings)
    monkeypatch.setattr('policy_dissertation.pipelines.policy_transformation.generation.ChatOpenAI', lambda **kwargs: MockModel())
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ['pip1_data', '--input', str(source), '--output', str(tmp_path/'out'), '--limit-documents', '1'])
    assert result.exit_code == 0, result.output
    assert 'Documents processed: 1' in result.output
    assert 'Documents excluded by limit: 1' in result.output
    report = json.loads((tmp_path/'out/report.json').read_text())
    assert report['limit_documents'] == 1
    assert report['documents_selected'] == 1
    assert 'Document limit: 1' in (tmp_path/'report.md').read_text()


@pytest.mark.parametrize('limit', ['0', '-1'])
def test_cli_rejects_nonpositive_document_limit(limit):
    result = CliRunner().invoke(app, ['pip1_data', '--limit-documents', limit])
    assert result.exit_code == 2


def test_python_api_rejects_nonpositive_document_limit(source, settings, tmp_path):
    from policy_dissertation.errors import DissertationError
    with pytest.raises(DissertationError, match='limit_documents must be positive'):
        run(source, tmp_path/'out', settings, MockModel(), tmp_path, limit_documents=0)
