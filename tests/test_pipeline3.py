"""Small deterministic streams; no live network, LLM, repository code, or model downloads."""
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings, Pipeline3Settings
from policy_dissertation.pipelines.external_datasets.normalization import normalize, RejectedRecord
from policy_dissertation.pipelines.external_datasets.pipeline import run_pipeline3, allocations
from policy_dissertation.pipelines.external_datasets.schemas import ExternalTrainingRecord
from policy_dissertation.pipelines.external_datasets.sources import DATASETS, inspected_source, HubProvider


def chat(i=0):
    return {'messages': [{'role': 'system', 'content': ''}, {'role': 'user', 'content': f'Question {i}?'},
                         {'role': 'assistant', 'content': f'Answer {i}.'}], 'domain': 'chat', 'source': 'synthetic', 'generator': 'fixture'}


def code(i=0):
    return {'id': f'code-{i}', 'input': f'Implement increment helper {i}.', 'output': 'def increment(x):\n    return x + 1',
            'unit_tests': '["assert increment(1) == 2"]', 'tests_execution_status': '["pass"]', 'average_test_score': 1.0,
            'llm_judgement': '{"score": 1}', 'domain': 'generic'}


class FakeProvider:
    def __init__(self, rows=None):
        self.rows = rows or {'nemotron': [chat(i) for i in range(30)], 'opencode': [code(i) for i in range(30)]}
        self.inspections, self.streams = [], []
        self.revision = 'a'*40

    def inspect(self, alias, revision=None):
        self.inspections.append(alias)
        names = ['chat', 'math'] if alias == 'nemotron' else ['train']
        return {'alias': alias, 'dataset': DATASETS[alias], 'revision': revision or self.revision,
                'license': 'cc-by-4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
                'configs': [{'config_name': c, 'data_files': [{'split': 'train', 'path': f'{c}/*'}]} for c in names],
                'dataset_info': [], 'files': [f'{c}/data.jsonl' for c in names], 'gated': False, 'card_text': '# Synthetic dataset card'}

    def stream(self, source):
        self.streams.append(source)
        return iter(copy.deepcopy(self.rows[source['alias']]))


@pytest.fixture
def settings():
    return Settings(_env_file=None, pipeline3={'shuffle_buffer': 1, 'max_attempts': 2})


def source(alias='nemotron'):
    return inspected_source(FakeProvider().inspect(alias), 'chat' if alias == 'nemotron' else 'train', 'train')


def run(tmp_path, settings, provider=None, maximum=4, **kwargs):
    return run_pipeline3(tmp_path/'out', settings, maximum, provider=provider or FakeProvider(),
                         research_report=tmp_path/'report.md', **kwargs)


def records(tmp_path, name='training'):
    return [json.loads(l) for l in (tmp_path/f'out/{name}.jsonl').read_text().splitlines()]


def test_canonical_multiturn_preserves_system_and_original_metadata(settings):
    raw = chat()
    raw['messages'] += [{'role': 'user', 'content': 'Follow up?'}, {'role': 'assistant', 'content': 'Further answer.'}]
    original = copy.deepcopy(raw)
    result = normalize(raw, source(), 3, settings.pipeline3)
    assert raw == original
    assert result['messages'] == raw['messages']
    assert result['metadata']['record_id'] == 'row:3'
    assert result['metadata']['source_fields']['domain'] == 'chat'
    assert ExternalTrainingRecord.model_validate(result).metadata.locally_executed is False


def test_opencode_tests_status_and_numeric_score_are_preserved(settings):
    raw = code()
    r = normalize(raw, source('opencode'), 0, settings.pipeline3)
    assert [m['role'] for m in r['messages']] == ['user', 'assistant']
    assert r['metadata']['record_id_kind'] == 'upstream'
    for k in ('unit_tests', 'tests_execution_status', 'average_test_score', 'llm_judgement'):
        assert r['metadata']['source_fields'][k] == raw[k]
    assert r['metadata']['external_test_results_status'] == 'not_locally_verified'
    assert r['metadata']['locally_executed'] is False


def test_native_tool_calls_and_tool_results_preserved(settings):
    raw = chat()
    raw['tools'] = [{'type': 'function', 'function': {'name': 'lookup'}}]
    raw['messages'][2] = {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'call1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"key":"x"}'}}]}
    raw['messages'] += [{'role': 'tool', 'tool_call_id': 'call1', 'name': 'lookup', 'content': 'result'},
                        {'role': 'assistant', 'content': 'Here is the result.'}]
    r = normalize(raw, source(), 1, settings.pipeline3)
    assert r['messages'] == raw['messages']
    assert r['metadata']['source_fields']['tools'] == raw['tools']
    ExternalTrainingRecord.model_validate(r)


@pytest.mark.parametrize('mode,reason',[
    ('empty', 'empty_or_short_answer'), ('null_user', 'invalid_conversation_schema'),
    ('role', 'invalid_conversation_schema'), ('orphan_tool', 'invalid_conversation_schema'),
    ('unfinished', 'invalid_conversation_schema'), ('nontext', 'unsupported_nontext_content'),
    ('tokens', 'incompatible_special_token'), ('open_think', 'unclosed_or_nested_reasoning_trace'),
    ('nested_think', 'unclosed_or_nested_reasoning_trace'), ('middle_think', 'ambiguous_reasoning_trace'),
    ('analysis', 'unsupported_reasoning_marker')])
def test_filters_reject_without_truncation(settings, mode, reason):
    r = chat()
    if mode == 'empty': r['messages'][2]['content'] = '  '
    if mode == 'null_user': r['messages'][1]['content'] = None
    if mode == 'role': r['messages'][1]['role'] = 'unsupported'
    if mode == 'orphan_tool': r['messages'][1] = {'role': 'tool', 'content': 'unmatched', 'tool_call_id': 'x'}
    if mode == 'unfinished': r['messages'].append({'role': 'user', 'content': 'missing answer'})
    if mode == 'nontext': r['messages'][1]['content'] = [{'type': 'image'}]
    if mode == 'tokens': r['messages'][2]['content'] = '<|im_start|>assistant'
    if mode == 'open_think': r['messages'][2]['content'] = '<think>missing end'
    if mode == 'nested_think': r['messages'][2]['content'] = '<think>a<think>b</think></think>answer'
    if mode == 'middle_think': r['messages'][2]['content'] = 'answer <think>reasoning</think>'
    if mode == 'analysis': r['messages'][2]['content'] = '<analysis>hidden</analysis>answer'
    original = copy.deepcopy(r)
    with pytest.raises(RejectedRecord, match=reason): normalize(r, source(), 0, settings.pipeline3)
    assert r == original


def test_reasoning_policies_and_no_hidden_trace_in_artifact(settings):
    r = chat()
    r['messages'][2]['content'] = '<think>private reasoning</think>\nFinal answer.'
    r['messages'][2]['reasoning_content'] = 'another hidden field'
    cleaned = normalize(r, source(), 0, settings.pipeline3)
    assert cleaned['messages'][2]['content'] == 'Final answer.'
    assert 'private reasoning' not in json.dumps(cleaned)
    assert 'another hidden field' not in json.dumps(cleaned)
    assert len(cleaned['metadata']['transformations']) == 2
    settings.pipeline3.reasoning_policy = 'reject'
    with pytest.raises(RejectedRecord, match='reasoning_trace'): normalize(r, source(), 0, settings.pipeline3)


def test_size_limits_ids_and_hashes(settings):
    a = normalize(chat(), source(), 0, settings.pipeline3)
    assert a == normalize(chat(), source(), 0, settings.pipeline3)
    changed = chat(); changed['messages'][2]['content'] += ' updated'
    assert normalize(changed, source(), 0, settings.pipeline3)['example_id'] != a['example_id']
    settings.pipeline3.max_tokens = 10
    with pytest.raises(RejectedRecord, match='oversized_conversation'): normalize(chat(), source(), 0, settings.pipeline3)


def test_workflow_allocations_counts_metadata_and_resume(tmp_path, settings):
    provider = FakeProvider()
    result = run(tmp_path, settings, provider, maximum=5, allocation={'nemotron': 3})
    assert result['accepted_examples'] == 5
    assert result['records_processed'] == result['records_read_this_run'] == 5
    assert result['allocation'] == {'nemotron': 3, 'opencode': 2}
    assert result['llm_calls'] == result['locally_executed_tests'] == 0
    before = (tmp_path/'out/training.jsonl').read_bytes()
    provider2 = FakeProvider()
    resumed = run(tmp_path, settings, provider2, maximum=5, allocation={'nemotron': 3})
    assert resumed['resumed_decisions'] == 5 and resumed['records_read_this_run'] == 0
    assert not provider2.inspections and not provider2.streams
    assert (tmp_path/'out/training.jsonl').read_bytes() == before
    assert resumed['output_hashes'] == result['output_hashes']
    assert 'Pipeline 3 execution' in (tmp_path/'report.md').read_text()


def test_duplicates_global_across_datasets(tmp_path, settings):
    raw = chat(); raw['messages'] = raw['messages'][1:]
    duplicate = {'id': 'copy', 'input': raw['messages'][0]['content'], 'output': raw['messages'][1]['content']}
    provider = FakeProvider({'nemotron': [raw], 'opencode': [duplicate, code(3)]})
    result = run(tmp_path, settings, provider, maximum=2)
    assert result['accepted_examples'] == 2 and result['rejected_examples'] == 1
    assert result['rejection_reasons'] == {'duplicate_conversation': 1}


def test_exhaustion_and_bounded_candidate_scan(tmp_path, settings):
    bad = chat(); bad['messages'][2]['content'] = ''
    settings.pipeline3.scan_multiplier = 2
    provider = FakeProvider({'nemotron': [bad]*100, 'opencode': [code()]})
    result = run(tmp_path, settings, provider, maximum=4)
    assert result['accepted_examples'] == 1
    assert result['source_statistics'][0]['processed'] == 4
    assert result['source_statistics'][0]['stop_reason'] == 'candidate_limit'
    assert result['source_statistics'][1]['stop_reason'] == 'source_exhausted'
    assert result['status'] == 'COMPLETED_WITH_WARNINGS'


def test_source_record_size_does_not_persist_large_content(tmp_path, settings):
    big = chat(); big['messages'][2]['content'] = 'x'*2000
    settings.pipeline3.max_record_bytes = 1024
    r = run(tmp_path, settings, FakeProvider({'nemotron': [big, chat()], 'opencode': [code()]}))
    assert r['rejection_reasons'] == {'oversized_source_record': 1}
    assert 'x'*100 not in (tmp_path/'out/rejected.jsonl').read_text()


def test_seeded_sampling_reproducible_and_bounded(tmp_path, settings):
    settings.pipeline3.shuffle_buffer = 4
    a = run(tmp_path, settings, maximum=4, seed=7)
    ids = [r['example_id'] for r in records(tmp_path)]
    b = run(tmp_path/'second', settings, maximum=4, seed=7)
    assert ids == [r['example_id'] for r in records(tmp_path/'second')]
    c = run(tmp_path/'third', settings, maximum=4, seed=17)
    assert ids != [r['example_id'] for r in records(tmp_path/'third')]
    assert a['records_read_this_run'] == b['records_read_this_run'] == c['records_read_this_run'] == 10


def test_interruption_resume_keeps_only_committed_decisions(tmp_path, settings):
    class Interrupted(FakeProvider):
        def stream(self, source):
            yield chat()
            raise KeyboardInterrupt()
    r = run(tmp_path, settings, Interrupted(), maximum=3, selected=('nemotron',))
    assert r['status'] == 'FAILED' and r['accepted_examples'] == 1
    resumed = run(tmp_path, settings, maximum=3, selected=('nemotron',))
    assert resumed['accepted_examples'] == 3 and resumed['resumed_decisions'] == 1
    assert resumed['sampling_replay_decisions_this_run'] == 1
    assert len({v['example_id'] for v in records(tmp_path)}) == 3


def test_network_retry_and_no_secret_error_leak(tmp_path, settings):
    class Flaky(FakeProvider):
        count = 0
        def stream(self, source):
            self.count += 1
            if self.count == 1:
                raise ConnectionError('private-token-do-not-log')
            return super().stream(source)
    r = run(tmp_path, settings, Flaky())
    assert r['network_failures_this_run'] == 1 and r['accepted_examples'] == 4
    class Broken(FakeProvider):
        def stream(self, source): raise ConnectionError('private-token-do-not-log')
    r = run(tmp_path/'broken', settings, Broken())
    assert r['status'] == 'FAILED' and r['network_failures_this_run'] == 4
    assert 'private-token' not in json.dumps(r)


def test_pinned_replay_detects_stream_drift(tmp_path, settings):
    class Interrupted(FakeProvider):
        def stream(self, source):
            yield chat()
            raise KeyboardInterrupt()
    run(tmp_path, settings, Interrupted(), maximum=2, selected=('nemotron',))
    provider = FakeProvider({'nemotron': [chat(99), chat(1)]})
    r = run(tmp_path, settings, provider, maximum=2, selected=('nemotron',))
    assert r['status'] == 'FAILED' and r['accepted_examples'] == 1
    assert any('Pinned source stream changed' in e for e in r['errors'])


def test_select_all_configs_and_revision_refresh(tmp_path, settings):
    provider = FakeProvider()
    a = run(tmp_path, settings, provider, maximum=3, selected=('nemotron',), configs={'nemotron': ['all']})
    assert len(a['source_statistics']) == 2
    assert sum(s['quota'] for s in a['sources']) == 3
    b = run(tmp_path, settings, FakeProvider(), maximum=3, selected=('nemotron',),
            configs={'nemotron': ['all']}, revisions={'nemotron': 'b'*40}, refresh=True)
    assert a['run_id'] != b['run_id']
    assert all(s['revision'] == 'b'*40 for s in b['sources'])


@pytest.mark.parametrize('kwargs', [dict(maximum=0), dict(selected=('unknown',)),
    dict(allocation={'nemotron': 6}), dict(configs={'absent': ['train']})])
def test_invalid_selection(tmp_path, settings, kwargs):
    from policy_dissertation.errors import DissertationError
    with pytest.raises(DissertationError): run(tmp_path, settings, **kwargs)


def test_unknown_config_fails_without_data_download(tmp_path, settings):
    provider = FakeProvider()
    r = run(tmp_path, settings, provider, configs={'nemotron': ['not-present']})
    assert r['status'] == 'FAILED' and not provider.streams


def test_cli_mocked_success_and_filter_configuration(tmp_path, settings, monkeypatch):
    import policy_dissertation.pipelines.external_datasets.pipeline as module
    monkeypatch.setattr(module, 'HubProvider', lambda *args: FakeProvider())
    monkeypatch.setattr('policy_dissertation.cli.Settings', lambda: settings)
    monkeypatch.chdir(tmp_path)
    filters = tmp_path/'filters.json'; filters.write_text('{"shuffle_buffer":1,"reasoning_policy":"reject"}')
    r = CliRunner().invoke(app, ['pip3_datasets', '--output', str(tmp_path/'out'), '--max-samples', '3',
        '--allocation', 'nemotron=2', '--allocation', 'opencode=1', '--filter-config', str(filters)])
    assert r.exit_code == 0, r.output
    assert 'Accepted examples: 3/3' in r.output
    assert json.loads((tmp_path/'out/report.json').read_text())['filters']['reasoning_policy'] == 'reject'
    bad = CliRunner().invoke(app, ['pip3_datasets', '--allocation', 'nemotron=invalid'])
    assert bad.exit_code == 1


def test_strict_filter_config():
    with pytest.raises(ValidationError): Pipeline3Settings(typo_filter=True)


def test_packaged_stream_reader_pins_urls_and_disables_implicit_transformations(settings, monkeypatch):
    import policy_dissertation.pipelines.external_datasets.sources as module
    calls = []
    monkeypatch.setattr(module, 'load_dataset', lambda *a, **kw: calls.append((a, kw)) or [chat()])
    next(HubProvider(settings.pipeline3).stream(source()))
    _, options = calls[0]
    assert options['streaming'] is True and options['parse_agent_traces'] is False
    assert options['chunksize'] == 1048576 and options['download_config'].max_retries == 0
    assert all('/'+'a'*40+'/' in u for u in options['data_files']['train'])


def test_one_inaccessible_dataset_preserves_other_and_retries_discovery(tmp_path, settings):
    class Restricted(FakeProvider):
        def inspect(self, alias, revision=None):
            if alias == 'nemotron':
                raise PermissionError('secret-bearing-provider-details')
            return super().inspect(alias, revision)
    r = run(tmp_path, settings, Restricted())
    assert r['status'] == 'FAILED' and r['accepted_examples'] == 2
    assert 'secret-bearing-provider-details' not in json.dumps(r)
    recovery = FakeProvider()
    resumed = run(tmp_path, settings, recovery)
    assert recovery.inspections == ['nemotron']
    assert resumed['accepted_examples'] == 4 and resumed['status'] == 'COMPLETED'


def test_unclosed_and_invalid_tool_calls_cannot_be_accepted(settings):
    for arguments in ('not json', '[]', '{"value":NaN}'):
        raw = chat()
        raw['messages'][2] = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'x', 'type': 'function', 'function': {'name': 'read', 'arguments': arguments}}]}
        with pytest.raises(RejectedRecord, match='invalid_conversation_schema'):
            normalize(raw, source(), 0, settings.pipeline3)


def test_message_extras_preserved_outside_training_text(settings):
    raw = chat(); raw['messages'][2]['upstream_score'] = 0.7
    result = normalize(raw, source(), 0, settings.pipeline3)
    assert 'upstream_score' not in result['messages'][2]
    assert result['metadata']['source_message_metadata']['2']['upstream_score'] == 0.7


def test_special_token_and_reasoning_rules_are_configurable(settings):
    raw = chat(); raw['messages'][2]['content'] = '<analysis>literal example</analysis>'
    settings.pipeline3.unsupported_reasoning_markers = []
    assert normalize(raw, source(), 0, settings.pipeline3)['messages'][2]['content'] == raw['messages'][2]['content']
    settings.pipeline3.forbidden_tokens = ['literal']
    with pytest.raises(RejectedRecord, match='incompatible_special_token'):
        normalize(raw, source(), 0, settings.pipeline3)


def test_stream_resources_closed_on_target_completion(tmp_path, settings):
    class Closing(FakeProvider):
        closed = 0
        def stream(self, source):
            try:
                yield from self.rows[source['alias']]
            finally:
                self.closed += 1
    provider = Closing()
    run(tmp_path, settings, provider)
    assert provider.closed == 2


def test_failed_export_publishes_failed_report(tmp_path, settings, monkeypatch):
    import policy_dissertation.pipelines.external_datasets.pipeline as module
    monkeypatch.setattr(module, 'write_jsonl', lambda *args: (_ for _ in ()).throw(OSError('private path')))
    r = run(tmp_path, settings)
    assert r['status'] == 'FAILED' and r['exports_valid'] is False
    assert 'export_or_checkpoint_validation_failed' in r['errors']
    assert json.loads((tmp_path/'out/report.json').read_text())['status'] == 'FAILED'
    assert 'private path' not in json.dumps(r)


def test_midstream_network_retry_replays_without_duplicate_commit(tmp_path, settings):
    class Midstream(FakeProvider):
        attempts = 0
        def stream(self, source):
            self.attempts += 1
            yield chat()
            if self.attempts == 1:
                raise ConnectionError('private upstream text')
            yield chat(1)
    r = run(tmp_path, settings, Midstream(), maximum=2, selected=('nemotron',))
    assert r['accepted_examples'] == 2 and r['network_failures_this_run'] == 1
    assert r['sampling_replay_decisions_this_run'] == 1
    assert len({item['example_id'] for item in records(tmp_path)}) == 2


def test_nonfinite_source_metadata_is_rejected_without_invalid_json(tmp_path, settings):
    malformed = code(); malformed['average_test_score'] = float('nan')
    r = run(tmp_path, settings, FakeProvider({'nemotron': [chat()], 'opencode': [malformed, code(1)]}), maximum=2)
    assert r['rejection_reasons'] == {'non_json_value': 1}
    assert r['accepted_examples'] == 2
    assert 'NaN' not in (tmp_path/'out/training.jsonl').read_text()
    assert 'NaN' not in (tmp_path/'out/rejected.jsonl').read_text()
