"""Static repository fixtures and mocked LLM/sandbox feedback; no source execution on host."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.pipelines.repository_data.discovery import index_repositories
from policy_dissertation.pipelines.repository_data.parsing import analyze
from policy_dissertation.pipelines.repository_data.pipeline import run_pipeline2
from policy_dissertation.pipelines.repository_data.sandbox import DockerSandbox, execution_preflight
from policy_dissertation.pipelines.repository_data.schemas import CodeTrainingRecord, ExecutionResult, TASK_TYPES

SOLUTION = 'def calculate(value):\n    return value + 1\n'
TESTS = 'import unittest\nfrom solution import calculate\n\nclass CalculationTests(unittest.TestCase):\n    def test_increment(self):\n        self.assertEqual(calculate(2), 3)\n'


class MockCodeModel:
    def __init__(self, mode='valid'):
        self.mode, self.calls = mode, 0
        self.repair_payloads = []

    def invoke(self, messages, config=None):
        self.calls += 1
        data, _ = json.JSONDecoder().raw_decode(messages[1].content)
        if self.mode == 'endpoint_error':
            raise RuntimeError('never expose test-p2-secret')
        if self.mode == 'malformed':
            return SimpleNamespace(content='not json')
        if 'requested_task_type' in data:
            context, kind = data['context'], data['requested_task_type']
            coding = kind not in ('explanation', 'architecture')
            answer = {'task_type': kind, 'instruction': f"Explain the supplied {context['symbol']} function and its input/output contract." if not coding else
                      'Implement a standalone increment helper and verify its behavior with unit tests.',
                      'response': 'The source increments its input by one. The example isolates that behavior and exercises a representative input.',
                      'source_unit_id': context['unit_id'], 'source_quotes': [context['code'].strip()],
                      'language': context['language'], 'solution': SOLUTION if coding else None,
                      'tests': TESTS if coding else None}
            if self.mode == 'invalid_syntax' and coding: answer['solution'] = 'def broken(:'
            if self.mode == 'bad_reference': answer['source_unit_id'] = 'unknown'
            if self.mode == 'bad_quote': answer['source_quotes'] = ['This string does not occur in the source.']
            if self.mode == 'claims_execution': answer['response'] = 'All tests passed and the code was executed successfully.'
            if self.mode == 'initial_bug' and coding: answer['solution'] = SOLUTION.replace('value + 1', 'value - 1')
            return SimpleNamespace(content=json.dumps({'examples': [answer]}))
        if 'actual_execution' in data:
            self.repair_payloads.append(data)
            return SimpleNamespace(content=json.dumps({'solution': SOLUTION, 'correction_summary': 'Replace subtraction with addition to meet the fixed expected behavior.'}))
        verdict = {'grounded': self.mode != 'not_grounded', 'self_contained': True,
                   'technically_sound': True, 'tests_meaningful': self.mode != 'weak_tests',
                   'reason': 'Mocked source review for an explicitly synthetic test fixture.'}
        return SimpleNamespace(content=json.dumps(verdict))


class MockSandbox:
    def __init__(self, results=('verified',), available=True):
        self.available = available
        self.reason = 'mock_sandbox' if available else 'mock_runtime_unavailable'
        self.image_id = 'sha256:'+'a'*64 if available else None
        self.results, self.calls = results, 0

    def run(self, language, solution, tests):
        index = self.calls
        self.calls += 1
        status = self.results[min(index, len(self.results)-1)] if self.available else 'unverified'
        return ExecutionResult(status=status, executor='mock_sandbox',
                               reason='mock_test_feedback' if self.available else self.reason,
                               tests_run=1 if self.available else 0,
                               failures=1 if status == 'failed' else 0,
                               exit_code=0 if status == 'verified' else 1 if status == 'failed' else None,
                               command=['python', '-I', '/work/runner.py'] if self.available else [],
                               feedback='MOCK AssertionError: expected 3, got 1' if status == 'failed' else 'MOCK test successful' if status == 'verified' else '')


@pytest.fixture
def settings():
    return Settings(_env_file=None, model={'base_url':'https://example.invalid/v1', 'api_key':'test-p2-secret', 'model_name':'mock-model'},
                    pipeline2={'max_attempts':1, 'execution_mode':'disabled', 'max_corrections':2})


@pytest.fixture
def repositories(tmp_path):
    root = tmp_path/'repos'
    repo = root/'calculator'
    repo.mkdir(parents=True)
    (repo/'pyproject.toml').write_text('[project]\nname="calculator"\ndependencies=[]\n')
    (repo/'calc.py').write_text('def increment(value):\n    """Increment a numeric value."""\n    return value + 1\n')
    (repo/'tests').mkdir()
    (repo/'tests/test_calc.py').write_text('from calc import increment\n\ndef test_increment():\n    assert increment(2) == 3\n')
    (repo/'LICENSE').write_text('Synthetic fixture license, used only by automated tests.')
    return root


def run(repositories, tmp_path, settings, model=None, sandbox=None, **kwargs):
    return run_pipeline2(repositories, tmp_path/'output', settings, model=model or MockCodeModel(),
                         sandbox=sandbox if sandbox is not None else MockSandbox(),
                         research_report=tmp_path/'research.md', **kwargs)


def test_static_source_index_and_provenance(repositories, tmp_path):
    output=tmp_path/'index'
    report=index_repositories(repositories,output,2048)
    unit=json.loads((output/'units.jsonl').read_text().splitlines()[0])
    assert report['counts']['source_files']==2
    assert unit['symbol']=='increment' and unit['start_line']==1 and unit['end_line']==3
    assert unit['related_tests'] and unit['dependencies'] and unit['licenses']
    assert unit['commit'] is None and unit['snapshot_hash']
    assert unit['code']==(repositories/'calculator/calc.py').read_text()
    assert index_repositories(repositories,output,2048)==report


def test_credentials_dependencies_generated_and_symlink_excluded(repositories,tmp_path):
    repo=repositories/'calculator'
    (repo/'.env').write_text('SECRET=do-not-read')
    (repo/'secret.py').write_text('API_KEY="sk-abcdefghijklmnopqrstuvwxy"\n')
    (repo/'auto.py').write_text('# automatically generated\ndef auto():\n    return 1\n')
    (repo/'node_modules').mkdir()
    (repo/'node_modules/evil.py').write_text('raise RuntimeError("never execute")')
    (repo/'link.py').symlink_to(repo/'calc.py')
    result=index_repositories(repositories,tmp_path/'index',2048)
    units=(tmp_path/'index/units.jsonl').read_text()
    assert 'sk-abcdefghijkl' not in units
    assert 'evil.py' not in units and 'auto.py' not in units and 'link.py' not in units
    assert result['counts']['excluded_sensitive_or_binary']==1


@pytest.mark.parametrize('language,code,suffix',[
    ('kotlin','fun increment(value: Int): Int {\n    return value + 1\n}\n','.kt'),
    ('typescript','export function increment(value: number): number { return value + 1; }','.ts'),
    ('typescript','const increment = (value: number) => value + 1;','.ts'),
    ('python','class Counter:\n    def increment(self, value):\n        return value+1\n','.py'),
])
def test_language_aware_symbols(language,code,suffix):
    parsed=analyze(code,language,suffix)
    assert parsed['valid']
    assert any(s['symbol']=='increment' for s in parsed['symbols'])


def test_invalid_and_oversized_source_is_not_truncated(repositories,tmp_path):
    repo=repositories/'calculator'
    (repo/'invalid.py').write_text('def broken(:\n')
    result=index_repositories(repositories,tmp_path/'index',10)
    assert result['counts']['invalid_symbols']>=1
    assert result['counts']['oversized_symbols']>=1
    assert (tmp_path/'index/units.jsonl').read_text()==''


def test_commit_read_without_running_git(repositories,tmp_path):
    git=repositories/'calculator/.git'
    (git/'refs/heads').mkdir(parents=True)
    (git/'HEAD').write_text('ref: refs/heads/main\n')
    (git/'refs/heads/main').write_text('a'*40+'\n')
    index_repositories(repositories,tmp_path/'index',2048)
    unit=json.loads((tmp_path/'index/units.jsonl').read_text().splitlines()[0])
    assert unit['commit']=='a'*40


def test_full_graph_verified_record_and_resume(repositories,tmp_path,settings):
    model,sandbox=MockCodeModel(),MockSandbox()
    result=run(repositories,tmp_path,settings,model,sandbox,task_types=('implementation',))
    assert result['status']=='COMPLETED'
    assert result['accepted_verified']==result['examples_generated']==1
    assert sandbox.calls==1 and model.calls==2
    record=CodeTrainingRecord.model_validate_json((tmp_path/'output/training.jsonl').read_text())
    assert record.metadata['execution']['executor']=='mock_sandbox'
    assert record.metadata['syntax_valid'] is True
    original=(tmp_path/'output/training.jsonl').read_bytes()
    resumed=run(repositories,tmp_path,settings,MockCodeModel('endpoint_error'),MockSandbox(),task_types=('implementation',))
    assert resumed['llm_calls_this_run']==0
    assert resumed['resumed_generation_units']==resumed['resumed_validation_units']==1
    assert (tmp_path/'output/training.jsonl').read_bytes()==original


def test_bounded_correction_uses_captured_feedback_and_fixed_tests(repositories,tmp_path,settings):
    model,sandbox=MockCodeModel('initial_bug'),MockSandbox(('failed','verified'))
    report=run(repositories,tmp_path,settings,model,sandbox,task_types=('debugging',))
    assert report['corrected_accepted_examples']==1
    assert sandbox.calls==2
    assert model.repair_payloads[0]['actual_execution']['feedback'].startswith('MOCK AssertionError')
    assert model.repair_payloads[0]['fixed_tests']==TESTS
    record=json.loads((tmp_path/'output/training.jsonl').read_text())
    assert len(record['metadata']['trajectory'])==2
    assert 'MOCK AssertionError' in record['messages'][2]['content']
    assert 'Initial implementation' in record['messages'][2]['content']


def test_failed_tests_quarantined_after_limit(repositories,tmp_path,settings):
    sandbox=MockSandbox(('failed',))
    report=run(repositories,tmp_path,settings,MockCodeModel('initial_bug'),sandbox,task_types=('debugging',))
    assert report['status']=='FAILED'
    assert sandbox.calls==3
    assert report['quarantined_examples']==1
    assert report['accepted_examples']==0
    assert report['rejection_reasons']['tests_failed_after_bounded_corrections']==1


def test_unavailable_execution_never_marks_code_verified(repositories,tmp_path,settings):
    report=run(repositories,tmp_path,settings,sandbox=MockSandbox(available=False),task_types=('implementation',))
    assert report['status']=='FAILED'
    assert report['accepted_verified']==0
    assert report['quarantined_examples']==1
    assert report['rejected_by_verification_status']=={'unverified':1}


def test_knowledge_example_allowed_with_explicit_unverified_status(repositories,tmp_path,settings):
    report=run(repositories,tmp_path,settings,sandbox=MockSandbox(available=False),task_types=('explanation',))
    assert report['status']=='COMPLETED_WITH_WARNINGS'
    assert report['accepted_unverified_knowledge']==1 and report['accepted_verified']==0
    record=json.loads((tmp_path/'output/training.jsonl').read_text())
    assert record['metadata']['execution']['reason']=='runtime_not_applicable_to_knowledge_task'
    assert record['metadata']['syntax_valid'] is None


@pytest.mark.parametrize('mode,reason',[
    ('bad_quote','source_evidence_not_exact'),('bad_reference','source_or_task_reference_mismatch'),
    ('invalid_syntax','generated_code_syntax_invalid'),('claims_execution','unsubstantiated_execution_claim'),
    ('not_grounded','failed_grounded'),('weak_tests','failed_tests_meaningful'),
])
def test_validation_rejection(repositories,tmp_path,settings,mode,reason):
    sandbox=MockSandbox()
    report=run(repositories,tmp_path,settings,MockCodeModel(mode),sandbox,task_types=('implementation',))
    assert report['accepted_examples']==0
    assert report['rejection_reasons'][reason]==1
    if mode in ('bad_quote','bad_reference','invalid_syntax','claims_execution'): assert sandbox.calls==0


def test_transport_errors_and_secrets(repositories,tmp_path,settings):
    result=run(repositories,tmp_path,settings,MockCodeModel('endpoint_error'),task_types=('explanation',))
    assert result['status']=='FAILED'
    assert result['failed_units']==1
    for path in (tmp_path/'output').rglob('*.json*'):
        assert 'test-p2-secret' not in path.read_text()


def test_recursive_repositories_and_diverse_tasks(repositories,tmp_path,settings):
    other=repositories/'other'
    other.mkdir()
    (other/'other.py').write_text('def double(value):\n    return value * 2\n')
    result=run(repositories,tmp_path,settings,sandbox=MockSandbox(available=False),limit_units=2,
               task_types=('explanation','architecture'))
    assert result['repositories_discovered']==result['repositories_selected']==2
    assert set(result['generated_by_type'])=={'explanation','architecture'}
    assert result['accepted_examples']==2


def test_global_duplicate_detection(repositories,tmp_path,settings):
    other=repositories/'copy'
    other.mkdir()
    (other/'calc.py').write_bytes((repositories/'calculator/calc.py').read_bytes())
    result=run(repositories,tmp_path,settings,task_types=('explanation',))
    assert result['examples_generated']==2 and result['accepted_examples']==1
    assert result['rejection_reasons']['duplicate_question_answer']==1


def test_changed_source_invalidates_resume(repositories,tmp_path,settings):
    first=run(repositories,tmp_path,settings,task_types=('explanation',))
    file=repositories/'calculator/calc.py'
    file.write_text(file.read_text().replace('value + 1','value + 2'))
    second=run(repositories,tmp_path,settings,task_types=('explanation',))
    assert first['run_id']!=second['run_id']
    assert second['llm_calls_this_run']==2


@pytest.mark.parametrize('solution,tests,reason',[
    ('import subprocess\n'+SOLUTION,TESTS,'unsupported_or_restricted_import'),
    ('def calculate(value):\n    return eval(value)',TESTS,'restricted_builtin'),
    (SOLUTION,TESTS.replace('self.assertEqual(calculate(2), 3)','self.skipTest("skip")'),'skipped_tests_not_allowed'),
    (SOLUTION,TESTS.replace('def test_increment(self):','@unittest.expectedFailure\n    def test_increment(self):'),'skipped_tests_not_allowed'),
    (SOLUTION,'import unittest\nclass Empty(unittest.TestCase):\n    pass','tests_must_import_solution_and_assert_behavior'),
])
def test_execution_preflight(solution,tests,reason):
    assert execution_preflight(solution,tests)==reason


def test_disabled_sandbox_has_no_host_execution(settings,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Subprocess must not run')
    monkeypatch.setattr('policy_dissertation.pipelines.repository_data.sandbox.subprocess.run',forbidden)
    sandbox=DockerSandbox(settings.pipeline2)
    assert sandbox.run('python',SOLUTION,TESTS).status=='unverified'
    assert sandbox.run('kotlin','fun test() {}','fun test() {}').status=='unverified'


def test_cli_single_workflow(repositories,tmp_path,settings,monkeypatch):
    monkeypatch.setattr('policy_dissertation.cli.Settings',lambda:settings)
    monkeypatch.setattr('policy_dissertation.pipelines.policy_transformation.generation.ChatOpenAI',lambda **kwargs:MockCodeModel())
    monkeypatch.chdir(tmp_path)
    result=CliRunner().invoke(app,['pip2_code','--input',str(repositories),'--output',str(tmp_path/'out'),
                                   '--limit-units','1','--task-types','explanation'])
    assert result.exit_code==0,result.output
    assert 'Accepted examples: 1' in result.output
    assert 'Runtime-verified accepted: 0' in result.output
    assert 'COMPLETED_WITH_WARNINGS' in result.output


def test_invalid_task_or_limits_rejected(repositories,tmp_path,settings):
    from policy_dissertation.errors import DissertationError
    with pytest.raises(DissertationError):run(repositories,tmp_path,settings,task_types=('invented',))
    with pytest.raises(DissertationError):run(repositories,tmp_path,settings,limit_units=0)


@pytest.mark.parametrize('task',TASK_TYPES)
def test_every_requested_training_type_supported(repositories,tmp_path,settings,task):
    result=run(repositories,tmp_path,settings,task_types=(task,))
    assert result['generated_by_type']=={task:1}
    assert result['accepted_examples']==1
    assert result['execution_attempts_this_run']==0  # mocked runner is never actual Docker evidence


def test_unavailable_runtime_requests_are_not_execution_counts(repositories,tmp_path,settings):
    result=run(repositories,tmp_path,settings,sandbox=MockSandbox(available=False),task_types=('unit_tests',))
    assert result['execution_requests_this_run']==1
    assert result['execution_attempts_this_run']==result['tests_run_this_run']==0


def test_docker_command_restrictions_and_actual_result_parser(settings,monkeypatch):
    import re
    import subprocess
    from pathlib import Path
    config=settings.pipeline2.model_copy(update={'execution_mode':'auto'})
    calls=[]
    mount_path=None
    def simulated(cmd,**kwargs):
        nonlocal mount_path
        calls.append(cmd)
        if cmd[1]=='info': return subprocess.CompletedProcess(cmd,0,stdout=b'fake daemon')
        if cmd[1]=='image': return subprocess.CompletedProcess(cmd,0,stdout='sha256:'+'a'*64+'\n')
        if cmd[1]=='create':
            mount=cmd[cmd.index('--mount')+1]
            mount_path=Path(mount.split('src=',1)[1].split(',dst=',1)[0])
            assert '--network' in cmd and cmd[cmd.index('--network')+1]=='none'
            assert '--read-only' in cmd and '--cap-drop' in cmd
            assert 'no-new-privileges' in cmd and '65534:65534' in cmd
            assert '--memory' in cmd and '--pids-limit' in cmd
            assert mount.endswith(',readonly')
            assert all(name in {'solution.py','test_solution.py','runner.py'} for name in [p.name for p in mount_path.iterdir()])
        if cmd[1]=='wait': return subprocess.CompletedProcess(cmd,0,stdout='0\n')
        if cmd[1]=='logs':
            runner=(mount_path/'runner.py').read_text()
            nonce=re.search(r'print\("([a-f0-9]+)"',runner)[1]
            payload={'tests_run':1,'failures':0,'errors':0,'skipped':0,'expected_failures':0,
                     'unexpected_successes':0,'feedback':'MOCK unittest result'}
            return subprocess.CompletedProcess(cmd,0,stdout=(nonce+json.dumps(payload)+'\n').encode())
        return subprocess.CompletedProcess(cmd,0,stdout=b'')
    monkeypatch.setattr('policy_dissertation.pipelines.repository_data.sandbox.subprocess.run',simulated)
    runner=DockerSandbox(config)
    result=runner.run('python',SOLUTION,TESTS)
    assert result.status=='verified' and result.tests_run==1
    assert all(cmd[0]=='docker' for cmd in calls)
    assert calls[-1][1:3]==['rm','-f']


def test_missing_docker_never_executes_host_python(settings,monkeypatch):
    calls=[]
    def absent(cmd,**kwargs):
        calls.append(cmd)
        raise FileNotFoundError('docker missing')
    monkeypatch.setattr('policy_dissertation.pipelines.repository_data.sandbox.subprocess.run',absent)
    runner=DockerSandbox(settings.pipeline2.model_copy(update={'execution_mode':'auto'}))
    result=runner.run('python',SOLUTION,TESTS)
    assert result.status=='unverified'
    assert all(cmd[0]=='docker' for cmd in calls)


def test_real_container_when_available(settings):
    runner=DockerSandbox(settings.pipeline2.model_copy(update={'execution_mode':'auto'}))
    if not runner.available:
        pytest.skip('Real container unavailable: '+runner.reason)
    result=runner.run('python',SOLUTION,TESTS)
    assert result.status=='verified',result.model_dump()
    assert result.tests_run==1 and result.executor=='docker'


def test_interruption_preserves_validated_progress(repositories, tmp_path, settings):
    (repositories/'calculator/other.py').write_text('def decrement(value):\n    """Decrement a numeric value."""\n    return value - 1\n')
    class Interrupted(MockCodeModel):
        generated = 0
        def invoke(self, messages, config=None):
            data, _ = json.JSONDecoder().raw_decode(messages[1].content)
            if 'requested_task_type' in data:
                self.generated += 1
                if self.generated == 2:
                    raise KeyboardInterrupt()
            return super().invoke(messages, config)
    result = run(repositories, tmp_path, settings, model=Interrupted(), task_types=('explanation',))
    assert result['status'] == 'FAILED'
    assert result['accepted_examples'] == 1
    assert result['failed_units'] == 0
    assert result['pending_units'] == 1
    resumed = run(repositories, tmp_path, settings, task_types=('explanation',))
    assert resumed['status'] == 'COMPLETED_WITH_WARNINGS'
    assert resumed['resumed_generation_units'] == 1
    assert resumed['resumed_validation_units'] == 1
    assert resumed['pending_units'] == resumed['failed_units'] == 0
    records = [json.loads(line) for line in (tmp_path/'output/training.jsonl').read_text().splitlines()]
    assert len(records) == len({r['example_id'] for r in records}) == 2


def test_gateway_reports_truncation_without_response_leak(settings):
    from policy_dissertation.pipelines.repository_data.llm import CodeGateway
    from policy_dissertation.pipelines.policy_transformation.generation import ModelCallError
    class Truncated:
        def invoke(self, *args, **kwargs):
            return SimpleNamespace(content='test-p2-secret', response_metadata={'finish_reason': 'length'})
    gateway = CodeGateway(settings, Truncated())
    with pytest.raises(ModelCallError) as caught:
        gateway.generate_code({})
    assert caught.value.kind == 'response_truncated'
    assert caught.value.diagnostics == [{'attempt': 1, 'category': 'response_truncated', 'finish_reason': 'length'}]
    assert 'test-p2-secret' not in json.dumps(caught.value.diagnostics)


def test_gateway_reports_schema_categories_without_input(settings):
    from policy_dissertation.pipelines.repository_data.llm import CodeGateway
    from policy_dissertation.pipelines.policy_transformation.generation import ModelCallError
    class Invalid:
        def invoke(self, *args, **kwargs):
            return SimpleNamespace(content='{"extra": "private-input-value"}')
    with pytest.raises(ModelCallError) as caught:
        CodeGateway(settings, Invalid()).generate_code({})
    assert caught.value.kind == 'malformed_response'
    assert 'private-input-value' not in json.dumps(caught.value.diagnostics)
    assert 'missing' in caught.value.diagnostics[0]['error_types']
    assert 'test-p2-secret' not in json.dumps(caught.value.diagnostics)


def test_nested_candidate_schema_errors_retry_with_safe_feedback(settings):
    from policy_dissertation.pipelines.repository_data.llm import CodeGateway
    from policy_dissertation.pipelines.policy_transformation.generation import ModelCallError
    settings.pipeline2.max_attempts = 2
    class InvalidCandidate:
        messages = []
        def invoke(self, messages, config=None):
            self.messages.append(messages)
            return SimpleNamespace(content='{"examples":[{"instruction":"private-input-value"}]}')
    model = InvalidCandidate()
    with pytest.raises(ModelCallError) as caught:
        CodeGateway(settings, model).generate_code({})
    assert len(model.messages) == 2
    assert 'missing' in model.messages[1][1].content
    assert 'private-input-value' not in model.messages[1][1].content
    assert len(caught.value.diagnostics) == 2


def test_endpoint_circuit_breaker_counts_unattempted_work_as_pending(repositories, tmp_path, settings):
    (repositories/'calculator/other.py').write_text('def decrement(value):\n    """Decrement a numeric value."""\n    return value - 1\n')
    result = run(repositories, tmp_path, settings, model=MockCodeModel('endpoint_error'))
    assert result['status'] == 'FAILED'
    assert result['failed_units'] == 1
    assert result['pending_units'] == 1
    assert result['rejection_reasons'] == {'endpoint_error': 1, 'not_attempted_after_endpoint_failure': 1}
