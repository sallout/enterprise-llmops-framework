"""Synthetic evaluation workflow tests; no GPU inference or hosted generation."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.model_evaluation.schemas import Case, EvaluationConfig
from policy_dissertation.pipelines.model_evaluation.suites import prepare_suite, read_suite, adapt
from policy_dissertation.pipelines.model_evaluation.pipeline import run_evaluation, compression_models
from policy_dissertation.pipelines.model_evaluation.execution import extract_code, DockerExecutor
from policy_dissertation.pipelines.model_evaluation.scoring import score, lexical, summarize
from policy_dissertation.pipelines.model_evaluation.contamination import audit
from policy_dissertation.pipelines.policy_transformation.data_io import digest, dump_json


def provider(name, token=None):
    source={'dataset':name,'configuration':'test','split':'test','revision':'a'*40,'license':'fixture'}
    if name=='humaneval':
        return [{'task_id':str(i),'prompt':f'def answer_{i}():\n    """Return one."""\n',
            'canonical_solution':'    return 1\n','test':f'def check(candidate):\n    assert candidate() == 1',
            'entry_point':f'answer_{i}'} for i in range(5)],source
    return [{'task_id':str(i),'prompt':f'Return one from answer_{i}.','code':f'def answer_{i}(): return 1',
        'test_imports':[], 'test_list':[f'assert answer_{i}() == 1']} for i in range(5)],source


def frozen(tmp_path, limit=2):
    path=tmp_path/'suite.json'
    prepare_suite(path,max_tasks=limit,provider=provider)
    return path


class FakeBackend:
    def __init__(self):self.calls=[];self.loaded=[];self.released=0
    def preflight(self):return {'device':'mock','versions':{'fake':'1'}}
    def load(self,source,revision,config):self.loaded.append(source);return {'model_memory_bytes':100}
    def generate(self,messages,config):
        self.calls.append(messages)
        return {'status':'ok','text':'def answer_0(): return 1','output_tokens':10,'generation_seconds':1,
                'peak_allocated_bytes':100,'peak_reserved_bytes':200}
    def release(self):self.released+=1


class FakeExecutor:
    def __init__(self):self.calls=[]
    def preflight(self):return {'executor':'mock','image_id':'abc'}
    def run(self,code,tests):self.calls.append((code,tests));return {'status':'passed','reason':'mocked_tests'}


def identity(source,settings):return {'source':source,'revision':'a'*40,'identity':source}


def test_seeded_selection_is_stable_and_bounded(tmp_path):
    a=prepare_suite(tmp_path/'a.json',max_tasks=2,seed=7,provider=provider)
    b=prepare_suite(tmp_path/'b.json',max_tasks=2,seed=7,provider=provider)
    assert a==b and len(a['cases'])==4
    c=prepare_suite(tmp_path/'c.json',max_tasks=0,provider=provider)
    assert len(c['cases'])==10
    with pytest.raises(DissertationError):prepare_suite(tmp_path/'a.json',provider=provider)


def test_targets_and_hidden_tests_not_in_humaneval_prompt():
    rows,source=provider('humaneval');case=adapt('humaneval',rows[0],source)
    assert 'return 1' not in case.messages[0]['content']
    assert 'check(candidate)' not in case.messages[0]['content']
    assert 'check(answer_0)' in case.tests
    namespace={};exec(case.reference+'\n'+case.tests,namespace)  # authored fixture only


def test_mbpp_adapter_uses_its_actual_schema():
    rows,source=provider('mbpp');case=adapt('mbpp',rows[0],source)
    assert 'Required behaviour' in case.messages[0]['content']
    assert 'def answer_0' not in case.messages[0]['content']
    exec(case.reference+'\n'+case.tests,{})  # authored fixture only


def test_checksum_tamper_is_refused(tmp_path):
    path=frozen(tmp_path);p=json.loads(path.read_text());p['seed']=55;path.write_text(json.dumps(p))
    with pytest.raises(DissertationError,match='checksum'):read_suite(path)


def test_sequential_models_export_matched_results_and_resume_without_duplicates(tmp_path):
    path=frozen(tmp_path);backend=FakeBackend();executor=FakeExecutor();out=tmp_path/'out'
    kwargs=dict(backend=backend,executor=executor,identity_provider=identity,progress=lambda _:None)
    report=run_evaluation(path,out,{'original':'first','final':'second'},Settings(),EvaluationConfig(),**kwargs)
    assert report['status']=='COMPLETED_WITH_WARNINGS'  # no contamination input
    assert report['evaluation_complete'] and report['evaluated_tasks_per_model']==4
    assert backend.loaded==['first','second']
    assert report['comparisons']['final']['humaneval']['delta_percentage_points']==0
    assert len((out/'results.jsonl').read_text().splitlines())==8
    calls=len(backend.calls)
    resumed=run_evaluation(path,out,{'original':'first','final':'second'},Settings(),EvaluationConfig(),resume=True,**kwargs)
    assert len(backend.calls)==calls+2  # warmup per model only
    assert len((out/'results.jsonl').read_text().splitlines())==8
    assert resumed['models']==report['models']


def test_resume_rejects_changed_configuration(tmp_path):
    path=frozen(tmp_path);out=tmp_path/'out'
    kwargs=dict(backend=FakeBackend(),executor=FakeExecutor(),identity_provider=identity,progress=lambda _:None)
    run_evaluation(path,out,{'final':'one'},Settings(),EvaluationConfig(),**kwargs)
    with pytest.raises(DissertationError):run_evaluation(path,out,{'final':'one'},Settings(),EvaluationConfig(max_new_tokens=100),resume=True,**kwargs)
    assert json.loads((out/'report.json').read_text())['status']=='FAILED'


def test_reference_failure_stops_before_model_load(tmp_path):
    executor=FakeExecutor();executor.run=lambda *_:{'status':'failed','reason':'bad_fixture'}
    backend=FakeBackend();path=frozen(tmp_path)
    with pytest.raises(DissertationError):
        run_evaluation(path,tmp_path/'out',{'final':'one'},Settings(),EvaluationConfig(),
            backend=backend,executor=executor,identity_provider=identity,progress=lambda _:None)
    assert not backend.loaded


def test_interruption_keeps_completed_task_checkpoints(tmp_path):
    path=frozen(tmp_path);backend=FakeBackend();original=backend.generate
    def interrupted(messages,config):
        if len(backend.calls)==2:raise KeyboardInterrupt()
        return original(messages,config)
    backend.generate=interrupted
    with pytest.raises(DissertationError):
        run_evaluation(path,tmp_path/'out',{'final':'one'},Settings(),EvaluationConfig(),
            backend=backend,executor=FakeExecutor(),identity_provider=identity,progress=lambda _:None)
    assert json.loads((tmp_path/'out/report.json').read_text())['status']=='INTERRUPTED'
    assert len(list((tmp_path/'out/_checkpoints').rglob('*.json')))==1


def test_generation_only_never_reports_pass_at_one(tmp_path):
    path=frozen(tmp_path)
    r=run_evaluation(path,tmp_path/'out',{'final':'one'},Settings(),EvaluationConfig(executor='disabled'),
        backend=FakeBackend(),identity_provider=identity,progress=lambda _:None)
    assert not r['evaluation_complete']
    assert r['models']['final']['metrics']['humaneval']['pass_at_1'] is None


@pytest.mark.parametrize('status',['output_limit','input_too_long','context_limit'])
def test_token_limits_are_failed_without_truncation(status):
    case=Case(id='x',suite='mbpp',messages=[],reference='x',tests='assert 1',source={})
    assert score(case,{'status':status,'text':'code'},FakeExecutor())=={'status':'failed','reason':status}


def test_lexical_score_is_not_semantic_certification():
    r=lexical('Approval is required','Approval is required')
    assert r['token_f1']==1 and r['semantic_correctness']=='NOT_ASSESSED' and r['human_review_required']
    assert lexical('approval is not required','approval is required')['token_f1']>.8  # dangerous semantic mismatch remains explicit


@pytest.mark.parametrize('text,expected',[('```python\ndef f(): pass\n```','def f(): pass\n'),
    ('def f(): pass','def f(): pass\n'),('```python\nx\n```\n```python\ny\n```',None),('```python\nx',None)])
def test_code_extraction(text,expected):assert extract_code(text)==expected


def test_known_training_overlap_is_detected(tmp_path):
    reference=' '.join('word'+str(i) for i in range(30))
    case=Case(id='x',suite='mbpp',messages=[{'role':'user','content':'short'}],reference=reference,source={})
    path=tmp_path/'tokens.jsonl';path.write_text(json.dumps({'input_ids':[1], 'record_id':'train-x'})+'\n')
    r=audit([case],path,SimpleNamespace(decode=lambda *_args,**_kwargs:reference),lambda _:None)
    assert r['training_rows_checked']==1 and 'x' in r['excluded']


def test_cli_prepare_only_and_help(tmp_path,monkeypatch):
    import policy_dissertation.pipelines.model_evaluation.suites as module
    monkeypatch.setattr(module,'official_rows',provider)
    runner=CliRunner()
    assert runner.invoke(app,['evaluate_model','--help']).exit_code==0
    r=runner.invoke(app,['evaluate_model','--output',str(tmp_path/'out'),'--max-tasks','1','--prepare-only'])
    assert r.exit_code==0,r.output
    assert 'SUITE_PREPARED' in r.output
    assert len(read_suite(tmp_path/'out/suite.json')[1])==2


def test_dashboard_forwards_evaluation_limits(tmp_path,monkeypatch):
    from test_interactive import Scripted
    monkeypatch.chdir(tmp_path)
    ui=Scripted(choices=['prepare','both'], answers=[str(tmp_path/'out'),'','3','','42'],confirmations=[True])
    ui.evaluation()
    assert ui.commands[0][0]=='evaluate_model'
    assert ui.commands[0][ui.commands[0].index('--max-tasks')+1]=='3'
    assert '--prepare-only' in ui.commands[0]


def test_compression_paths_preserve_existing_weights(tmp_path):
    report={'final_model_ready':True,'status':'COMPLETED_WITH_WARNINGS','fingerprint':'a'*64}
    dump_json(tmp_path/'report.json',report)
    result=compression_models(tmp_path,True)
    assert list(result)==['original_bf16','pruned_bf16','recovered_bf16','final_nf4']
    assert result['final_nf4']==str(tmp_path/'model_4bit')


def test_docker_unavailable_fails_without_host_execution(monkeypatch):
    import policy_dissertation.pipelines.model_evaluation.execution as module
    monkeypatch.setattr(module.subprocess,'run',lambda *_args,**_kwargs:(_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(DissertationError,match='Docker'):DockerExecutor(EvaluationConfig()).preflight()


def test_enterprise_selects_only_hash_verified_test_and_keeps_source(tmp_path):
    from test_embedded_preparation import record
    root=tmp_path/'enterprise';root.mkdir()
    hashes={}
    for i,split in enumerate(('train','validation','test')):
        row=record(i);row['metadata']['training_split']=split
        path=root/(split+'.jsonl');path.write_text(json.dumps(row)+'\n')
        from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
        hashes[path.name]=file_hash(path)
    dump_json(root/'manifest.json',{'training_ready':True,'output_hashes':hashes})
    prepared=prepare_suite(tmp_path/'enterprise_suite.json',benchmarks=[],enterprise=root,enterprise_limit=0)
    assert len(prepared['cases'])==1
    assert prepared['cases'][0]['source']['record_id']==row['id']
    assert prepared['cases'][0]['messages'][-1]['role']=='user'
    assert prepared['cases'][0]['reference']==row['messages'][-1]['content']
    (root/'test.jsonl').write_text('{}\n')
    with pytest.raises(DissertationError,match='checksum'):
        prepare_suite(tmp_path/'changed.json',benchmarks=[],enterprise=root)


def test_enterprise_never_uses_training_union(tmp_path):
    root=tmp_path/'enterprise';root.mkdir();(root/'training.jsonl').write_text('{}\n')
    with pytest.raises(DissertationError,match='test.jsonl'):
        prepare_suite(tmp_path/'suite.json',benchmarks=[],enterprise=root)


def test_docker_command_has_isolation_and_timeout_cleanup(tmp_path,monkeypatch):
    import policy_dissertation.pipelines.model_evaluation.execution as module
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if args[1]=='create':return SimpleNamespace(returncode=0)
        if args[1]=='start':return SimpleNamespace(returncode=0)
        if args[1]=='wait':raise module.subprocess.TimeoutExpired(args,1)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',run)
    e=DockerExecutor(EvaluationConfig());e.image_id='sha256:'+'1'*64
    result=e.run('def f(): return 1','assert f()==1')
    assert result['reason']=='timeout'
    create=calls[0]
    assert create[create.index('--network')+1]=='none'
    assert '--read-only' in create and '--cap-drop' in create and '--gpus' not in create
    assert calls[-1][1:3]==['rm','-f']


def test_saved_answers_can_be_scored_without_gpu_or_regeneration(tmp_path):
    from policy_dissertation.pipelines.model_evaluation.pipeline import score_saved
    path=frozen(tmp_path);generated=tmp_path/'generated';backend=FakeBackend()
    run_evaluation(path,generated,{'original':'one','final':'two'},Settings(),EvaluationConfig(executor='disabled'),
        backend=backend,identity_provider=identity,progress=lambda _:None)
    calls=len(backend.calls)
    report=score_saved(generated,tmp_path/'scored',EvaluationConfig(),executor=FakeExecutor(),progress=lambda _:None)
    assert report['evaluation_complete'] and len(backend.calls)==calls
    assert report['models']['final']['metrics']['humaneval']['pass_at_1']==1
    assert 'code_execution_disabled_no_pass_at_1' not in report['warnings']
    with (generated/'results.jsonl').open('a') as stream:stream.write('{}\n')
    with pytest.raises(DissertationError,match='checksum'):
        score_saved(generated,tmp_path/'tampered',EvaluationConfig(),executor=FakeExecutor())


def test_ui_exposes_score_only_without_requiring_model(tmp_path,monkeypatch):
    from test_interactive import Scripted
    monkeypatch.chdir(tmp_path);source=tmp_path/'generated';source.mkdir()
    ui=Scripted(choices=['score'],answers=[str(source),str(tmp_path/'scored'),'python:3.12-slim'],confirmations=[True])
    ui.evaluation()
    assert ui.commands[0][0]=='score_evaluation'


def test_cli_refuses_output_inside_compression_before_writing_suite(tmp_path):
    source=tmp_path/'compression';source.mkdir();(source/'report.json').write_text('preserve')
    r=CliRunner().invoke(app,['evaluate_model','--compression-run',str(source),'--output',str(source)])
    assert r.exit_code==1 and 'separate' in r.output
    assert not (source/'suite.json').exists() and (source/'report.json').read_text()=='preserve'


def test_different_model_templates_fail_matched_comparison(tmp_path):
    path=frozen(tmp_path);backend=FakeBackend()
    backend.load=lambda source,*_args: {'chat_template_hash':source}
    with pytest.raises(DissertationError):
        run_evaluation(path,tmp_path/'out',{'original':'template-a','final':'template-b'},Settings(),EvaluationConfig(),
            backend=backend,executor=FakeExecutor(),identity_provider=identity,progress=lambda _:None)
    r=json.loads((tmp_path/'out/report.json').read_text())
    assert 'templates differ' in r['errors'][0] and r['status']=='FAILED'
