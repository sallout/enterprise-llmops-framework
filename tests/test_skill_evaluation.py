"""Skill profiling/suite selection uses source fixtures, never model outcomes."""
import json
from pathlib import Path
import pytest
from test_native_datasets import enterprise,settings,run,SimulatedTokenizer
from test_pipeline3 import FakeProvider
from test_model_evaluation import provider as benchmarks,FakeExecutor
from policy_dissertation.skill_evaluation import skills,language,function_hashes,OverlapScreen,select_cases,prepare_skill_evaluation
from policy_dissertation.pipelines.model_evaluation.suites import adapt,read_suite


def test_labels_are_multilabel_and_do_not_claim_semantic_ground_truth():
    assert {'sequences','numeric_math','recursion_dp'}<=set(skills('Recursively sum integers in a nested list.'))
    assert language('Write it','```python\ndef f(x): return x\n```')==('python','code_fence')
    assert language('No language','Hello world')[0]=='unknown'
    assert skills('Explain the business outcome.')==['unclassified']


def test_code_identity_ignores_docstrings_and_entry_name_but_not_behavior():
    a='def first(x):\n    """text"""\n    return [i+1 for i in x if i>0]'
    b='def other(x):\n    return [i+1 for i in x if i>0]'
    assert function_hashes(a)==function_hashes(b) and function_hashes(a)
    assert function_hashes(a)!=function_hashes(b.replace('i+1','i-1'))


def test_overlap_detects_training_reference_without_exposing_it_in_question():
    rows,source=benchmarks('humaneval');case=adapt('humaneval',rows[0],source)
    long=' '.join('term'+str(i) for i in range(35))
    case.source['task_prompt']=long
    screen=OverlapScreen([case]);screen.check('training-id',long,'different')
    assert case.id in screen.matches


class Controls(FakeExecutor):
    def run(self,code,tests):
        return {'status':'failed' if 'return None' in code else 'passed'}


def test_frozen_suites_disjoint_exclude_old_and_compatible(tmp_path,enterprise,settings):
    result=run(tmp_path,enterprise,settings,selected=('opencode',))
    assert result['data_prepared']
    out=tmp_path/'analysis'
    r=prepare_skill_evaluation(tmp_path/'native',enterprise,out,settings,sample_size=2,max_tasks=10,
        provider=FakeProvider(),benchmark_provider=benchmarks,executor=Controls(),progress=lambda _:None)
    assert r['status']=='COMPLETED_WITH_WARNINGS',r
    assert r['profile']['sample_size']==2 and r['native_accepted_rows_checked']==result['accepted_examples']
    _,dev=read_suite(out/'development_suite.json');_,final=read_suite(out/'final_suite.json')
    assert not {c.id for c in dev}&{c.id for c in final}
    assert len(dev)+len(final)==10
    assert all(c.source['local_reference_test_status']=='passed' for c in dev+final)
    for c in dev+final:
        assert c.reference not in c.messages[0]['content']
    second=tmp_path/'analysis2'
    r2=prepare_skill_evaluation(tmp_path/'native',enterprise,second,settings,sample_size=2,max_tasks=10,
        excluded_suites=[out/'development_suite.json'],provider=FakeProvider(),benchmark_provider=benchmarks,executor=Controls(),progress=lambda _:None)
    # Only unseen IDs can survive, even when the requested task count cannot be reached.
    if r2['status']!='FAILED':
        _,d2=read_suite(second/'development_suite.json');_,f2=read_suite(second/'final_suite.json')
        assert not {c.id for c in dev}&{c.id for c in d2+f2}


def test_corrupt_selection_fails_without_approved_suites(tmp_path,enterprise,settings):
    run(tmp_path,enterprise,settings,selected=('opencode',))
    path=tmp_path/'native/source_plan.json';plan=json.loads(path.read_text());plan['decision_database_hash']='bad';path.write_text(json.dumps(plan))
    out=tmp_path/'analysis'
    r=prepare_skill_evaluation(tmp_path/'native',enterprise,out,settings,provider=FakeProvider(),benchmark_provider=benchmarks,executor=Controls())
    assert r['status']=='FAILED' and not (out/'final_suite.json').exists()


def test_cli_exposes_limits():
    from typer.testing import CliRunner
    from policy_dissertation.cli import app
    result=CliRunner().invoke(app,['prepare_skill_evaluation','--help'])
    assert result.exit_code==0 and '--sample-size' in result.output and '--exclude-suite' in result.output


def test_menu_forwards_skill_limits_without_training(tmp_path,monkeypatch):
    from test_interactive import Scripted
    monkeypatch.chdir(tmp_path)
    native=tmp_path/'native';native.mkdir();ent=tmp_path/'enterprise';ent.mkdir()
    ui=Scripted(choices=['skills'],answers=[str(native),str(ent),str(tmp_path/'out'),'100','40','20',''],confirmations=[True])
    ui.evaluation()
    assert ui.commands[0][0]=='prepare_skill_evaluation'
    assert ui.commands[0][ui.commands[0].index('--sample-size')+1]=='100'
    assert '--compression-run' not in ui.commands[0]


def test_duplicate_reference_tasks_cannot_cross_evaluation_splits():
    from policy_dissertation.skill_evaluation import verify_distinct_references
    from policy_dissertation.errors import DissertationError
    rows,source=benchmarks('humaneval')
    first=adapt('humaneval',rows[0],source)
    second=first.model_copy(deep=True);second.id='different-id'
    with pytest.raises(DissertationError,match='Duplicate'):
        verify_distinct_references([first,second])


def test_explicit_final_count_leaves_disjoint_development(tmp_path,enterprise,settings):
    run(tmp_path,enterprise,settings,selected=('opencode',))
    out=tmp_path/'explicit_final'
    r=prepare_skill_evaluation(tmp_path/'native',enterprise,out,settings,sample_size=2,max_tasks=10,
        final_tasks=5,provider=FakeProvider(),benchmark_provider=benchmarks,executor=Controls(),progress=lambda _:None)
    assert r['status']=='COMPLETED_WITH_WARNINGS'
    assert r['final_tasks']==5 and r['development_tasks']==5


def test_final_count_cannot_consume_all_tasks(tmp_path,enterprise,settings):
    from policy_dissertation.errors import DissertationError
    with pytest.raises(DissertationError,match='smaller'):
        prepare_skill_evaluation(tmp_path/'native',enterprise,tmp_path/'bad',settings,max_tasks=10,final_tasks=10)
