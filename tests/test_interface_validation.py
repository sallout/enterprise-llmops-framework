"""Read-only native-validation action behind the interactive workspace."""
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.interface.validation import validate_native_plan
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.native_datasets.pipeline import run_native_datasets
from policy_dissertation.pipelines.enterprise.pipeline import run_enterprise
from policy_dissertation.pipelines.enterprise.preparation import prepare_enterprise
from test_enterprise import dataset
from test_native_datasets import SimulatedTokenizer
from test_pipeline3 import FakeProvider


@pytest.fixture
def artifacts(tmp_path):
    settings=Settings(_env_file=None,pipeline3={'shuffle_buffer':1})
    old=tmp_path/'old';dataset(old,count=8)
    accepted=tmp_path/'accepted'
    run_enterprise(accepted,settings,code_dataset=old,research_report=tmp_path/'research.md')
    ready=tmp_path/'ready'
    prepare_enterprise(accepted,ready,settings,tokenizer=SimulatedTokenizer())
    plan=tmp_path/'plan'
    r=run_native_datasets(ready,plan,settings,max_samples=4,provider=FakeProvider(),tokenizer=SimulatedTokenizer())
    assert r['data_prepared']
    return settings,ready,plan


def test_full_validation_does_not_modify_prepared_plan_or_enterprise(tmp_path,artifacts):
    settings,ready,plan=artifacts
    source_files=[p for folder in (ready,plan) for p in folder.rglob('*') if p.is_file()]
    before={p:p.read_bytes() for p in source_files}
    r=validate_native_plan(plan,ready,tmp_path/'audit',settings,provider=FakeProvider(),tokenizer=SimulatedTokenizer())
    assert r['status']=='COMPLETED' and r['full_plan_validated'] and r['records_checked']==4
    assert r['llm_calls']==0 and r['external_tests_locally_executed']==0 and not r['training_started']
    assert all(p.read_bytes()==b for p,b in before.items())
    assert not list((tmp_path/'audit').rglob('*.jsonl'))


def test_partial_validation_cannot_claim_full_success(tmp_path,artifacts):
    settings,ready,plan=artifacts
    r=validate_native_plan(plan,ready,tmp_path/'audit',settings,limit_records=1,provider=FakeProvider(),tokenizer=SimulatedTokenizer())
    assert r['status']=='COMPLETED_WITH_WARNINGS' and r['records_checked']==1
    assert not r['full_plan_validated'] and r['scope']=='sample only'


def test_corrupted_plan_reports_failure_not_green_readiness(tmp_path,artifacts):
    settings,ready,plan=artifacts
    source=json.loads((plan/'source_plan.json').read_text())
    with (plan/source['decision_database']).open('ab') as f:f.write(b'corrupted')
    r=validate_native_plan(plan,ready,tmp_path/'audit',settings,provider=FakeProvider(),tokenizer=SimulatedTokenizer())
    assert r['status']=='FAILED' and not r['full_plan_validated']


def test_validation_preserves_failed_source_and_rejects_overlap(tmp_path,artifacts):
    settings,ready,plan=artifacts
    with pytest.raises(DissertationError):validate_native_plan(plan,ready,plan,settings)
    with pytest.raises(DissertationError):validate_native_plan(plan,ready,ready,settings)
    p=json.loads((plan/'source_plan.json').read_text());p['data_prepared']=False
    (plan/'source_plan.json').write_text(json.dumps(p))
    r=validate_native_plan(plan,ready,tmp_path/'audit',settings,tokenizer=SimulatedTokenizer())
    assert r['status']=='FAILED' and not r['full_plan_validated']


def test_cli_maps_validation_failure_and_allows_explicit_reaudit(tmp_path,artifacts,monkeypatch):
    import policy_dissertation.interface.validation as module
    settings,ready,plan=artifacts
    monkeypatch.setattr(module,'load_tokenizer',lambda *a:SimulatedTokenizer())
    original=module.iter_prepared_native
    monkeypatch.setattr(module,'iter_prepared_native',lambda *a,**k:original(*a,**(k|{'provider':FakeProvider()})))
    args=['nvidia_validate','--plan',str(plan),'--enterprise',str(ready),'--output',str(tmp_path/'audit')]
    r=CliRunner().invoke(app,args)
    assert r.exit_code==0 and 'full approved selection validated: True' in r.output
    assert CliRunner().invoke(app,args).exit_code==1
    assert CliRunner().invoke(app,args+['--overwrite']).exit_code==0
