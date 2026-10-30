"""Enterprise composition tests: actual branch graphs, synthetic sources, mocked LLMs."""
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.data.adapters import adapt
from policy_dissertation.data.schemas import UnifiedRecord
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.enterprise.pipeline import run_enterprise
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
from policy_dissertation.pipelines.dataset_preparation.pipeline import run_pipeline4
from test_pipeline1 import MockModel
from test_pipeline2 import MockCodeModel, MockSandbox
from test_pipeline4 import policy, repository, FakeTokenizer


@pytest.fixture
def settings():
    return Settings(_env_file=None,model={'base_url':'https://example.invalid/v1','api_key':'test-secret','model_name':'mock'},
                    pipeline1={'max_attempts':1},pipeline2={'max_attempts':1,'execution_mode':'disabled'})


@pytest.fixture
def documents(tmp_path):
    p=tmp_path/'documents';p.mkdir()
    (p/'access.md').write_text('# Access Policy\n\n## 1. Controls\n\nEmployees MUST obtain approval before transferring restricted records unless Legal grants an exception.\n')
    return p


@pytest.fixture
def code(tmp_path):
    p=tmp_path/'repos'/'calculator';p.mkdir(parents=True)
    (p/'pyproject.toml').write_text('[project]\nname="calculator"\n')
    (p/'calc.py').write_text('def increment(value):\n    """Increment a numeric value."""\n    return value + 1\n')
    return p.parent


def dataset(path,origin='pipeline2',count=3,failed=False):
    path.mkdir(parents=True)
    factory=repository if origin=='pipeline2' else policy
    records=[adapt(factory(i),origin,{}) for i in range(count)]
    (path/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    report={'status':'FAILED' if failed else 'COMPLETED','errors':['mandatory_unit_processing_incomplete'] if failed else [],
            'accepted_examples':count,'output_hashes':{'training.jsonl':file_hash(path/'training.jsonl')}}
    (path/'report.json').write_text(json.dumps(report))
    return records


def run(tmp_path,settings,**kwargs):
    return run_enterprise(tmp_path/'enterprise',settings,research_report=tmp_path/'research.md',**kwargs)


def rows(tmp_path):
    return [json.loads(l) for l in (tmp_path/'enterprise/training.jsonl').read_text().splitlines()]


def test_documents_only_real_parsing_graph(documents,tmp_path,settings):
    model=MockModel();original=(documents/'access.md').read_bytes()
    result=run(tmp_path,settings,documents=documents,model=model,limit_documents=1)
    assert result['status']=='COMPLETED' and result['accepted_examples']==1
    assert set(result['branches'])=={'documents'}
    r=rows(tmp_path)[0];UnifiedRecord.model_validate(r)
    assert r['domain']=='corporate_compliance'
    assert 'unless Legal grants an exception' in r['messages'][-1]['content']
    assert (documents/'access.md').read_bytes()==original
    assert not (tmp_path/'enterprise/_branches/code').exists()


def test_code_only_real_static_graph_with_mock_generation(code,tmp_path,settings):
    model=MockCodeModel()
    result=run(tmp_path,settings,code=code,model=model,sandbox=MockSandbox(),limit_units=1,task_types=('explanation',))
    assert result['status']=='COMPLETED_WITH_WARNINGS' and result['accepted_examples']==1
    assert set(result['branches'])=={'code'}
    r=rows(tmp_path)[0]
    assert r['domain']=='coding' and r['verification']['status']=='not_applicable'
    assert not (tmp_path/'enterprise/_branches/documents').exists()


def test_both_generated_branches_and_resume(documents,code,tmp_path,settings):
    # Separate endpoint clients are injected by patching branch runners below; no network is used.
    import policy_dissertation.pipelines.enterprise.pipeline as module
    from unittest.mock import patch
    original_policy,original_code=module.run_pipeline,module.run_pipeline2
    def policy_runner(*a,**k):k['model']=MockModel();return original_policy(*a,**k)
    def code_runner(*a,**k):k['model']=MockCodeModel();return original_code(*a,**k)
    with patch.object(module,'run_pipeline',policy_runner),patch.object(module,'run_pipeline2',code_runner):
        r=run(tmp_path,settings,documents=documents,code=code,sandbox=MockSandbox(),limit_units=1,task_types=('explanation',))
        assert r['accepted_examples']==2 and not r['errors']
        before=(tmp_path/'enterprise/training.jsonl').read_bytes()
        again=run(tmp_path,settings,documents=documents,code=code,sandbox=MockSandbox(),limit_units=1,task_types=('explanation',),overwrite=True)
    assert again['accepted_examples']==2
    assert (tmp_path/'enterprise/training.jsonl').read_bytes()==before
    assert {r['metadata']['pipeline'] for r in rows(tmp_path)}=={'pipeline1','pipeline2'}


def test_documents_plus_existing_code_never_calls_code_generation(documents,tmp_path,settings,monkeypatch):
    import policy_dissertation.pipelines.enterprise.pipeline as module
    old=tmp_path/'old-code';dataset(old)
    before={p.name:p.read_bytes() for p in old.iterdir()}
    def forbidden(*a,**k):raise AssertionError('Code generation must not be called for imported data')
    monkeypatch.setattr(module,'run_pipeline2',forbidden)
    r=run(tmp_path,settings,documents=documents,code_dataset=old,model=MockModel())
    assert r['accepted_examples']==4 and not r['errors']
    assert all((old/n).read_bytes()==b for n,b in before.items())
    assert not (tmp_path/'enterprise/_branches/code').exists()


def test_existing_datasets_only_and_duplicates(tmp_path,settings):
    old=tmp_path/'old';dataset(old)
    p=old/'training.jsonl';p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
    r=json.loads((old/'report.json').read_text());r['accepted_examples']=4;r['output_hashes']['training.jsonl']=file_hash(p)
    (old/'report.json').write_text(json.dumps(r))
    result=run(tmp_path,settings,code_dataset=old)
    assert result['accepted_examples']==3 and result['duplicates_removed']==1
    row=next(r for r in rows(tmp_path) if r['metadata'].get('enterprise_duplicate_references'))
    assert len(row['metadata']['enterprise_duplicate_references'])==1
    assert result['branches']['code']['mode']=='import'


@pytest.mark.parametrize('allow',[False,True])
def test_incomplete_import_requires_explicit_flag_and_retains_status(tmp_path,settings,allow):
    old=tmp_path/'old';dataset(old,failed=True)
    original=(old/'report.json').read_bytes()
    r=run(tmp_path,settings,code_dataset=old,allow_incomplete_import=allow)
    assert r['source_run_complete'] is False
    assert r['status']==('COMPLETED_WITH_WARNINGS' if allow else 'FAILED')
    assert r['accepted_examples']==(3 if allow else 0)
    if allow:assert all(x['metadata']['enterprise_import']['producer_complete'] is False for x in rows(tmp_path))
    assert (old/'report.json').read_bytes()==original


@pytest.mark.parametrize('problem',['checksum','count','wrong_origin','invalid_record'])
def test_bad_import_does_not_get_accepted_silently(tmp_path,settings,problem):
    old=tmp_path/'old';dataset(old,origin='pipeline1' if problem=='wrong_origin' else 'pipeline2')
    report=json.loads((old/'report.json').read_text())
    if problem=='checksum':report['output_hashes']['training.jsonl']='a'*64
    if problem=='count':report['accepted_examples']=20
    if problem=='invalid_record':
        p=old/'training.jsonl';data=p.read_text().splitlines();r=json.loads(data[0]);r['id']='b'*64;data[0]=json.dumps(r)
        p.write_text('\n'.join(data)+'\n');report['output_hashes']['training.jsonl']=file_hash(p)
    (old/'report.json').write_text(json.dumps(report))
    result=run(tmp_path,settings,code_dataset=old)
    assert result['status']=='FAILED' and result['errors']


def test_failure_of_requested_branch_not_hidden_by_successful_import(documents,tmp_path,settings):
    old=tmp_path/'old';dataset(old)
    r=run(tmp_path,settings,documents=documents,code_dataset=old,model=MockModel('endpoint_error'))
    assert r['status']=='FAILED' and r['accepted_examples']==3
    assert 'documents:mandatory_branch_failed_or_incomplete' in r['errors']


def test_source_overlap_and_legacy_dataset_output_protected(tmp_path,settings,code):
    old=tmp_path/'old';dataset(old)
    original=(old/'training.jsonl').read_bytes()
    with pytest.raises(DissertationError):run_enterprise(old,settings,code=code,overwrite=True)
    with pytest.raises(DissertationError):run_enterprise(code,settings,code=code)
    assert (old/'training.jsonl').read_bytes()==original


def test_no_inputs_and_conflicting_inputs_rejected(tmp_path,settings,documents):
    with pytest.raises(DissertationError):run(tmp_path,settings)
    with pytest.raises(DissertationError):run(tmp_path,settings,documents=documents,documents_dataset=documents)
    r=CliRunner().invoke(app,['enterprise_data'])
    assert r.exit_code==1 and 'Provide --documents' in r.output


def test_cli_import_and_overwrite_protection(tmp_path,monkeypatch):
    old=tmp_path/'old';dataset(old)
    monkeypatch.chdir(tmp_path)
    command=['enterprise_data','--skip-preparation','--code-dataset',str(old),'--output',str(tmp_path/'enterprise')]
    r=CliRunner().invoke(app,command)
    assert r.exit_code==0 and 'Accepted enterprise examples: 3' in r.output
    assert CliRunner().invoke(app,command).exit_code==1
    assert CliRunner().invoke(app,command+['--overwrite']).exit_code==0


def test_enterprise_export_prepares_with_existing_pipeline4(tmp_path,settings):
    old=tmp_path/'old';dataset(old,count=8)
    run(tmp_path,settings,code_dataset=old)
    from policy_dissertation.data.sources import Source
    settings.pipeline4.mix_ratios={'enterprise':1}
    result=run_pipeline4(tmp_path/'unused',tmp_path/'prepared',settings,
        sources=[Source(name='enterprise',type='local',path=str(tmp_path/'enterprise'),weight=1)],
        tokenizer=FakeTokenizer(),research_report=tmp_path/'research.md')
    assert not result['errors'] and result['accepted_examples']==8
    assert result['splitting']['component_leakage']==0


def test_publishing_incomplete_enterprise_subset_requires_explicit_opt_in(tmp_path,settings):
    from policy_dissertation.data.hub import publish
    from test_data_portability import MockHub
    from pydantic import SecretStr
    old=tmp_path/'old';dataset(old,failed=True)
    run(tmp_path,settings,code_dataset=old,allow_incomplete_import=True)
    settings.hf_token=SecretStr('synthetic-token')
    hub=MockHub()
    with pytest.raises(DissertationError):publish(tmp_path/'enterprise','owner/enterprise',settings,api=hub)
    assert not hub.uploads
    r=publish(tmp_path/'enterprise','owner/enterprise',settings,api=hub,allow_incomplete_run=True)
    assert r['accepted_subset_only'] is True


def test_invalid_code_branch_does_not_stop_documents(documents,tmp_path,settings):
    old=tmp_path/'old';dataset(old)
    (old/'training.jsonl').write_text('corrupted')
    r=run(tmp_path,settings,documents=documents,code_dataset=old,model=MockModel())
    assert r['status']=='FAILED' and r['accepted_examples']==1
    assert set(r['branches'])=={'documents'}
    assert any('checksum mismatch' in e for e in r['errors'])


def test_incomplete_import_disclosure_survives_pipeline4_loading(tmp_path,settings):
    from policy_dissertation.data.sources import Source
    old=tmp_path/'old';dataset(old,count=8,failed=True)
    run(tmp_path,settings,code_dataset=old,allow_incomplete_import=True)
    settings.pipeline4.mix_ratios={'enterprise':1}
    result=run_pipeline4(tmp_path/'unused',tmp_path/'prepared',settings,
        sources=[Source(name='enterprise',type='local',path=str(tmp_path/'enterprise'),weight=1)],
        tokenizer=FakeTokenizer(),research_report=tmp_path/'research.md')
    assert 'enterprise:accepted_subset_from_incomplete_producer' in result['warnings']
    assert result['inputs'][0]['producer_run']['complete'] is False
