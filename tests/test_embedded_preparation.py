"""Embedded finalization/direct native loading tests; fake tokenizers never certify real training."""
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.data.adapters import adapt, InvalidExample
from policy_dissertation.data.training_inputs import TrainingInputGate, iter_nvidia
from policy_dissertation.pipelines.enterprise.pipeline import run_enterprise
from policy_dissertation.pipelines.enterprise.preparation import prepare_enterprise
from policy_dissertation.pipelines.dataset_preparation.quality import validate_tokens
from policy_dissertation.errors import DissertationError
from test_enterprise import dataset
from test_pipeline4 import FakeTokenizer, external
from test_pipeline3 import FakeProvider, chat


def accepted(tmp_path,n=12):
    old=tmp_path/'original';dataset(old,count=n)
    root=tmp_path/'enterprise'
    run_enterprise(root,Settings(_env_file=None),code_dataset=old,research_report=tmp_path/'research.md')
    return root


def test_existing_enterprise_prepares_without_regeneration_and_preserves_source(tmp_path):
    source=accepted(tmp_path)
    before={p.name:p.read_bytes() for p in source.glob('*.json*')}
    r=prepare_enterprise(source,tmp_path/'ready',Settings(_env_file=None),tokenizer=FakeTokenizer())
    assert not r['errors'] and sum(r['splits'].values())==12
    assert r['training_ready'] is False  # Fake tokenizer cannot certify actual Qwen compatibility.
    assert r['preparation']['splitting']['component_leakage']==0
    assert all((source/n).read_bytes()==b for n,b in before.items())
    ids=set()
    for split in ('train','validation','test'):
        for line in (tmp_path/f'ready/{split}.jsonl').read_text().splitlines():
            row=json.loads(line)
            assert row['metadata']['training_split']==split and row['id'] not in ids
            ids.add(row['id'])
    assert len(ids)==12


def test_one_group_fails_closed_without_discarding_original_accepted(tmp_path):
    source=accepted(tmp_path,1)
    original=(source/'training.jsonl').read_bytes()
    r=prepare_enterprise(source,tmp_path/'ready',Settings(_env_file=None),tokenizer=FakeTokenizer())
    assert r['status']=='FAILED' and not r['training_ready']
    assert (source/'training.jsonl').read_bytes()==original


def test_default_enterprise_cli_embeds_preparation(tmp_path,monkeypatch):
    import policy_dissertation.pipelines.dataset_preparation.pipeline as module
    monkeypatch.setattr(module,'load_tokenizer',lambda *a:FakeTokenizer())
    old=tmp_path/'old';dataset(old,count=9)
    monkeypatch.chdir(tmp_path)
    r=CliRunner().invoke(app,['enterprise_data','--code-dataset',str(old),'--output',str(tmp_path/'enterprise')])
    assert r.exit_code==0,r.output
    assert (tmp_path/'enterprise/train.jsonl').exists()
    report=json.loads((tmp_path/'enterprise/report.json').read_text())
    assert 'preparation' in report and sum(report['splits'].values())==9


def test_enterprise_prepare_cli_preserves_old_output(tmp_path,monkeypatch):
    import policy_dissertation.pipelines.dataset_preparation.pipeline as module
    monkeypatch.setattr(module,'load_tokenizer',lambda *a:FakeTokenizer())
    source=accepted(tmp_path)
    before=(source/'training.jsonl').read_bytes()
    monkeypatch.chdir(tmp_path)
    r=CliRunner().invoke(app,['enterprise_prepare','--input',str(source),'--output',str(tmp_path/'ready')])
    assert r.exit_code==0,r.output
    assert (source/'training.jsonl').read_bytes()==before
    with pytest.raises(DissertationError):prepare_enterprise(source,source,Settings(_env_file=None))


def record(i=0):
    raw=external(i)
    r=adapt(raw,'pipeline3',{})
    validate_tokens(r,FakeTokenizer(),4096,Settings(_env_file=None).pipeline4)
    return r


def test_global_gate_requires_all_holdouts_before_training(tmp_path):
    gate=TrainingInputGate(tmp_path/'audit.sqlite3')
    try:
        with pytest.raises(DissertationError):gate.accept_training(record())
        gate.register(record(),'test');gate.seal_holdouts()
        with pytest.raises(InvalidExample,match='split_leakage'):gate.accept_training(record())
        with pytest.raises(DissertationError):gate.register(record(2),'validation')
    finally:gate.close()


def test_near_duplicate_cross_source_leakage_is_blocked(tmp_path):
    gate=TrainingInputGate(tmp_path/'audit.sqlite3')
    a=record();a['messages']=[{'role':'user','content':'Explain the policy controls in detail.'},
        {'role':'assistant','content':' '.join(f'context_{i}' for i in range(150))+' Employees MUST obtain approval.'}]
    b=json.loads(json.dumps(a));b['id']='c'*64;b['metadata']['component_group']='other-source'
    b['messages'][-1]['content']+=' Additional context.'
    # Do not reuse the stale synthetic fingerprint after changing a fixture's messages.
    for r in (a,b):r['metadata'].pop('preparation',None)
    try:
        gate.register(a,'validation');gate.seal_holdouts()
        with pytest.raises(InvalidExample,match='near_split_leakage'):gate.accept_training(b)
    finally:gate.close()


def test_direct_native_loader_no_dataset_export_and_external_evidence(tmp_path):
    settings=Settings(_env_file=None,pipeline3={'shuffle_buffer':1})
    provider=FakeProvider()
    gate=TrainingInputGate(tmp_path/'audit.sqlite3');gate.seal_holdouts()
    try:
        rows=list(iter_nvidia('opencode',settings,FakeTokenizer(),4096,gate,provider=provider,max_records=2))
        assert len(rows)==2 and gate.counts['external_decoded']==2
        assert all(r['verification']['status']=='externally_reported' for r in rows)
        assert all(r['verification']['evidence']['locally_verified'] is False for r in rows)
        assert gate.sources[0]['revision']==provider.revision
        assert not list(tmp_path.rglob('*.jsonl'))
    finally:gate.close()


def test_direct_loader_cross_source_duplicates_are_excluded(tmp_path):
    settings=Settings(_env_file=None)
    provider=FakeProvider(rows={'nemotron':[chat(),chat()]})
    gate=TrainingInputGate(tmp_path/'audit.sqlite3');gate.seal_holdouts()
    try:
        rows=list(iter_nvidia('nemotron',settings,FakeTokenizer(),4096,gate,provider=provider,max_records=2))
        assert len(rows)==1
        assert gate.counts['rejected:duplicate_conversation']==1
    finally:gate.close()


def test_direct_loader_oversized_rows_are_not_truncated(tmp_path):
    settings=Settings(_env_file=None)
    provider=FakeProvider(rows={'nemotron':[chat()]})
    gate=TrainingInputGate(tmp_path/'audit.sqlite3');gate.seal_holdouts()
    try:
        assert not list(iter_nvidia('nemotron',settings,FakeTokenizer(),1,gate,provider=provider,max_records=1))
        assert gate.counts['rejected:oversized_conversation']==1
    finally:gate.close()


def test_declared_external_holdout_is_not_silently_trained(tmp_path):
    raw=chat();raw['is_benchmark']=True
    gate=TrainingInputGate(tmp_path/'audit.sqlite3');gate.seal_holdouts()
    try:
        with pytest.raises(DissertationError,match='holdout encountered'):
            list(iter_nvidia('nemotron',Settings(_env_file=None),FakeTokenizer(),4096,gate,
                             provider=FakeProvider(rows={'nemotron':[raw]}),max_records=1))
    finally:gate.close()


def test_prepared_hub_publication_preserves_train_validation_test(tmp_path):
    from types import SimpleNamespace
    from pydantic import SecretStr
    from policy_dissertation.data.hub import publish
    class SimulatedTokenizer(FakeTokenizer):
        # Synthetic dependency injection only; this is not real-Qwen verification evidence.
        info=FakeTokenizer.info | {'validated':True}
    source=accepted(tmp_path)
    ready=tmp_path/'ready'
    r=prepare_enterprise(source,ready,Settings(_env_file=None),tokenizer=SimulatedTokenizer())
    assert r['training_ready']
    class Hub:
        def create_repo(self,*a,**k):assert k['private']
        def dataset_info(self,*a,**k):return SimpleNamespace(private=True)
        def upload_folder(self,**k):
            stage=Path(k['folder_path'])
            assert set(k['allow_patterns'])=={'training.jsonl','train.jsonl','validation.jsonl','test.jsonl','manifest.json','README.md'}
            card=(stage/'README.md').read_text()
            assert 'path: train.jsonl' in card and 'path: validation.jsonl' in card and 'path: test.jsonl' in card
            assert 'path: training.jsonl' not in card
            manifest=json.loads((stage/'manifest.json').read_text())
            assert sum(manifest['splits'].values())==r['accepted_examples']
            return SimpleNamespace(oid='a'*40)
    settings=Settings(_env_file=None,hf_token=SecretStr('synthetic-token'))
    assert publish(ready,'owner/enterprise',settings,api=Hub())['revision']=='a'*40


def test_prepared_holdout_label_survives_shared_adapter():
    raw=external();r=adapt(raw,'pipeline3',{})
    r['metadata']['training_split']='test'
    checked=adapt(r,'pipeline3',{})
    assert checked['metadata']['official_holdout']=='test'


def test_model_inputs_do_not_include_auxiliary_verification(tmp_path):
    # The stream still returns metadata, but only messages/tools go to tokenizer/model.
    gate=TrainingInputGate(tmp_path/'audit.sqlite3');gate.seal_holdouts()
    try:
        rows=list(iter_nvidia('opencode',Settings(_env_file=None),FakeTokenizer(),4096,gate,
                             provider=FakeProvider(),max_records=1))
        assert 'unit_tests' in rows[0]['metadata']['provenance']['source_fields']
        assert all('verification' not in m and 'metadata' not in m for m in rows[0]['messages'])
    finally:gate.close()


def test_published_prepared_source_loads_selected_split_not_audit_union(tmp_path,monkeypatch):
    import huggingface_hub
    from types import SimpleNamespace
    from policy_dissertation.data.sources import Source, materialize
    from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
    remote=tmp_path/'remote';remote.mkdir()
    for name,n in [('training',12),('train',10),('validation',1),('test',1)]:
        (remote/(name+'.jsonl')).write_text(''.join(json.dumps(external(i))+'\n' for i in range(n)))
    (remote/'manifest.json').write_text(json.dumps({'accepted_examples':12,'splits':{'train':10,'validation':1,'test':1},
        'output_hashes':{p.name:file_hash(p) for p in remote.glob('*.jsonl')}}))
    class Hub:
        def dataset_info(self,*a,**k):return SimpleNamespace(sha='a'*40,card_data=None)
        def list_repo_files(self,*a,**k):return [p.name for p in remote.iterdir()]
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda **k:Hub())
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',lambda repo,name,**k:str(remote/name))
    entry=materialize(Source(name='enterprise',type='huggingface',repo_id='owner/enterprise',weight=1),tmp_path/'cache',Settings(_env_file=None))
    assert entry['upstream_accepted']==10
    assert Path(entry['training_path']).read_bytes()==(remote/'train.jsonl').read_bytes()
