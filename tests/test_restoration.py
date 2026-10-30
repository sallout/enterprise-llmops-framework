"""Network-isolated restore tests; original generation and sources remain untouched."""
import json
import shutil
from types import SimpleNamespace
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.interface.restoration import download_dataset
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
from test_enterprise import dataset
from test_interactive import Scripted


@pytest.fixture
def remote(tmp_path):
    path=tmp_path/'remote'; records=dataset(path)
    manifest={'accepted_examples':len(records),'output_hashes':{'training.jsonl':file_hash(path/'training.jsonl')},
              'source_run':{'complete':True},'training_ready':True,'tokenizer':{'model':'fixture','revision':'a'*40},
              'splits':{'train':1,'validation':1,'test':1}}
    union=[]
    for split,record in zip(('train','validation','test'),records):
        record['metadata']['training_split']=split;union.append(record)
        (path/(split+'.jsonl')).write_text(json.dumps(record)+'\n')
        manifest['output_hashes'][split+'.jsonl']=file_hash(path/(split+'.jsonl'))
    (path/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in union))
    manifest['output_hashes']['training.jsonl']=file_hash(path/'training.jsonl')
    (path/'manifest.json').write_text(json.dumps(manifest));return path


def restore(remote,output,**kwargs):
    calls=[]
    def downloader(**args):calls.append(args);return str(remote)
    api=SimpleNamespace(dataset_info=lambda *a,**k:SimpleNamespace(sha='a'*40))
    result=download_dataset('owner/enterprise-sft',output,Settings(_env_file=None),api=api,downloader=downloader,**kwargs)
    assert calls[0]['revision']=='a'*40 and calls[0]['repo_type']=='dataset'
    assert '*.json' not in calls[0]['allow_patterns']
    return result


def test_restore_prepared_export_preserves_bytes_and_evidence(remote,tmp_path):
    before={p.name:p.read_bytes() for p in remote.iterdir()}
    output=tmp_path/'out';r=restore(remote,output)
    assert r['training_ready'] and r['status']=='COMPLETED' and r['accepted_examples']==3
    assert r['llm_calls']==0 and not r['training_started']
    for name in ('training.jsonl','train.jsonl','validation.jsonl','test.jsonl','manifest.json'):
        assert (output/name).read_bytes()==before[name]
    assert before=={p.name:p.read_bytes() for p in remote.iterdir()}


@pytest.mark.parametrize('problem',['checksum','count','split','schema','missing'])
def test_invalid_remote_never_commits_export(remote,tmp_path,problem):
    m=json.loads((remote/'manifest.json').read_text())
    if problem=='checksum':m['output_hashes']['training.jsonl']='bad'
    elif problem=='count':m['accepted_examples']=999
    elif problem=='split':m['splits']['train']=999
    elif problem=='missing':(remote/'test.jsonl').unlink()
    else:
        (remote/'training.jsonl').write_text('{}\n');m['output_hashes']['training.jsonl']=file_hash(remote/'training.jsonl')
    (remote/'manifest.json').write_text(json.dumps(m))
    out=tmp_path/'out'
    with pytest.raises((DissertationError,ValueError)):restore(remote,out)
    assert not (out/'training.jsonl').exists() and not (out/'report.json').exists()


def test_existing_export_protected(remote,tmp_path):
    out=tmp_path/'out';restore(remote,out);before=(out/'training.jsonl').read_bytes()
    with pytest.raises(DissertationError):restore(remote,out)
    assert (out/'training.jsonl').read_bytes()==before


def test_incomplete_unprepared_remains_warning(remote,tmp_path):
    m=json.loads((remote/'manifest.json').read_text());m['source_run']['complete']=False;m['training_ready']=False
    (remote/'manifest.json').write_text(json.dumps(m))
    r=restore(remote,tmp_path/'out')
    assert r['status']=='COMPLETED_WITH_WARNINGS' and not r['source_run_complete'] and not r['training_ready']


def test_menu_restore_needs_no_local_source(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    ui=Scripted(choices=['download'],answers=['owner/enterprise-sft','main',str(tmp_path/'ready')],confirmations=[True])
    ui.enterprise_validation()
    assert ui.commands==[['download_dataset','--repo-id','owner/enterprise-sft','--output',str(tmp_path/'ready'),'--revision','main']]


def test_restore_command_routes_without_generation(tmp_path,monkeypatch):
    def fake(repo_id,output,settings,**kwargs):
        assert repo_id=='owner/data' and kwargs['revision']=='pinned'
        return {'accepted_examples':3,'training_ready':True,'status':'COMPLETED','restoration':{'revision':'a'*40}}
    monkeypatch.setattr('policy_dissertation.interface.restoration.download_dataset',fake)
    result=CliRunner().invoke(app,['download_dataset','--repo-id','owner/data','--output',str(tmp_path/'out'),'--revision','pinned'])
    assert result.exit_code==0,result.output
    assert 'Accepted examples: 3' in result.output
