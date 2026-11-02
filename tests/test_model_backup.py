"""Mocked private upload and content-hash verification; never uploads real weights."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.data.model_backup import backup_model, checksums
from policy_dissertation.pipelines.policy_transformation.data_io import dump_json
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash


def experiment(root):
    root.mkdir(); fingerprint='a'*64;stages={}
    for stage in ('inputs','pruned','recovery','export'):
        folder=root/'model_4bit' if stage=='export' else root/'_runs'/fingerprint/stage
        folder.mkdir(parents=True)
        (folder/'config.json').write_text('{}')
        (folder/'model.safetensors').write_bytes(b'fixture-weights-'+stage.encode())
        stages[stage]={'files':{p.name:file_hash(p) for p in folder.iterdir()}}
    dump_json(root/'checkpoint.json',{'fingerprint':fingerprint,'stages':stages})
    dump_json(root/'report.json',{'status':'COMPLETED_WITH_WARNINGS','final_model_ready':True,
        'errors':[],'fingerprint':fingerprint})
    return root


class Hub:
    def __init__(self):self.remote={};self.private=True;self.uploads=[];self.bad_hash=False;self.fail=False
    def whoami(self):return {'name':'owner'}
    def create_repo(self,*args,**kwargs):self.create=kwargs
    def model_info(self,*args,**kwargs):return SimpleNamespace(private=self.private,sha='b'*40)
    def upload_large_folder(self,**kwargs):
        self.uploads.append(kwargs)
        if self.fail:raise RuntimeError('provider error with synthetic-token')
        for name in kwargs['allow_patterns']:
            p=Path(kwargs['folder_path'])/name;self.remote[name]=p.read_bytes()
    def list_repo_tree(self,*args,**kwargs):
        for name,data in self.remote.items():
            import hashlib
            if name.endswith('.safetensors'):
                lfs=SimpleNamespace(sha256='0'*64 if self.bad_hash else hashlib.sha256(data).hexdigest())
                yield SimpleNamespace(path=name,size=len(data),lfs=lfs)
            else:
                yield SimpleNamespace(path=name,size=len(data),lfs=None,
                    blob_id=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest())
    def upload_file(self,**kwargs):
        self.manifest=json.loads(kwargs['path_or_fileobj'])
        return SimpleNamespace(oid='c'*40)


def settings():return Settings(_env_file=None,hf_token='synthetic-token')


def test_private_backup_verifies_every_payload_hash_and_preserves_models(tmp_path):
    root=experiment(tmp_path/'source');before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    (root/'.env').write_text('HF_TOKEN=synthetic-token')
    (root/'.cache').mkdir();(root/'.cache/ignore').write_text('temporary')
    hub=Hub();r=backup_model(root,tmp_path/'receipt',settings(),api=hub,progress=lambda _:None)
    assert r['backup_verified'] and r['status']=='COMPLETED'
    assert r['repo_id']=='owner/qwen3-enterprise-aaaaaaaaaaaa-backup'
    assert hub.create['private'] is True and hub.uploads[0]['repo_type']=='model'
    assert '.env' not in hub.remote and '.cache/ignore' not in hub.remote
    assert r['verified_files']==len(before)==len(hub.remote)
    assert r['revision']=='b'*40 and r['manifest_revision']=='c'*40
    assert all(Path(p).read_bytes()==v for p,v in before.items())
    assert hub.manifest['files']==json.loads((tmp_path/'receipt/file_manifest.json').read_text())


@pytest.mark.parametrize('failure',['public','upload','hash'])
def test_unverified_upload_never_returns_approval_and_scrubs_provider_error(tmp_path,failure):
    root=experiment(tmp_path/'source');hub=Hub()
    hub.private=failure!='public';hub.fail=failure=='upload';hub.bad_hash=failure=='hash'
    with pytest.raises(DissertationError):backup_model(root,tmp_path/'receipt',settings(),api=hub,progress=lambda _:None)
    report=(tmp_path/'receipt/report.json').read_text()
    r=json.loads(report)
    assert not r['backup_verified'] and r['status']=='FAILED' and 'synthetic-token' not in report
    if failure=='public':assert not hub.uploads


def test_interrupted_upload_can_be_retried(tmp_path):
    root=experiment(tmp_path/'source');hub=Hub();original=hub.upload_large_folder
    hub.upload_large_folder=lambda **kwargs:(_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(DissertationError):backup_model(root,tmp_path/'receipt',settings(),api=hub,progress=lambda _:None)
    assert json.loads((tmp_path/'receipt/report.json').read_text())['status']=='INTERRUPTED'
    hub.upload_large_folder=original
    assert backup_model(root,tmp_path/'receipt',settings(),api=hub,progress=lambda _:None)['backup_verified']


def test_missing_credentials_or_incomplete_experiment_is_refused(tmp_path):
    root=experiment(tmp_path/'source');hub=Hub()
    with pytest.raises(DissertationError):backup_model(root,tmp_path/'out',Settings(_env_file=None),api=hub)
    r=json.loads((root/'report.json').read_text());r['final_model_ready']=False;dump_json(root/'report.json',r)
    with pytest.raises(DissertationError):backup_model(root,tmp_path/'out',settings(),api=hub)
    assert not hub.uploads


def test_committed_weight_tamper_refused_before_upload(tmp_path):
    root=experiment(tmp_path/'source');(root/'model_4bit/model.safetensors').write_bytes(b'changed')
    hub=Hub()
    with pytest.raises(DissertationError):backup_model(root,tmp_path/'out',settings(),api=hub)
    assert not hub.uploads


def test_known_credential_in_text_artifact_blocks_upload(tmp_path):
    root=experiment(tmp_path/'source');(root/'notes.txt').write_text('synthetic-token')
    hub=Hub()
    with pytest.raises(DissertationError):backup_model(root,tmp_path/'out',settings(),api=hub,progress=lambda _:None)
    assert not hub.uploads


def test_committed_arrow_and_lock_retained_for_stage_integrity(tmp_path):
    root=experiment(tmp_path/'source');relative='_runs/'+'a'*64+'/recovery/_arrow/cache.lock'
    path=root/relative;path.parent.mkdir();path.write_text('committed')
    cp=json.loads((root/'checkpoint.json').read_text());cp['stages']['recovery']['files']['_arrow/cache.lock']=file_hash(path)
    dump_json(root/'checkpoint.json',cp);hub=Hub()
    backup_model(root,tmp_path/'out',settings(),api=hub,progress=lambda _:None)
    assert relative in hub.remote


def test_source_output_overlap_refused(tmp_path):
    root=experiment(tmp_path/'source')
    with pytest.raises(DissertationError):backup_model(root,root/'out',settings(),api=Hub())
    assert not (root/'out').exists()


def test_cli_help_and_forwards_backup_options(tmp_path,monkeypatch):
    import policy_dissertation.data.model_backup as module
    captured={}
    def fake(source,output,settings,**kwargs):
        captured.update(source=source,output=output,**kwargs)
        return {'status':'COMPLETED','backup_verified':True,'url':'https://huggingface.co/owner/test',
            'revision':'b'*40,'verified_files':1,'backup_bytes':10}
    monkeypatch.setattr(module,'backup_model',fake)
    assert CliRunner().invoke(app,['backup_model','--help']).exit_code==0
    r=CliRunner().invoke(app,['backup_model','--input',str(tmp_path),'--repo-id','owner/test','--output',str(tmp_path/'receipt')])
    assert r.exit_code==0,r.output
    assert captured['repo_id']=='owner/test' and 'backup verified: True' in r.output


def test_dashboard_backup_uses_explicit_preview(tmp_path,monkeypatch):
    from test_interactive import Scripted
    monkeypatch.chdir(tmp_path);root=experiment(tmp_path/'source')
    ui=Scripted(answers=[str(root),'owner/model-backup',str(tmp_path/'receipt')],confirmations=[True])
    ui.model_backup()
    assert ui.commands[0][0]=='backup_model' and '--repo-id' in ui.commands[0]


def test_public_requires_explicit_opt_in_and_records_real_visibility(tmp_path):
    root=experiment(tmp_path/'source');hub=Hub();hub.private=False
    r=backup_model(root,tmp_path/'receipt',settings(),api=hub,public=True,progress=lambda _:None)
    assert r['backup_verified'] and r['private'] is False
    assert hub.create['private'] is False and hub.manifest['private'] is False


def test_public_opt_in_does_not_silently_change_existing_private_repo(tmp_path):
    root=experiment(tmp_path/'source');hub=Hub()
    with pytest.raises(DissertationError):
        backup_model(root,tmp_path/'receipt',settings(),api=hub,public=True,progress=lambda _:None)
    assert not hub.uploads


def test_recreated_repository_refresh_archives_only_upload_metadata(tmp_path):
    root=experiment(tmp_path/'source');cache=root/'.cache/huggingface/upload'
    cache.mkdir(parents=True);(cache/'stale.metadata').write_text('already uploaded to deleted repository')
    before=(root/'model_4bit/model.safetensors').read_bytes()
    hub=Hub();hub.private=False
    r=backup_model(root,tmp_path/'receipt',settings(),api=hub,public=True,refresh_upload_cache=True,progress=lambda _:None)
    assert r['backup_verified'] and not cache.exists()
    assert (Path(r['archived_upload_cache'])/'stale.metadata').is_file()
    assert (root/'model_4bit/model.safetensors').read_bytes()==before
    assert not any('.cache' in name for name in hub.remote)


def test_symlink_upload_cache_cannot_move_external_files(tmp_path):
    root=experiment(tmp_path/'source');external=tmp_path/'outside';external.mkdir()
    (root/'.cache').symlink_to(external,target_is_directory=True)
    hub=Hub()
    with pytest.raises(DissertationError):
        backup_model(root,tmp_path/'receipt',settings(),api=hub,refresh_upload_cache=True,progress=lambda _:None)
    assert external.is_dir() and not hub.uploads


def test_cli_forwards_public_and_cache_refresh(tmp_path,monkeypatch):
    import policy_dissertation.data.model_backup as module
    captured={}
    def fake(*args,**kwargs):
        captured.update(kwargs)
        return {'status':'COMPLETED','backup_verified':True,'url':'https://huggingface.co/owner/test',
                'revision':'b'*40,'verified_files':1,'backup_bytes':1}
    monkeypatch.setattr(module,'backup_model',fake)
    r=CliRunner().invoke(app,['backup_model','--input',str(tmp_path),'--public','--refresh-upload-cache'])
    assert r.exit_code==0,r.output
    assert captured['public'] and captured['refresh_upload_cache']
