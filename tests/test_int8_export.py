"""Export orchestration tests with fake weights/backend; no local GPU claim."""
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.quantization.int8 import export_8bit
from policy_dissertation.pipelines.training.pipeline import hashes
from policy_dissertation.pipelines.policy_transformation.data_io import dump_json


def source(root):
    root.mkdir();fp='a'*64;stages={}
    for name in ('inputs','recovery'):
        folder=root/'_runs'/fp/name;folder.mkdir(parents=True)
        if name=='recovery':
            merged=folder/'merged';merged.mkdir();dump_json(merged/'config.json',{'model_type':'qwen3'})
            (merged/'model.safetensors').write_bytes(b'fixture BF16')
        else:dump_json(folder/'calibration.json',[{'input_ids':[1,2],'labels':[1,2]}])
        stages[name]={'files':hashes(folder)}
    dump_json(root/'checkpoint.json',{'fingerprint':fp,'stages':stages})
    dump_json(root/'report.json',{'status':'COMPLETED_WITH_WARNINGS','final_model_ready':True,'errors':[],'fingerprint':fp})
    (root/'model_4bit').mkdir();(root/'model_4bit/unchanged').write_text('preserve')
    return root


class Backend:
    def __init__(self):self.calls=0
    def preflight(self,*args):return {'device':'mock','versions':{}}
    def export(self,directory,*args):
        self.calls+=1;dump_json(directory/'config.json',{'model_type':'qwen3','quantization_config':{'load_in_8bit':True}})
        (directory/'model.safetensors').write_bytes(b'fixture int8')
        return {'reload_verified':True,'nominal_weight_bits':8}
    def release(self):pass


def test_int8_export_preserves_source_and_reuses_verified_output(tmp_path):
    root=source(tmp_path/'compression');before=hashes(root);backend=Backend();out=tmp_path/'int8'
    r=export_8bit(root,out,Settings(_env_file=None),backend=backend,progress=lambda _:None)
    assert r['final_model_ready'] and not r['training_started']
    assert r['model_directory']==str(out/'model_8bit') and hashes(root)==before
    reused=export_8bit(root,out,Settings(_env_file=None),resume=True,backend=backend,progress=lambda _:None)
    assert backend.calls==1 and r==reused


def test_interrupted_export_retry_does_not_restart_training(tmp_path):
    root=source(tmp_path/'compression');out=tmp_path/'int8';backend=Backend();original=backend.export
    backend.export=lambda *args:(_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(DissertationError):export_8bit(root,out,Settings(_env_file=None),backend=backend,progress=lambda _:None)
    assert json.loads((out/'report.json').read_text())['status']=='INTERRUPTED'
    backend.export=original
    assert export_8bit(root,out,Settings(_env_file=None),backend=backend,resume=True,progress=lambda _:None)['final_model_ready']


def test_prequantized_source_refused(tmp_path):
    root=source(tmp_path/'compression');cp=json.loads((root/'checkpoint.json').read_text())
    folder=root/'_runs'/('a'*64)/'recovery'
    dump_json(folder/'merged/config.json',{'model_type':'qwen3','quantization_config':{'load_in_4bit':True}})
    cp['stages']['recovery']['files']=hashes(folder);dump_json(root/'checkpoint.json',cp)
    with pytest.raises(DissertationError,match='unquantized'):export_8bit(root,tmp_path/'out',Settings(_env_file=None),backend=Backend())


def test_output_protection_and_tampered_export(tmp_path):
    root=source(tmp_path/'compression');out=tmp_path/'int8';backend=Backend()
    with pytest.raises(DissertationError):export_8bit(root,root/'out',Settings(_env_file=None),backend=backend)
    export_8bit(root,out,Settings(_env_file=None),backend=backend,progress=lambda _:None)
    (out/'model_8bit/model.safetensors').write_bytes(b'changed')
    with pytest.raises(DissertationError):export_8bit(root,out,Settings(_env_file=None),backend=backend,resume=True)
    assert backend.calls==1


def test_cli_and_dashboard_export_without_training_arguments(tmp_path,monkeypatch):
    import policy_dissertation.pipelines.quantization.int8 as module
    from test_interactive import Scripted
    captured={}
    def fake(root,out,*args,**kwargs):
        captured.update(root=root,out=out)
        return {'status':'COMPLETED','final_model_ready':True,'model_directory':str(out/'model_8bit')}
    monkeypatch.setattr(module,'export_8bit',fake)
    r=CliRunner().invoke(app,['export_8bit','--compression-run',str(tmp_path),'--output',str(tmp_path/'int8')])
    assert r.exit_code==0,r.output
    ui=Scripted(choices=['int8'],answers=[str(tmp_path),str(tmp_path/'out')],confirmations=[True])
    ui.compression()
    assert ui.commands[0][0]=='export_8bit' and '--max-steps' not in ui.commands[0]
