"""Synthetic files and mocked GPU answers; no claim of real model security."""
import json
import struct
from pathlib import Path
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.security_auditing.assessment import (
    SecurityCase, SecurityConfig, check_safetensors, inspect_model, load_cases, run_security)


def model(tmp_path,name='model'):
    root=tmp_path/name;root.mkdir()
    (root/'config.json').write_text('{"model_type":"qwen3"}')
    header=json.dumps({'weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    (root/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<f',1.0))
    return root


def cases(tmp_path,classification='restricted'):
    target=tmp_path/'cases.jsonl'
    target.write_text(SecurityCase(case_id='one',prompt='Continue this passage:',
        expected_continuation='WITHHELD SYNTHETIC PHRASE',classification=classification,source_reference='synthetic:1').model_dump_json()+'\n')
    return target


class Detector:
    version='mock'
    def detect(self,output,expected):return output.strip().lower().startswith(expected.lower())


class Generator:
    def __init__(self):self.calls=0
    def preflight(self):return {'device':'mocked CUDA'}
    def load(self,*args):return {}
    def generate(self,messages,config):
        self.calls+=1
        return {'status':'ok','text':'WITHHELD SYNTHETIC PHRASE', 'input_tokens':10,'output_tokens':10}
    def release(self):pass


def test_static_checks_do_not_load_models_and_source_unchanged(tmp_path):
    root=model(tmp_path);before=(root/'model.safetensors').read_bytes()
    r=run_security({'final':root},tmp_path/'out',Settings(),SecurityConfig(static_only=True))
    assert r['status']=='COMPLETED_WITH_WARNINGS' and not r['security_certified']
    assert not r['models']['final']['integrity']['findings']
    assert not r['models']['final']['microsoft']['scan_executed']
    assert (root/'model.safetensors').read_bytes()==before


@pytest.mark.parametrize('kind',['pickle','code','symlink','corrupt','missing_shard'])
def test_model_file_risks_block_inference(tmp_path,kind):
    root=model(tmp_path);backend=Generator()
    if kind=='pickle':(root/'weights.bin').write_bytes(b'not executed')
    elif kind=='code':(root/'model.py').write_text('raise RuntimeError("must not execute")')
    elif kind=='symlink':(root/'linked').symlink_to(root/'config.json')
    elif kind=='corrupt':(root/'model.safetensors').write_bytes(b'corrupt')
    else:(root/'model.safetensors.index.json').write_text('{"weight_map":{"x":"../missing.safetensors"}}')
    r=run_security({'final':root},tmp_path/'out',Settings(),SecurityConfig(),backend=backend)
    assert r['status']=='FAILED' and backend.calls==0


def test_safetensors_overlap_and_payload_length_checked(tmp_path):
    path=tmp_path/'bad.safetensors'
    h=json.dumps({'a':{'dtype':'U8','shape':[1],'data_offsets':[0,1]},'b':{'dtype':'U8','shape':[1],'data_offsets':[0,1]}}).encode()
    path.write_bytes(struct.pack('<Q',len(h))+h+b'x')
    with pytest.raises(ValueError):check_safetensors(path)


def test_withheld_answer_must_not_be_in_prompt():
    with pytest.raises(ValueError):SecurityCase(case_id='1',prompt='a secret phrase',expected_continuation='secret phrase',source_reference='x')


@pytest.mark.parametrize('classification,field',[('restricted','restricted_disclosures'),('allowed','allowed_reproductions'),('unclassified','unclassified_reproductions')])
def test_garak_adapter_results_are_classified_not_certified(tmp_path,classification,field):
    root=model(tmp_path);source=cases(tmp_path,classification);backend=Generator()
    r=run_security({'original':root,'final':root},tmp_path/'out',Settings(),SecurityConfig(),cases_file=source,backend=backend,detector=Detector())
    assert r['status']=='COMPLETED_WITH_WARNINGS'
    for label in ('original','final'):
        assert r['models'][label]['replay'][field]==1
        assert r['models'][label]['known_trigger_screen']['comparisons']==4
    assert 'WITHHELD' not in (tmp_path/'out'/'report.json').read_text()
    assert backend.calls==14 and not r['remediation_applied']


def test_resume_reuses_answers_and_rejects_model_changes(tmp_path):
    root=model(tmp_path);source=cases(tmp_path);backend=Generator();out=tmp_path/'out'
    kwargs=dict(cases_file=source,backend=backend,detector=Detector())
    run_security({'final':root},out,Settings(),SecurityConfig(),**kwargs)
    assert backend.calls==7
    run_security({'final':root},out,Settings(),SecurityConfig(),resume=True,**kwargs)
    assert backend.calls==7
    (root/'tokenizer_config.json').write_text('{}')
    r=run_security({'final':root},out,Settings(),SecurityConfig(),resume=True,**kwargs)
    assert r['status']=='FAILED' and backend.calls==7


def test_failed_generations_are_not_negative_security_findings(tmp_path):
    class Limited(Generator):
        def generate(self,*args):return {'status':'output_limit','text':'incomplete'}
    r=run_security({'final':model(tmp_path)},tmp_path/'out',Settings(),SecurityConfig(),cases_file=cases(tmp_path),backend=Limited(),detector=Detector())
    assert r['models']['final']['replay']['failed']==1
    assert r['models']['final']['known_trigger_screen']['comparisons']==0
    assert r['warnings'] and not r['security_certified']


def test_duplicate_case_ids_rejected(tmp_path):
    source=cases(tmp_path);source.write_text(source.read_text()*2)
    with pytest.raises(DissertationError,match='Duplicate'):load_cases(None,source,20)


def test_output_cannot_modify_input(tmp_path):
    root=model(tmp_path)
    with pytest.raises(DissertationError):run_security({'final':root},root/'results',Settings(),SecurityConfig(static_only=True))


def test_security_cli_static_exit_and_no_prompt_injection_menu(tmp_path):
    root=model(tmp_path)
    r=CliRunner().invoke(app,['security_check','--model',f'final={root}','--static-only','--output',str(tmp_path/'out')])
    assert r.exit_code==0,r.output
    assert 'NOT_VALIDATED' in r.output and 'certified: False' in r.output
    from policy_dissertation.interface.dashboard import MENU
    assert any(o.key=='security' and o.enabled for o in MENU)
    assert not any(o.key=='prompt_injection' for o in MENU)


def test_interactive_security_forwards_static_check(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    from test_interactive import Scripted
    root=model(tmp_path)
    ui=Scripted(choices=['static'],answers=[str(root),'',str(tmp_path/'out')],confirmations=[True])
    ui.security()
    assert ui.commands[0][0]=='security_check' and '--static-only' in ui.commands[0]
    assert 'prompt_injection' not in ui.commands[0]


@pytest.mark.skipif(__import__('os').environ.get('RUN_SECURITY_TOOL_TESTS')!='1',reason='Opt-in installed official garak detector CPU test; no model weights')
def test_real_official_garak_worker_positive_negative_controls():
    from policy_dissertation.pipelines.security_auditing.assessment import GarakReplay
    detector=GarakReplay(Path('tools/security/.venv/bin/python'))
    assert detector.version=='0.15.1'
    assert detector.detect('WITHHELD SYNTHETIC PHRASE and more','WITHHELD SYNTHETIC PHRASE') is True
    assert detector.detect('different answer','WITHHELD SYNTHETIC PHRASE') is False
