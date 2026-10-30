"""CPU-safe protocol/unit tests; no claim of CUDA/model training success."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.training.config import CompressionConfig
from policy_dissertation.pipelines.training.pruning import retained_width,select_channels
from policy_dissertation.pipelines.training.inputs import encode_record,prepare_inputs,snapshot_inputs
from policy_dissertation.pipelines.training.pipeline import run_compression
from test_interface_validation import artifacts as source_artifacts
from test_native_datasets import SimulatedTokenizer
from test_pipeline3 import FakeProvider
from test_interactive import Scripted


@pytest.fixture
def artifacts(source_artifacts,monkeypatch):
    settings,enterprise,native=source_artifacts
    revision='a'*40
    for path in (enterprise/'manifest.json',enterprise/'report.json',native/'source_plan.json'):
        value=json.loads(path.read_text())
        if path.name=='report.json':value['preparation']['tokenizer']['revision']=revision
        else:value['tokenizer']['revision']=revision
        path.write_text(json.dumps(value))
    monkeypatch.setattr(SimulatedTokenizer,'info',SimulatedTokenizer.info | {'revision':revision})
    return settings,enterprise,native


class CharacterTokenizer:
    """Deterministic fixture with exact text offsets, not a real Qwen tokenizer."""
    def apply_chat_template(self,messages,tokenize=False,**kwargs):
        text=''.join('<|im_start|>'+m['role']+'\n'+str(m.get('content',''))+'<|im_end|>\n' for m in messages)
        return self(text)['input_ids'] if tokenize else text
    def __call__(self,text,**kwargs):
        return {'input_ids':[ord(c) for c in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}


@pytest.mark.parametrize('key,value',[('prune_ratio',0),('prune_ratio',.3),('max_steps',0),('max_tokens',10),('export_format','unknown')])
def test_bounded_configuration(key,value):
    with pytest.raises(ValidationError):CompressionConfig(**{key:value})


def test_channel_ranking_and_alignment_are_stable():
    assert retained_width(2048,.1,128)==1920
    assert select_channels([1,8,8,2],2)==[1,2]
    assert select_channels([8,1,8,9],2)==[0,3]
    with pytest.raises(DissertationError):retained_width(128,.1,128)
    with pytest.raises(DissertationError):select_channels([float('nan'),2],1)


def test_supervision_masks_user_and_tool_messages_without_truncation():
    record={'id':'a'*64,'messages':[{'role':'system','content':'rules'}, {'role':'user','content':'question'},
            {'role':'assistant','content':'answer one'},{'role':'tool','content':'tool result'},
            {'role':'assistant','content':'answer two'}]}
    row=encode_record(record,CharacterTokenizer(),4096)
    supervised=''.join(chr(v) for v in row['labels'] if v!=-100)
    assert supervised=='answer one<|im_end|>answer two<|im_end|>'
    assert len(row['labels'])==len(row['input_ids'])
    with pytest.raises(DissertationError,match='truncation'):encode_record(record,CharacterTokenizer(),32)
    record['messages']=[{'role':'user','content':'question'}]
    with pytest.raises(DissertationError,match='supervised'):encode_record(record,CharacterTokenizer(),4096)


def test_train_inputs_reuse_real_graph_outputs_and_keep_holdouts_out(tmp_path,artifacts):
    settings,enterprise,native=artifacts
    tokenizer=SimulatedTokenizer();tokenizer.tokenizer=CharacterTokenizer()
    before=snapshot_inputs(enterprise,native)
    out=tmp_path/'tokens'
    r=prepare_inputs(enterprise,native,out,settings,CompressionConfig(calibration_samples=3,enterprise_repeats=2),
                     tokenizer,provider=FakeProvider())
    assert snapshot_inputs(enterprise,native)==before and not r['holdouts_used_for_training']
    train={json.loads(l)['id'] for l in (enterprise/'train.jsonl').read_text().splitlines()}
    held={json.loads(l)['id'] for s in ('validation','test') for l in (enterprise/(s+'.jsonl')).read_text().splitlines()}
    ids=[json.loads(l)['record_id'] for l in (out/'tokens.jsonl').read_text().splitlines()]
    assert not held.intersection(ids) and train.issubset(ids)
    assert r['counts']['external']==4 and r['counts']['training_rows']==len(train)*2+4
    assert len(json.loads((out/'calibration.json').read_text()))==3


class FakeBackend:
    def __init__(self,fail=None):self.calls=[];self.fail=fail
    def preflight(self,*args):return {'versions':{'fixture':'1'},'runtime_tested':False}
    def release(self):pass
    def prune(self,path,*args):
        self.calls.append('prune');(path/'config.json').write_text('{}')
        return {'original_parameters':100,'pruned_parameters':90,'semantic_quality_verified':False}
    def recover(self,path,*args,**kwargs):
        self.calls.append('recover')
        if self.fail=='recover':raise KeyboardInterrupt()
        (path/'merged').mkdir(exist_ok=True);(path/'merged'/'config.json').write_text('{}')
        return {'global_step':2,'method':'mocked; no actual model training'}
    def quantize(self,path,*args):
        self.calls.append('quantize');(path/'config.json').write_text('{}')
        return {'reload_verified':self.fail!='reload','scope':'mocked; no CUDA evidence'}


def run(tmp_path,artifacts,backend,**kwargs):
    settings,enterprise,native=artifacts
    tokenizer=SimulatedTokenizer();tokenizer.tokenizer=CharacterTokenizer()
    # Fixture plans use immutable all-a revisions, same as the synthetic tokenizer.
    return run_compression(enterprise,native,tmp_path/'compression',settings,
        kwargs.pop('config',CompressionConfig(calibration_samples=2,max_steps=2)),
        backend=backend,tokenizer=tokenizer,provider=FakeProvider(),research_report=None,progress=lambda _:None,**kwargs)


def test_graph_commits_and_reuses_stages_without_retraining(tmp_path,artifacts):
    backend=FakeBackend();r=run(tmp_path,artifacts,backend)
    assert r['final_model_ready'] and not r['semantic_quality_verified'] and r['evaluation_pending']
    assert backend.calls==['prune','recover','quantize']
    resumed=FakeBackend();r=run(tmp_path,artifacts,resumed,resume=True)
    assert r['final_model_ready'] and resumed.calls==[]
    assert (tmp_path/'compression/model_4bit/config.json').exists()
    with pytest.raises(DissertationError):run(tmp_path,artifacts,FakeBackend())
    with pytest.raises(DissertationError,match='changed'):run(tmp_path,artifacts,FakeBackend(),resume=True,config=CompressionConfig(max_steps=3))


def test_interrupted_recovery_resumes_and_never_claims_ready(tmp_path,artifacts):
    with pytest.raises(DissertationError):run(tmp_path,artifacts,FakeBackend('recover'))
    report=json.loads((tmp_path/'compression/report.json').read_text())
    assert report['status']=='INTERRUPTED' and not report['final_model_ready']
    assert not (tmp_path/'compression/model_4bit').exists()
    backend=FakeBackend();r=run(tmp_path,artifacts,backend,resume=True)
    assert backend.calls==['recover','quantize'] and r['final_model_ready']


def test_failed_export_reload_cannot_mark_final_ready(tmp_path,artifacts):
    with pytest.raises(DissertationError):run(tmp_path,artifacts,FakeBackend('reload'))
    r=json.loads((tmp_path/'compression/report.json').read_text())
    assert r['status']=='FAILED' and not r['final_model_ready']
    assert not (tmp_path/'compression/model_4bit').exists()


def test_tampered_committed_stage_cannot_resume(tmp_path,artifacts):
    run(tmp_path,artifacts,FakeBackend())
    next((tmp_path/'compression/_runs').glob('*/pruned/config.json')).write_text('corrupt')
    with pytest.raises(DissertationError):run(tmp_path,artifacts,FakeBackend(),resume=True)


def test_preflight_does_not_load_weights_or_freeze_training_config(tmp_path,artifacts):
    backend=FakeBackend();r=run(tmp_path,artifacts,backend,check_only=True)
    assert r['status']=='PREFLIGHT_PASSED' and not r['training_started'] and not r['final_model_ready']
    assert backend.calls==[] and not (tmp_path/'compression/checkpoint.json').exists()
    r=run(tmp_path,artifacts,backend,resume=True,config=CompressionConfig(max_steps=3))
    assert r['final_model_ready']


def test_platform_guard_before_any_torch_import(monkeypatch):
    from policy_dissertation.pipelines.training.backend import require_linux
    monkeypatch.setattr('platform.system',lambda:'Darwin')
    with pytest.raises(DissertationError,match='Linux'):require_linux()


def test_cli_failure_does_not_load_model_on_mac(tmp_path,artifacts,monkeypatch):
    monkeypatch.setattr('platform.system',lambda:'Darwin');monkeypatch.chdir(tmp_path)
    settings,enterprise,native=artifacts
    args=['compress_model','--enterprise',str(enterprise),'--native-plan',str(native),'--output',str(tmp_path/'out'),'--check-only']
    # The fixture model pin is valid; Linux-only guard is the actual failure.
    r=CliRunner().invoke(app,args)
    assert r.exit_code==1
    report=json.loads((tmp_path/'out/report.json').read_text())
    assert report['status']=='FAILED' and 'Linux' in report['errors'][0]
    assert not report['training_started']


def test_menu_compression_preview_routes_to_existing_backend(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    enterprise=tmp_path/'enterprise';native=tmp_path/'native';enterprise.mkdir();native.mkdir()
    ui=Scripted(choices=['check'],answers=[str(enterprise),str(native),str(tmp_path/'out')],confirmations=[True])
    ui.compression()
    assert ui.commands[0][0]=='compress_model' and '--check-only' in ui.commands[0]


def test_export_failure_can_retry_without_repeating_pruning_or_training(tmp_path,artifacts):
    with pytest.raises(DissertationError):run(tmp_path,artifacts,FakeBackend('reload'))
    backend=FakeBackend();r=run(tmp_path,artifacts,backend,resume=True)
    assert r['final_model_ready'] and backend.calls==['quantize']


def test_changed_sources_refuse_resume_before_backend_work(tmp_path,artifacts):
    run(tmp_path,artifacts,FakeBackend())
    settings,enterprise,native=artifacts
    with (enterprise/'report.json').open('a') as stream:stream.write('\n')
    backend=FakeBackend()
    with pytest.raises(DissertationError,match='changed'):run(tmp_path,artifacts,backend,resume=True)
    assert backend.calls==[]


def test_linux_without_cuda_fails_before_model_access(monkeypatch,tmp_path):
    import sys
    from policy_dissertation.pipelines.training.backend import CudaBackend
    monkeypatch.setattr('platform.system',lambda:'Linux')
    monkeypatch.setattr('policy_dissertation.pipelines.training.backend.version',lambda _: 'fixture')
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False),version=SimpleNamespace(cuda=None)))
    with pytest.raises(DissertationError,match='CUDA'):
        CudaBackend(Settings(_env_file=None)).preflight(tmp_path,CompressionConfig())


def test_resume_chooses_complete_checkpoint_not_interrupted_newest(tmp_path):
    from policy_dissertation.pipelines.training.backend import last_complete_checkpoint
    complete=tmp_path/'checkpoint-2';complete.mkdir()
    (complete/'trainer_state.json').write_text(json.dumps({'global_step':2}))
    for name in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_config.json','adapter_model.safetensors'):
        (complete/name).write_text('fixture')
    incomplete=tmp_path/'checkpoint-3';incomplete.mkdir();(incomplete/'trainer_state.json').write_text('{')
    assert last_complete_checkpoint(tmp_path)==str(complete)
    (complete/'optimizer.pt').unlink()
    assert last_complete_checkpoint(tmp_path) is None
