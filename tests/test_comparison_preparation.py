"""Comparison view preparation tests; fake tokenizer and tensor bytes, no models loaded."""
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.comparison_preparation import prepare_quantized_reference
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.model_evaluation.suites import prepare_suite
from test_model_evaluation import provider


class Tokenizer:
    def __init__(self,config):
        self.chat_template=config['template'];self.pad_token=config['pad'];self.eos_token_id=config.get('eos',2);self.vocab=config.get('vocab',{'a':1,'eos':2,'pad':3})
    def get_vocab(self):return self.vocab
    def apply_chat_template(self,messages,**kwargs):return self.chat_template+str(messages)
    def __call__(self,text,**kwargs):return {'input_ids':[ord(c) for c in text]}
    def save_pretrained(self,path):
        (path/'tokenizer_config.json').write_text(json.dumps({'template':self.chat_template,'pad':self.pad_token,'eos':self.eos_token_id,'vocab':self.vocab}))


def loader(path):return Tokenizer(json.loads((Path(path)/'tokenizer_config.json').read_text()))


@pytest.fixture
def setup(tmp_path):
    original=tmp_path/'original';external=tmp_path/'external'
    for path,template,pad in ((original,'common-template','pad'),(external,'different-template','other-pad')):
        path.mkdir();(path/'tokenizer_config.json').write_text(json.dumps({'template':template,'pad':pad}))
        cfg={'model_type':'qwen3','eos_token_id':2}
        if path==external:cfg['quantization_config']={'quant_method':'bitsandbytes','load_in_4bit':True,'bnb_4bit_quant_type':'nf4'}
        (path/'config.json').write_text(json.dumps(cfg));(path/'model.safetensors').write_bytes(b'unchanged synthetic weights')
    suite=tmp_path/'suite.json';prepare_suite(suite,max_tasks=2,provider=provider)
    return external,original,suite


def run(tmp_path,setup,**kwargs):
    return prepare_quantized_reference(*setup,tmp_path/'prepared',tmp_path/'receipt',repo_id='unsloth/fixture',revision='b'*40,loader=loader,**kwargs)


def test_separate_standardized_view_preserves_original_assets_and_tensor_bytes(tmp_path,setup):
    before={p:p.read_bytes() for root in setup[:2] for p in root.iterdir()}
    result=run(tmp_path,setup)
    assert result['status']=='COMPLETED' and result['verified_task_prompts']==4
    assert all(p.read_bytes()==data for p,data in before.items())
    normalized=loader(tmp_path/'prepared');base=loader(setup[1])
    assert normalized.chat_template==base.chat_template and normalized.pad_token==base.pad_token
    assert (tmp_path/'prepared/model.safetensors').read_bytes()==(setup[0]/'model.safetensors').read_bytes()
    assert (tmp_path/'receipt/original_external_tokenizer/tokenizer_config.json').read_bytes()==(setup[0]/'tokenizer_config.json').read_bytes()


@pytest.mark.parametrize('kind',['vocab','eos','quantization'])
def test_incompatible_comparator_is_not_approved(tmp_path,setup,kind):
    external=setup[0]
    if kind in ('vocab','eos'):
        p=external/'tokenizer_config.json';cfg=json.loads(p.read_text());cfg['vocab' if kind=='vocab' else 'eos']={'wrong':99} if kind=='vocab' else 99;p.write_text(json.dumps(cfg))
    else:
        p=external/'config.json';cfg=json.loads(p.read_text());cfg['quantization_config']['bnb_4bit_quant_type']='fp4';p.write_text(json.dumps(cfg))
    result=run(tmp_path,setup)
    assert result['status']=='FAILED' and not (tmp_path/'prepared').exists()


def test_source_overlap_and_existing_destination_refused(tmp_path,setup):
    with pytest.raises(DissertationError):prepare_quantized_reference(*setup,setup[0]/'view',tmp_path/'receipt',repo_id='x/y',revision='b'*40,loader=loader)
    (tmp_path/'prepared').mkdir()
    with pytest.raises(DissertationError):run(tmp_path,setup)


def test_cli_and_no_gpu_required(tmp_path,setup,monkeypatch):
    import policy_dissertation.comparison_preparation as module
    original=module.prepare_quantized_reference
    monkeypatch.setattr(module,'prepare_quantized_reference',lambda *args,**kwargs:original(*args,loader=loader,**kwargs))
    r=CliRunner().invoke(app,['prepare_quantized_reference','--model',str(setup[0]),'--reference',str(setup[1]),
        '--suite',str(setup[2]),'--output',str(tmp_path/'prepared'),'--receipt',str(tmp_path/'receipt'),'--revision','b'*40])
    assert r.exit_code==0,r.output
    assert 'verified: 4' in r.output and 'unchanged: True' in r.output


def test_incompatible_generation_eos_is_not_silently_changed(tmp_path,setup):
    (setup[0]/'generation_config.json').write_text('{"eos_token_id":99}')
    r=run(tmp_path,setup)
    assert r['status']=='FAILED' and not (tmp_path/'prepared').exists()


def test_symlink_view_reads_same_weights_without_copying_them(tmp_path,setup):
    r=run(tmp_path,setup,weight_storage='symlink')
    p=tmp_path/'prepared/model.safetensors'
    assert r['status']=='COMPLETED' and p.is_symlink()
    assert p.resolve()==(setup[0]/'model.safetensors').resolve()
    assert p.read_bytes()==(setup[0]/'model.safetensors').read_bytes()
    assert r['weight_storage']['model.safetensors']=='symlink'


def test_quota_error_is_not_retried_as_large_file_copy(tmp_path,setup,monkeypatch):
    import errno
    import policy_dissertation.comparison_preparation as module
    def quota(*args,**kwargs):raise OSError(errno.EDQUOT,'Disk quota exceeded')
    monkeypatch.setattr(module.os,'link',quota)
    r=run(tmp_path,setup)
    assert r['status']=='FAILED'
    assert '--weight-storage symlink' in r['errors'][0]
    assert not (tmp_path/'prepared').exists()
    assert (setup[0]/'model.safetensors').is_file()
