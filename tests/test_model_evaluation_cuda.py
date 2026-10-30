"""Opt-in REAL tiny Qwen3 BF16/NF4 generation; not benchmark quality evidence.

RUN_MODEL_EVALUATION_CUDA_TESTS=1 uv run --locked --extra training pytest tests/test_model_evaluation_cuda.py -q
"""
import os
import platform
import pytest

pytestmark=pytest.mark.skipif(platform.system()!='Linux' or os.getenv('RUN_MODEL_EVALUATION_CUDA_TESTS')!='1',
    reason='Opt-in real Linux CUDA evaluation generation; not run on macOS/CPU')


def test_real_tiny_qwen_bf16_and_saved_nf4_generation(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast,BitsAndBytesConfig,AutoModelForCausalLM
    from policy_dissertation.config import Settings
    from policy_dissertation.pipelines.model_evaluation.backend import CudaGenerator
    from policy_dissertation.pipelines.model_evaluation.schemas import EvaluationConfig
    assert torch.cuda.is_available(),'Opt-in real CUDA integration requested but unavailable'
    raw=Tokenizer(WordLevel({'<unk>':0,'<pad>':1,'<eos>':2,'ready':3},unk_token='<unk>'))
    raw.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    tok.chat_template="{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}"
    cfg=Qwen3Config(vocab_size=32,hidden_size=64,intermediate_size=256,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=256,
        pad_token_id=1,eos_token_id=2,bos_token_id=2)
    bf16=tmp_path/'bf16';model=Qwen3ForCausalLM(cfg);model.save_pretrained(bf16);tok.save_pretrained(bf16)
    del model
    config=EvaluationConfig(max_new_tokens=16,executor='disabled',max_input_tokens=128)
    backend=CudaGenerator(Settings(_env_file=None));runtime=backend.preflight()
    assert runtime['device']
    try:
        backend.load(str(bf16),None,config)
        result=backend.generate([{'role':'user','content':'ready'}],config)
        assert result['output_tokens']>0 and result['generation_seconds']>0
        assert result['status'] in ('ok','output_limit')
    finally:backend.release()
    nf4=tmp_path/'nf4'
    quantized=AutoModelForCausalLM.from_pretrained(bf16,dtype=torch.bfloat16,device_map={'':0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16))
    quantized.save_pretrained(nf4);tok.save_pretrained(nf4);del quantized;torch.cuda.empty_cache()
    try:
        backend.load(str(nf4),None,config)
        assert backend.model.is_loaded_in_4bit
        result=backend.generate([{'role':'user','content':'ready'}],config)
        assert result['output_tokens']>0 and result['peak_allocated_bytes']>0
    finally:backend.release()


# Exercise the actual saved 8-bit export/reload using the same opt-in tiny model
# pattern without downloading a model or claiming coding quality.
def test_real_tiny_qwen_int8_export_reload(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
    from policy_dissertation.config import Settings
    from policy_dissertation.pipelines.quantization.int8 import Int8Backend,Int8Config
    from policy_dissertation.pipelines.training.config import CompressionConfig
    from policy_dissertation.pipelines.policy_transformation.data_io import dump_json
    assert torch.cuda.is_available()
    raw=Tokenizer(WordLevel({'<unk>':0,'<pad>':1,'<eos>':2,'ready':3},unk_token='<unk>'))
    raw.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    tok.chat_template="{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}"
    cfg=Qwen3Config(vocab_size=32,hidden_size=64,intermediate_size=256,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=256,
        pad_token_id=1,eos_token_id=2,bos_token_id=2)
    source=tmp_path/'bf16';model=Qwen3ForCausalLM(cfg);model.save_pretrained(source);tok.save_pretrained(source)
    del model
    inputs=tmp_path/'inputs';inputs.mkdir()
    dump_json(inputs/'calibration.json',[{'input_ids':[3,3,2],'labels':[-100,3,2]}])
    output=tmp_path/'int8';output.mkdir()
    backend=Int8Backend(Settings(_env_file=None))
    backend.preflight(tmp_path,CompressionConfig(minimum_free_vram_gib=1,minimum_free_disk_gib=1))
    try:
        result=backend.export(output,source,inputs,Int8Config())
        assert result['reload_verified'] and result['quantized_linear_modules']>0
        assert result['nominal_weight_bits']==8
    finally:backend.release()
