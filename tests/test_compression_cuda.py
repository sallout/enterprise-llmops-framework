"""Opt-in REAL Linux/CUDA tiny-Qwen test, no remote model/data downloads.

RUN_COMPRESSION_CUDA_TESTS=1 uv run --extra training pytest tests/test_compression_cuda.py -q
This tests technical execution only, not Qwen3-14B capacity or quality.
"""
import json
import os
import platform
from types import SimpleNamespace
import pytest

pytestmark=pytest.mark.skipif(platform.system()!='Linux' or os.getenv('RUN_COMPRESSION_CUDA_TESTS')!='1',
                            reason='Opt-in Linux NVIDIA CUDA compression integration; not run on macOS/CPU')


def test_real_tiny_qwen_prune_recovery_nf4_save_reload(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
    from policy_dissertation.config import Settings
    from policy_dissertation.pipelines.training.config import CompressionConfig
    from policy_dissertation.pipelines.training.backend import CudaBackend
    from policy_dissertation.pipelines.training.pruning import prune_qwen3
    if not torch.cuda.is_available():pytest.fail('Opt-in CUDA test requested, but CUDA is unavailable')
    config=CompressionConfig(prune_ratio=.125,channel_multiple=32,calibration_samples=1,max_tokens=128,
                             max_steps=2,gradient_accumulation=1,save_steps=1,minimum_free_vram_gib=1,minimum_free_disk_gib=1)
    backend=CudaBackend(Settings(_env_file=None))
    backend.preflight(tmp_path,config)
    vocabulary={'<unk>':0,'<pad>':1,'<eos>':2,**{chr(97+i):i+3 for i in range(26)}}
    raw=Tokenizer(WordLevel(vocabulary,unk_token='<unk>'));raw.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=raw,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    tok.chat_template="{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}"
    tokenizer=SimpleNamespace(tokenizer=tok)
    cfg=Qwen3Config(vocab_size=32,hidden_size=64,intermediate_size=256,num_hidden_layers=2,
                    num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=128,
                    pad_token_id=1,eos_token_id=2,bos_token_id=2)
    model=Qwen3ForCausalLM(cfg).to(device='cuda',dtype=torch.bfloat16)
    row={'input_ids':[3,4,5,6,7,2],'attention_mask':[1]*6,'labels':[-100,-100,5,6,7,2]}
    attention_before=model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    evidence=prune_qwen3(model,[row],config,torch)
    assert torch.equal(attention_before,model.model.layers[0].self_attn.q_proj.weight)
    assert evidence['removed_parameters']==2*3*64*32
    assert model.model.layers[0].mlp.down_proj.weight.shape==(64,224)
    pruned=tmp_path/'pruned';model.save_pretrained(pruned);tok.save_pretrained(pruned)
    del model;backend.release()
    inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'tokens.jsonl').write_text(json.dumps(row)+'\n')
    (inputs/'calibration.json').write_text(json.dumps([row]))
    recovery=tmp_path/'recovery';recovery.mkdir()
    trained=backend.recover(recovery,pruned,inputs,tokenizer,config)
    assert trained['global_step']==2 and trained['trainable_adapter_parameters']>0
    exported=tmp_path/'model_4bit';exported.mkdir()
    result=backend.quantize(exported,recovery/'merged',inputs,tokenizer,config)
    assert result['reload_verified'] and result['quantized_linear_modules']>0
    assert (exported/'model.safetensors').is_file()
