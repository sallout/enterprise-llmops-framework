"""Lazy Linux/NVIDIA implementation. No torch import on ordinary data-only CLI paths."""
import gc
import json
import math
import platform
import shutil
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
from policy_dissertation.errors import DissertationError
from .pruning import prune_qwen3

RUNTIME_PACKAGES=('torch','transformers','peft','accelerate','bitsandbytes','datasets','huggingface-hub','safetensors')


def require_linux():
    if platform.system()!='Linux':
        raise DissertationError('Compression execution supports Linux with NVIDIA CUDA only; macOS/MPS and CPU execution are intentionally unsupported.')


def token_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            yield {k:row[k] for k in ('input_ids','attention_mask','labels')}


def last_complete_checkpoint(folder):
    """Ignore interrupted checkpoint writes; the caller reports when no complete checkpoint exists."""
    candidates=[]
    for path in Path(folder).glob('checkpoint-*'):
        try:
            step=int(path.name.removeprefix('checkpoint-'))
            state=json.loads((path/'trainer_state.json').read_text())
            required=('optimizer.pt','scheduler.pt','rng_state.pth','adapter_config.json','adapter_model.safetensors')
            if state.get('global_step')==step and all((path/name).is_file() for name in required):
                candidates.append((step,path))
        except (ValueError,OSError):continue
    return str(max(candidates,key=lambda item:item[0])[1]) if candidates else None


class CudaBackend:
    def __init__(self,settings,progress=print):
        self.settings=settings;self.progress=progress;self.torch=None

    def preflight(self,output,config):
        require_linux()
        installed={}
        for name in RUNTIME_PACKAGES:
            try:installed[name]=version(name)
            except PackageNotFoundError:
                raise DissertationError('Linux training dependencies missing. Install with uv sync --extra training --locked.') from None
        import torch
        self.torch=torch
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise DissertationError('A working NVIDIA CUDA PyTorch installation is required; CPU fallback is disabled.')
        if torch.cuda.device_count()!=1:
            raise DissertationError('Expose exactly one GPU using CUDA_VISIBLE_DEVICES=0; multi-GPU execution is not implemented.')
        if not torch.cuda.is_bf16_supported():raise DissertationError('This pipeline requires a BF16-capable NVIDIA GPU.')
        if int(__import__('os').environ.get('WORLD_SIZE','1'))!=1:
            raise DissertationError('This initial pipeline supports one process / one CUDA device; distributed training is not implemented.')
        torch.cuda.set_device(0)
        free,total=torch.cuda.mem_get_info(0)
        if free<config.minimum_free_vram_gib*2**30:
            raise DissertationError('Insufficient free CUDA memory for BF16 pruning/recovery. Use a larger GPU or release existing workloads.')
        disk=shutil.disk_usage(output).free/2**30
        owned=sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())/2**30
        if disk+owned<config.minimum_free_disk_gib:
            raise DissertationError('Insufficient disk space for original/pruned/merged/4-bit checkpoints.')
        torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32=False
        return {'platform':'Linux','device':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
                'free_vram_gib':free/2**30,'total_vram_gib':total/2**30,'free_disk_gib':disk,
                'versions':installed,'runtime_tested':True,'weights_loaded':False,
                'memory_note':'Preflight is a lower-bound capacity guard, not a guarantee against out-of-memory.'}

    def release(self):
        gc.collect()
        if self.torch is not None:self.torch.cuda.empty_cache()

    def load(self,path,**kwargs):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(path,dtype=self.torch.bfloat16,device_map={'':0},
                    trust_remote_code=False,attn_implementation='sdpa',**kwargs)

    def smoke(self,model,row):
        t=self.torch;model.eval()
        ids=t.tensor([row['input_ids']],device=model.device)
        labels=t.tensor([row['labels']],device=model.device)
        with t.no_grad():
            result=model(input_ids=ids,attention_mask=t.ones_like(ids),labels=labels,use_cache=False)
            loss=float(result.loss.detach().float().cpu())
        if not math.isfinite(loss):raise DissertationError('Model smoke loss is non-finite.')
        # A bounded continuation verifies the generation code path, not answer quality.
        with t.no_grad():generated=model.generate(ids[:,:min(ids.shape[1],64)],max_new_tokens=4,do_sample=False,
                                                    pad_token_id=model.config.eos_token_id,use_cache=True)
        if generated.shape[-1]<=min(ids.shape[1],64):raise DissertationError('Model smoke generation returned no tokens.')
        return {'finite_assistant_loss':loss,'generated_tokens':generated.shape[-1]-min(ids.shape[1],64),
                'scope':'technical load/forward/generation only; not semantic evaluation'}

    def prune(self,directory,input_dir,tokenizer,config):
        from transformers import AutoConfig
        token=self.settings.hf_token.get_secret_value() if self.settings.hf_token else False
        c=AutoConfig.from_pretrained(config.model_id,revision=config.model_revision,token=token,trust_remote_code=False)
        if c.model_type!='qwen3' or getattr(c,'quantization_config',None):
            raise DissertationError('Pinned model must be original unquantized dense Qwen3.')
        calibration=json.loads((Path(input_dir)/'calibration.json').read_text())
        model=self.load(config.model_id,revision=config.model_revision,token=token)
        try:
            evidence=prune_qwen3(model,calibration,config,self.torch)
            evidence['smoke_before_save']=self.smoke(model,calibration[0])
            model.save_pretrained(directory,safe_serialization=True,max_shard_size='4GB')
            tokenizer.tokenizer.save_pretrained(directory)
        finally:
            del model;self.release()
        reloaded=self.load(directory,local_files_only=True)
        try:
            if sum(p.numel() for p in reloaded.parameters())!=evidence['pruned_parameters']:
                raise DissertationError('Pruned checkpoint reloaded with a different parameter count.')
            evidence['reload_smoke']=self.smoke(reloaded,calibration[0])
        finally:del reloaded;self.release()
        return evidence

    def recover(self,directory,pruned,input_dir,tokenizer,config,resume=False):
        from datasets import Dataset
        from transformers import Trainer,TrainingArguments,TrainerCallback
        from peft import LoraConfig,get_peft_model
        t=self.torch
        model=self.load(pruned,local_files_only=True)
        original_count=sum(p.numel() for p in model.parameters())
        model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(r=config.lora_rank,lora_alpha=2*config.lora_rank,lora_dropout=.05,
            bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        dataset=Dataset.from_generator(token_rows,gen_kwargs={'path':str(Path(input_dir)/'tokens.jsonl')},
                                       cache_dir=str(Path(directory)/'_arrow'))
        def collate(rows):
            size=max(len(r['input_ids']) for r in rows)
            padding=tokenizer.tokenizer.pad_token_id
            if padding is None:padding=tokenizer.tokenizer.eos_token_id
            return {key:t.tensor([r[key]+[fill]*(size-len(r[key])) for r in rows],dtype=t.long)
                    for key,fill in (('input_ids',padding),('attention_mask',0),('labels',-100))}
        checkpoints=Path(directory)/'checkpoints';checkpoints.mkdir(parents=True,exist_ok=True)
        arguments=TrainingArguments(output_dir=str(checkpoints),max_steps=config.max_steps,
            per_device_train_batch_size=config.batch_size,gradient_accumulation_steps=config.gradient_accumulation,
            learning_rate=config.learning_rate,bf16=True,fp16=False,seed=config.seed,data_seed=config.seed,
            save_strategy='steps',save_steps=config.save_steps,save_total_limit=2,logging_steps=1,
            report_to=[],push_to_hub=False,eval_strategy='no',optim='adamw_torch',warmup_ratio=.03,
            gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
            dataloader_num_workers=0,remove_unused_columns=False,logging_nan_inf_filter=False)
        class FiniteLoss(TrainerCallback):
            def on_log(self,args,state,control,logs=None,**kwargs):
                for key in ('loss','grad_norm'):
                    if key in (logs or {}) and not math.isfinite(float(logs[key])):
                        raise DissertationError('Non-finite recovery loss/gradient detected; training stopped.')
        trainer=Trainer(model=model,args=arguments,train_dataset=dataset,data_collator=collate,callbacks=[FiniteLoss()])
        last=last_complete_checkpoint(checkpoints) if resume else None
        if resume and not last:
            self.progress('No complete optimizer checkpoint available; recovery starts from the committed pruned model.')
        try:
            result=trainer.train(resume_from_checkpoint=last)
            loss=float(result.training_loss)
            if not math.isfinite(loss) or trainer.state.global_step<config.max_steps:
                raise DissertationError('Recovery training did not reach the required finite-loss step count.')
            adapter=Path(directory)/'adapter';trainer.save_model(str(adapter))
            tokenizer.tokenizer.save_pretrained(adapter)
            # Release optimizer states BEFORE creating merged BF16 weights.
            if trainer.optimizer is not None:trainer.optimizer.state.clear()
            trainer.optimizer=None;trainer.lr_scheduler=None
            self.release()
            merged=model.merge_and_unload(safe_merge=True)
            if sum(p.numel() for p in merged.parameters())!=original_count:
                raise DissertationError('Merged recovery model parameter count changed unexpectedly.')
            merged.config.use_cache=True
            calibration=json.loads((Path(input_dir)/'calibration.json').read_text())
            smoke=self.smoke(merged,calibration[0])
            merged.save_pretrained(Path(directory)/'merged',safe_serialization=True,max_shard_size='4GB')
            tokenizer.tokenizer.save_pretrained(Path(directory)/'merged')
            return {'global_step':trainer.state.global_step,'training_loss':loss,'trainable_adapter_parameters':trainable,
                    'base_parameters':original_count,'resumed_checkpoint':Path(last).name if last else None,
                    'smoke':smoke,'method':'BF16 LoRA SFT after pruning; no teacher distillation',
                    'semantic_quality_verified':False}
        finally:
            del trainer,model
            if 'merged' in locals():del merged
            self.release()

    def quantize(self,directory,merged,input_dir,tokenizer,config):
        from transformers import BitsAndBytesConfig
        import bitsandbytes as bnb
        t=self.torch
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=t.bfloat16)
        model=self.load(merged,local_files_only=True,quantization_config=q)
        try:
            count=sum(isinstance(m,bnb.nn.Linear4bit) for m in model.modules())
            if not count or not getattr(model,'is_loaded_in_4bit',False):raise DissertationError('NF4 quantization was not applied.')
            model.save_pretrained(directory,safe_serialization=True,max_shard_size='4GB')
            tokenizer.tokenizer.save_pretrained(directory)
        finally:del model;self.release()
        saved=json.loads((Path(directory)/'config.json').read_text()).get('quantization_config',{})
        if not saved.get('load_in_4bit',saved.get('_load_in_4bit')) or saved.get('bnb_4bit_quant_type')!='nf4':
            raise DissertationError('Saved model is missing its NF4 quantization configuration.')
        model=self.load(directory,local_files_only=True)
        try:
            if not getattr(model,'is_loaded_in_4bit',False):raise DissertationError('Saved NF4 checkpoint did not reload in 4-bit.')
            row=json.loads((Path(input_dir)/'calibration.json').read_text())[0]
            smoke=self.smoke(model,row)
            return {'format':'Hugging Face Transformers + bitsandbytes NF4','nominal_weight_bits':4,
                    'quantized_linear_modules':count,'reload_verified':True,'smoke':smoke,
                    'mixed_precision_note':'Embeddings, norms and some output weights remain higher precision; not all tensors are 4-bit.',
                    'model_memory_bytes':model.get_memory_footprint(),'semantic_quality_verified':False}
        finally:del model;self.release()
