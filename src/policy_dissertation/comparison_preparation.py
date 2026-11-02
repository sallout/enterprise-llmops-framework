"""Prepare a separate quantized-reference view with a documented common input protocol."""
import hashlib
import errno
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
from policy_dissertation.pipelines.model_evaluation.suites import read_suite
from policy_dissertation.pipelines.policy_transformation.data_io import dump_json,digest


def prepare_quantized_reference(model,reference,suite,output,receipt,*,repo_id,revision,loader=None,weight_storage="auto"):
    if weight_storage not in ("auto","copy","symlink"):
        raise DissertationError("Weight storage must be auto, copy or symlink.")
    model,reference,suite,output,receipt=(Path(p).resolve() for p in (model,reference,suite,output,receipt))
    if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',revision):
        raise DissertationError('Provide an immutable external model revision.')
    protected=(model,reference,suite)
    if any(p==q or p.is_relative_to(q) or q.is_relative_to(p) for p in (output,receipt) for q in protected):
        raise DissertationError('Comparison output/receipt must be separate from all original inputs.')
    if output==receipt or output.is_relative_to(receipt) or receipt.is_relative_to(output):
        raise DissertationError('Comparison model and receipt must be separate directories.')
    if output.exists() or receipt.exists():
        raise DissertationError('Comparison model and receipt paths must be new.')
    if not model.is_dir() or not reference.is_dir():
        raise DissertationError('Provide local downloaded external and reference model directories.')
    receipt.mkdir(parents=True);output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'RUNNING','model_weights_modified':False,'model_inference':False,'training_started':False,
            'source_repo':repo_id,'source_revision':revision,'source_revision_verification':'Caller-provided download pin; local file hashes recorded.',
            'source_directory':str(model),'reference_directory':str(reference),'output_directory':str(output),
            'requested_weight_storage':weight_storage,'errors':[]}
    try:
        cfg=json.loads((model/'config.json').read_text());ref_cfg=json.loads((reference/'config.json').read_text())
        quant=cfg.get('quantization_config',{})
        if cfg.get('model_type')!='qwen3' or ref_cfg.get('model_type')!='qwen3' or cfg.get('auto_map'):
            raise DissertationError('Only standard local Qwen3 models are supported.')
        if quant.get('quant_method')!='bitsandbytes' or quant.get('bnb_4bit_quant_type')!='nf4' or not quant.get('load_in_4bit',quant.get('_load_in_4bit')):
            raise DissertationError('External reference must be a saved bitsandbytes NF4 checkpoint.')
        payload,cases=read_suite(suite)
        if loader is None:
            from transformers import AutoTokenizer
            loader=lambda path:AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        base=loader(reference);external=loader(model)
        if not isinstance(base.chat_template,str) or not base.chat_template:
            raise DissertationError('Reference tokenizer lacks a single chat template.')
        if external.get_vocab()!=base.get_vocab():
            raise DissertationError('Vocabulary/token IDs differ; automatic input standardization refused.')
        if external.eos_token_id!=base.eos_token_id or cfg.get('eos_token_id')!=ref_cfg.get('eos_token_id'):
            raise DissertationError('EOS configuration differs; automatic standardization refused.')
        if base.pad_token is None:
            raise DissertationError('Reference pad token is not defined.')
        generation={}
        for label,folder,model_cfg in (('external',model,cfg),('reference',reference,ref_cfg)):
            path=folder/'generation_config.json'
            generation[label]=json.loads(path.read_text()) if path.is_file() else model_cfg
        for key in ('eos_token_id','bos_token_id','forced_eos_token_id','forced_bos_token_id'):
            if generation['external'].get(key)!=generation['reference'].get(key):
                raise DissertationError('Generation stopping/BOS configuration differs; automatic standardization refused.')
        original_template=external.chat_template
        original_pad=external.pad_token
        external.chat_template=base.chat_template;external.pad_token=base.pad_token
        def encode(tokenizer,messages):
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            return text,tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
        prompts={}
        for case in cases:
            a,ids=encode(base,case.messages);b,other=encode(external,case.messages)
            if a!=b or ids!=other:
                raise DissertationError('Rendered prompt/token IDs differ after standardization.')
            prompts[case.id]=digest(ids)
        weights=sorted(model.glob('*.safetensors'))
        if not weights or any(p.is_symlink() for p in weights):
            raise DissertationError('Expected regular saved safetensors shards.')
        hashes={p.name:file_hash(p) for p in weights}
        assets={p.name:file_hash(p) for p in model.iterdir() if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jinja','.txt')}
        originals=receipt/'original_external_tokenizer';originals.mkdir()
        for p in model.iterdir():
            if p.is_file() and not p.is_symlink() and (p.name.startswith(('tokenizer','special_tokens','added_tokens','chat_template','vocab','merges'))):
                shutil.copyfile(p,originals/p.name)
        modes={}
        with tempfile.TemporaryDirectory(prefix='.quantized-reference-',dir=output.parent) as temp:
            stage=Path(temp)/'model';stage.mkdir()
            for p in model.iterdir():
                if not p.is_file() or p.is_symlink() or p.name.startswith('.'):
                    continue
                if p.suffix=='.safetensors':
                    if weight_storage=='symlink':
                        os.symlink(p,stage/p.name);modes[p.name]='symlink'
                    elif weight_storage=='copy':
                        shutil.copyfile(p,stage/p.name);modes[p.name]='copy'
                    else:
                        try:os.link(p,stage/p.name);modes[p.name]='hardlink'
                        except OSError as exc:
                            if exc.errno in (errno.EDQUOT,errno.ENOSPC):
                                raise DissertationError('Storage quota/full filesystem: use --weight-storage symlink with new output and receipt paths under /tmp; do not delete trained weights.') from exc
                            shutil.copyfile(p,stage/p.name);modes[p.name]='copy'
                elif p.suffix in ('.json','.txt','.jinja') or p.name in ('README.md','LICENSE'):
                    shutil.copyfile(p,stage/p.name)
            external.save_pretrained(stage)
            loaded=loader(stage)
            for case in cases:
                _,ids=encode(loaded,case.messages)
                if digest(ids)!=prompts[case.id]:raise DissertationError('Saved tokenizer does not preserve the verified prompt protocol.')
            if {p.name:file_hash(stage/p.name) for p in weights}!=hashes or {p.name:file_hash(p) for p in weights}!=hashes:
                raise DissertationError('External tensor bytes changed; comparison view refused.')
            if any(file_hash(model/name)!=h for name,h in assets.items()):
                raise DissertationError('Original external tokenizer/config files changed.')
            stage.rename(output)
        template_hash=lambda s:hashlib.sha256((s or '').encode()).hexdigest()
        report.update(status='COMPLETED',suite_digest=payload['digest'],verified_task_prompts=len(cases),
            vocabulary_hash=digest(base.get_vocab()),prompt_token_hashes=prompts,
            original_template_sha256=template_hash(original_template),common_template_sha256=template_hash(base.chat_template),
            original_pad_token=original_pad,common_pad_token=base.pad_token,quantization=quant,
            generation_stop_configuration={k:generation['external'].get(k) for k in ('eos_token_id','bos_token_id','forced_eos_token_id','forced_bos_token_id')},
            weight_hashes=hashes,weight_storage=modes,original_asset_hashes=assets,
            protocol='Original Qwen chat template and padding, identical vocabulary/rendered prompts/token IDs; external tensor weights unchanged.',
            limitations=['External model is evaluated under a standardized input protocol, not its unmodified default tokenizer settings.',
                'Upstream source weights may reflect a different base revision/quantization implementation; this is an external comparator, not a fully controlled quantization-only ablation.',
                'Symlink views require their original downloaded source to remain available; they are temporary inference views, not independent model backups.'])
    except (Exception,KeyboardInterrupt) as exc:
        report['status']='FAILED'
        if isinstance(exc,OSError) and exc.errno is not None:
            report['filesystem_errno']=exc.errno
            report['filesystem_reason']=os.strerror(exc.errno)
        report['errors'].append(str(exc) if isinstance(exc,DissertationError) else 'reference_preparation_failed:'+type(exc).__name__)
    dump_json(receipt/'report.json',report)
    return report
