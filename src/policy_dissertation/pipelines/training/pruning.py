"""Qwen3 dense gated-MLP channel pruning; no sparsity masks or attention surgery."""
import math
from policy_dissertation.errors import DissertationError


def retained_width(width, ratio, multiple):
    keep=math.ceil(width*(1-ratio)/multiple)*multiple
    if not 0 < keep < width:
        raise DissertationError('Pruning ratio/alignment removes no channels or removes the whole MLP.')
    return keep


def select_channels(scores, keep):
    # Stable ties preserve original channel order. Kept channels are then stored in
    # original order so gate/up/down projections remain exactly aligned.
    if not 0 < keep < len(scores) or any(not math.isfinite(v) for v in scores):
        raise DissertationError('Invalid pruning scores or retained channel count.')
    return sorted(sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:keep])


def prune_qwen3(model, calibration, config, torch):
    c=model.config
    if c.model_type!='qwen3' or c.hidden_act!='silu' or getattr(c,'quantization_config',None):
        raise DissertationError('Only unquantized dense Qwen3 SiLU models support this pruning implementation.')
    layers=model.model.layers
    if len(layers)!=c.num_hidden_layers:
        raise DissertationError('Qwen3 layer/config mismatch.')
    width=c.intermediate_size;keep=retained_width(width,config.prune_ratio,config.channel_multiple)
    before=sum(p.numel() for p in model.parameters())
    totals=[];counts=[];handles=[]
    for layer in layers:
        mlp=layer.mlp
        if (mlp.gate_proj.weight.shape!=(width,c.hidden_size) or
            mlp.up_proj.weight.shape!=(width,c.hidden_size) or
            mlp.down_proj.weight.shape!=(c.hidden_size,width) or
            any(getattr(getattr(mlp,n),'bias',None) is not None for n in ('gate_proj','up_proj','down_proj'))):
            raise DissertationError('Unsupported Qwen3 MLP projection shapes/biases.')
        totals.append(torch.zeros(width,device=mlp.down_proj.weight.device,dtype=torch.float32));counts.append(0)
    def capture(i):
        def hook(module,args):
            x=args[0].detach().float()
            totals[i].add_(x.square().sum(dim=tuple(range(x.ndim-1))))
            counts[i]+=x.numel()//x.shape[-1]
        return hook
    try:
        for i,layer in enumerate(layers):handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture(i)))
        model.eval()
        with torch.no_grad():
            for row in calibration:
                # Calibration records are unpadded and come exclusively from train.
                ids=torch.tensor([row['input_ids']],device=model.device)
                model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        if not counts or not all(counts):raise DissertationError('No calibration activations were collected.')
    finally:
        for handle in handles:handle.remove()
    selections=[]
    for i,layer in enumerate(layers):
        mlp=layer.mlp
        scores=(totals[i]/counts[i]).sqrt()*mlp.down_proj.weight.detach().float().norm(dim=0)
        if not float(scores.max())>0:raise DissertationError('Calibration produced no positive MLP importance scores.')
        ids=select_channels(scores.cpu().tolist(),keep);selections.append(ids)
        index=torch.tensor(ids,device=mlp.down_proj.weight.device)
        for name,axis in (('gate_proj',0),('up_proj',0),('down_proj',1)):
            old=getattr(mlp,name)
            new=torch.nn.Linear(c.hidden_size if axis==0 else keep,keep if axis==0 else c.hidden_size,
                                bias=False,device=old.weight.device,dtype=old.weight.dtype)
            new.weight=torch.nn.Parameter(old.weight.detach().index_select(axis,index).contiguous())
            setattr(mlp,name,new)
        mlp.intermediate_size=keep
    c.intermediate_size=keep
    after=sum(p.numel() for p in model.parameters())
    expected=len(layers)*3*c.hidden_size*(width-keep)
    if before-after!=expected or after>=before:raise DissertationError('Structural parameter reduction verification failed.')
    return {'method':'activation-RMS times down-projection L2 norm; gated-MLP channel slicing',
            'original_parameters':before,'pruned_parameters':after,'removed_parameters':before-after,
            'parameter_reduction_fraction':(before-after)/before,'original_intermediate_size':width,
            'retained_intermediate_size':keep,'removed_channel_fraction':1-keep/width,
            'calibration_tokens_per_layer':counts,'kept_channels_per_layer':selections,
            'attention_changed':False,'semantic_quality_verified':False}
