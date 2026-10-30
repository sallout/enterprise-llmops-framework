"""Lazy Transformers generation for original, pruned, merged and saved NF4 Qwen3."""
import gc
import platform
from importlib.metadata import version
from time import perf_counter
from pathlib import Path
from policy_dissertation.errors import DissertationError


def json_metadata(value):
    """Convert Transformers quantization objects/dtypes without importing torch."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metadata(item) for item in value]
    if callable(getattr(value, 'to_dict', None)):
        return json_metadata(value.to_dict())
    if type(value).__module__ == 'torch' and type(value).__name__ in ('dtype', 'device'):
        return str(value)
    raise TypeError('Unsupported model metadata type: ' + type(value).__name__)


class CudaGenerator:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.torch = None

    def preflight(self):
        if platform.system() != 'Linux':
            raise DissertationError('Model evaluation inference requires Linux/NVIDIA CUDA; macOS is for suite preparation and mocked tests only.')
        import torch
        self.torch = torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise DissertationError('Expose one BF16-capable NVIDIA CUDA GPU for sequential evaluation.')
        return {'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                'versions': {n: version(n) for n in ('torch', 'transformers', 'accelerate', 'bitsandbytes')}}

    def load(self, source, revision, config):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        token = self.settings.hf_token.get_secret_value() if self.settings.hf_token else False
        kwargs = {'local_files_only': True} if Path(source).is_dir() else {'revision': revision, 'token': token}
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(source, dtype=self.torch.bfloat16,
            device_map={'': 0}, attn_implementation='sdpa', trust_remote_code=False, **kwargs)
        if self.model.config.model_type != 'qwen3':
            raise DissertationError('This evaluator currently supports dense Qwen3 checkpoints only.')
        self.model.eval()
        self.torch.manual_seed(config.seed); self.torch.cuda.manual_seed_all(config.seed)
        self.torch.backends.cuda.matmul.allow_tf32 = False
        return {'model_memory_bytes': self.model.get_memory_footprint(),
                'quantization': json_metadata(getattr(self.model.config, 'quantization_config', None)),
                'chat_template_hash': __import__('hashlib').sha256((self.tokenizer.chat_template or '').encode()).hexdigest()}

    def generate(self, messages, config):
        t = self.torch
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=False)
        encoded = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False, truncation=False)
        length = encoded['input_ids'].shape[-1]
        if length > config.max_input_tokens:
            return {'status': 'input_too_long', 'text': '', 'input_tokens': length, 'output_tokens': 0}
        context = getattr(self.model.config, 'max_position_embeddings', 0)
        if context and length + config.max_new_tokens > context:
            return {'status': 'context_limit', 'text': '', 'input_tokens': length, 'output_tokens': 0}
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        t.cuda.synchronize(); t.cuda.reset_peak_memory_stats(); start = perf_counter()
        with t.inference_mode():
            ids = self.model.generate(**encoded, do_sample=False, max_new_tokens=config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                use_cache=True)
        t.cuda.synchronize(); seconds = perf_counter() - start
        new = ids[0, length:]; eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        ended = bool(len(new) and int(new[-1]) in eos)
        return {'status': 'ok' if ended else 'output_limit',
                'text': self.tokenizer.decode(new, skip_special_tokens=True), 'input_tokens': length,
                'output_tokens': len(new), 'generation_seconds': seconds,
                'output_tokens_per_second_including_prefill': len(new) / seconds,
                'peak_allocated_bytes': t.cuda.max_memory_allocated(),
                'peak_reserved_bytes': t.cuda.max_memory_reserved()}

    def release(self):
        self.model = None
        if hasattr(self, 'tokenizer'): del self.tokenizer
        gc.collect()
        if self.torch is not None: self.torch.cuda.empty_cache()
