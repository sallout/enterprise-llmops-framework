"""Pinned tokenizer assets only; no weights, GPU runtime, or remote Python code."""
import json
from pathlib import Path
import shutil
import time
from huggingface_hub import HfApi, hf_hub_download
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.data_io import digest, dump_json
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash

ASSETS = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json',
          'merges.txt', 'chat_template.jinja', 'config.json')


class QwenTokenizer:
    def __init__(self, output, settings, token=None):
        cache = output/'_tokenizer'
        manifest_path = cache/(digest([settings.tokenizer_model, settings.tokenizer_revision])+'.json')
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest is None:
            if settings.local_files_only:
                raise DissertationError('Pinned tokenizer assets unavailable offline.')
            info = HfApi(token=token).model_info(settings.tokenizer_model, revision=settings.tokenizer_revision, files_metadata=True)
            names = {f.rfilename: f for f in info.siblings}
            wanted = [name for name in ASSETS if name in names]
            if not info.sha or 'tokenizer.json' not in wanted or 'tokenizer_config.json' not in wanted:
                raise DissertationError('Required Qwen tokenizer assets missing.')
            folder = cache/info.sha
            folder.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for name in wanted:
                size = getattr(names[name], 'size', None)
                if size and size > 25000000:
                    raise DissertationError('Tokenizer asset exceeds download guard.')
                path = hf_hub_download(settings.tokenizer_model, name, revision=info.sha, token=token)
                shutil.copyfile(path, folder/name)
                hashes[name] = file_hash(folder/name)
            manifest = {'model': settings.tokenizer_model, 'revision': info.sha, 'files': hashes}
            dump_json(manifest_path, manifest)
        folder = cache/manifest['revision']
        if any(name not in ASSETS or not (folder/name).is_file() or file_hash(folder/name) != h for name, h in manifest['files'].items()):
            raise DissertationError('Tokenizer asset checksum mismatch.')
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise DissertationError('Qwen tokenizer has no chat template.')
        config = json.loads((folder/'config.json').read_text()) if (folder/'config.json').exists() else {}
        limits = [n for n in (config.get('max_position_embeddings'), self.tokenizer.model_max_length) if isinstance(n,int) and 0<n<10000000]
        self.info = manifest | {'validated': True, 'max_context_tokens':min(limits) if limits else None, 'chat_template_hash': digest(self.tokenizer.chat_template),
                                'add_generation_prompt': False, 'enable_thinking': False,
                                'tokenization': 'apply_chat_template(tokenize=True); no truncation'}
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        self.measure([{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hello.'}])

    def analyze(self, messages, tools=None):
        options = {'tokenize': True, 'add_generation_prompt': False, 'enable_thinking': False, 'truncation': False}
        if tools:
            options['tools'] = tools
        ids = self.tokenizer.apply_chat_template(messages, **options)
        # New Transformers versions may return BatchEncoding even without return_dict.
        if hasattr(ids, 'keys'):
            ids = ids['input_ids']
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids), digest(ids)

    def measure(self, messages, tools=None):
        return self.analyze(messages, tools)[0]


def load_tokenizer(output, settings, token=None, progress=lambda text: None):
    for attempt in range(settings.max_attempts):
        try:
            return QwenTokenizer(output, settings, token)
        except Exception as exc:
            progress(f'  Tokenizer attempt {attempt+1}/{settings.max_attempts}: {type(exc).__name__}')
            if attempt+1 == settings.max_attempts:
                raise DissertationError('Actual Qwen3-14B tokenizer validation unavailable; preparation is not training-ready.') from None
            time.sleep(min(.25*2**attempt, 1))
