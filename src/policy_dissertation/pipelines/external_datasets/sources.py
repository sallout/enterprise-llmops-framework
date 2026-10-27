"""Inspect pinned Hub metadata and stream only inspected data files using Datasets."""
from fnmatch import fnmatch
import time
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from datasets import DownloadConfig, load_dataset
import aiohttp
from policy_dissertation.errors import DissertationError

DATASETS = {'nemotron': 'nvidia/Nemotron-Cascade-2-SFT-Data', 'opencode': 'nvidia/OpenCodeInstruct'}
DEFAULT_CONFIGS = {'nemotron': ['chat'], 'opencode': ['train']}
LICENSE_URLS = {'cc-by-4.0': 'https://creativecommons.org/licenses/by/4.0/',
    'nvidia-open-model-license': 'https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/'}


def error_category(exc):
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    if isinstance(status, int):
        return f'huggingface_http_{status}'
    if isinstance(exc, DissertationError):
        return str(exc)  # Only locally constructed, fixed error messages.
    return 'dataset_access_or_stream_error:' + type(exc).__name__


class HubProvider:
    def __init__(self, settings, token=None):
        self.settings = settings
        self.token = token
        self.api = HfApi(token=token)

    def inspect(self, alias, revision=None):
        dataset = DATASETS[alias]
        info = self.api.dataset_info(dataset, revision=revision, timeout=self.settings.request_timeout)
        if not info.sha:
            raise DissertationError('dataset_revision_unavailable')
        card = info.card_data.to_dict() if info.card_data else {}
        configs = card.get('configs', [])
        license_name = card.get('license_name') if card.get('license') == 'other' else card.get('license')
        if license_name not in LICENSE_URLS or not configs:
            raise DissertationError('unsupported_license_or_missing_configuration_metadata')
        path = hf_hub_download(dataset, 'README.md', repo_type='dataset', revision=info.sha,
                               token=self.token, etag_timeout=self.settings.request_timeout)
        with open(path, encoding='utf-8') as stream:
            card_text = stream.read(262145)
        if len(card_text.encode()) > 262144:
            raise DissertationError('dataset_card_too_large')
        return {'alias': alias, 'dataset': dataset, 'revision': info.sha,
                'license': license_name, 'license_url': LICENSE_URLS[license_name],
                'gated': info.gated, 'configs': configs, 'dataset_info': card.get('dataset_info', []),
                'files': sorted(s.rfilename for s in info.siblings), 'card_text': card_text}

    def stream(self, source):
        urls = [hf_hub_url(source['dataset'], f, repo_type='dataset', revision=source['revision'])
                for f in source['data_files']]
        fmt = source['format']
        # Generic packaged readers, not remote repository scripts. All URLs pin the inspected SHA.
        options = {'chunksize': 1048576, 'parse_agent_traces': False} if fmt == 'json' else {'batch_size': 16}
        config = DownloadConfig(token=self.token, max_retries=0,
            storage_options={'block_size': 1048576,
                             'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=self.settings.request_timeout)}})
        stream = load_dataset(fmt, data_files={source['split']: urls}, split=source['split'],
                              streaming=True, download_config=config, **options)
        return iter(stream)


def inspected_source(info, config_name, split):
    configs = {c['config_name']: c for c in info['configs']}
    if config_name not in configs:
        raise DissertationError('requested_configuration_unavailable')
    declared = configs[config_name]['data_files']
    patterns = []
    for entry in declared:
        if entry['split'] == split:
            patterns.extend(entry['path'] if isinstance(entry['path'], list) else [entry['path']])
    files = sorted(f for f in info['files'] if any(fnmatch(f, p) for p in patterns))
    if not files:
        raise DissertationError('requested_split_or_files_unavailable')
    formats = {'parquet' if f.endswith('.parquet') else 'json' if f.endswith(('.jsonl', '.jsonl.gz', '.json', '.json.gz')) else 'unsupported' for f in files}
    if len(formats) != 1 or 'unsupported' in formats:
        raise DissertationError('unsupported_or_mixed_data_file_formats')
    return {k: info[k] for k in ('alias', 'dataset', 'revision', 'license', 'license_url')} | {
        'config': config_name, 'split': split, 'data_files': files, 'format': formats.pop()}


def retry(action, settings, progress):
    for attempt in range(settings.max_attempts):
        try:
            return action()
        except Exception as exc:
            category = error_category(exc)
            progress(f'  Metadata attempt {attempt+1}/{settings.max_attempts}: {category}')
            if attempt + 1 == settings.max_attempts:
                raise DissertationError(category) from None
            time.sleep(min(.25 * 2**attempt, 1))
