"""Portable, explicit source configuration and immutable remote snapshots."""
import json
import re
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from policy_dissertation.errors import DissertationError
from policy_dissertation.data.hub import retry
from policy_dissertation.pipelines.policy_transformation.data_io import digest, dump_json, atomic_text


class Source(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(pattern=r'^[A-Za-z][A-Za-z0-9_-]*$')
    type: Literal['local', 'huggingface']
    path: str | None = None
    repo_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$')
    revision: str = Field(default='main', min_length=1)
    config: str | None = None
    split: str = 'train'
    adapter: Literal['auto', 'nemotron', 'opencode'] = 'auto'
    max_records: int | None = Field(default=None, ge=1)
    weight: float = Field(ge=0, allow_inf_nan=False)
    sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def valid_location(self):
        if self.type == 'local' and (not self.path or self.repo_id):
            raise ValueError('Local sources require path only.')
        if self.type == 'huggingface' and (not self.repo_id or self.path):
            raise ValueError('Hub sources require repo_id only.')
        return self


class Output(BaseModel):
    model_config = ConfigDict(extra='forbid')
    directory: str = './artifacts/pipeline4'
    max_tokens: int = Field(default=4096, ge=1)


class Split(BaseModel):
    model_config = ConfigDict(extra='forbid')
    train: float = Field(default=.9, gt=0, allow_inf_nan=False)
    validation: float = Field(default=.05, ge=0, allow_inf_nan=False)
    test: float = Field(default=.05, ge=0, allow_inf_nan=False)
    seed: int = 42


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sources: list[Source] = Field(min_length=1)
    output: Output = Field(default_factory=Output)
    split: Split = Field(default_factory=Split)

    @model_validator(mode='after')
    def names(self):
        if len({s.name for s in self.sources}) != len(self.sources) or not sum(s.weight for s in self.sources):
            raise ValueError('Source names must be unique and total weight positive.')
        return self


def read_config(path):
    path = Path(path).resolve()
    try:
        config = DatasetConfig.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
    except Exception:
        raise DissertationError('Invalid dataset YAML: check source names, paths, types, and finite weights.') from None
    # All relative paths are relative to this YAML, not the shell working directory.
    for source in config.sources:
        if source.type == 'local':
            source.path = str((path.parent/Path(source.path).expanduser()).resolve())
    config.output.directory = str((path.parent/Path(config.output.directory).expanduser()).resolve())
    return config


def materialize(source, output, settings):
    """Return a checksum-addressed JSONL entry; incomplete streams never become cache hits."""
    from huggingface_hub import HfApi, hf_hub_download
    from datasets import load_dataset
    from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    entry = {'pipeline': source.name, 'source_spec': source.model_dump(exclude_none=True), 'readable': True}
    if source.type == 'local':
        path = Path(source.path).resolve()
        if path.is_dir():
            path = path/'training.jsonl'
        if not path.is_file():
            raise DissertationError('Configured local dataset is missing: '+source.name)
        actual = file_hash(path)
        if source.sha256 and source.sha256 != actual:
            raise DissertationError('Configured source checksum mismatch: '+source.name)
        entry.update(training_path=str(path), training_hash=actual)
        adjacent = path.parent/'report.json'
        if adjacent.exists():
            report = json.loads(adjacent.read_text())
            if report.get('status') not in ('COMPLETED', 'COMPLETED_WITH_WARNINGS') or report.get('errors'):
                raise DissertationError('Configured upstream report is failed/incomplete: '+source.name)
            if report.get('output_hashes', {}).get(path.name) != actual:
                raise DissertationError('Configured upstream checksum mismatch: '+source.name)
            entry.update(report_path=str(adjacent), report_hash=file_hash(adjacent), upstream_accepted=report.get('accepted_examples'))
            if report.get('source_run_complete') is False:
                entry['producer_run'] = {'status': report.get('status'), 'complete': False,
                                         'accepted_subset_only': True}
        entry['evidence_scope'] = 'source checksum and per-record validation; standalone files do not prove upstream completion'
        return entry
    api = HfApi(token=token)
    cache = Path(output)/'_sources'/digest(source.model_dump())
    cache.mkdir(parents=True, exist_ok=True)
    pin = cache/'revision.json'
    if pin.exists():
        revision = json.loads(pin.read_text())['revision']
    else:
        info = retry(lambda: api.dataset_info(source.repo_id, revision=source.revision, timeout=30))
        revision = info.sha
        if not re.fullmatch(r'[0-9a-f]{40,64}', revision or ''):
            raise DissertationError('Source has no immutable revision: '+source.name)
        card = info.card_data.to_dict() if getattr(info, 'card_data', None) else {}
        dump_json(pin, {'revision': revision, 'license': card.get('license'), 'license_name': card.get('license_name')})
    if not re.fullmatch(r'[0-9a-f]{40,64}', revision or ''):
        raise DissertationError('Invalid cached immutable source revision.')
    snapshot = cache/revision
    snapshot.mkdir(exist_ok=True)
    data, manifest_path = snapshot/'training.jsonl', snapshot/'snapshot.json'
    if data.exists() and manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if file_hash(data) == cached['training_hash']:
            return {**entry, **cached, 'training_path': str(data)}
        raise DissertationError('Remote cache checksum mismatch: '+source.name)
    files = retry(lambda: api.list_repo_files(source.repo_id, revision=revision, repo_type='dataset'))
    producer_run = None
    if 'manifest.json' in files and 'training.jsonl' in files and source.config in (None, 'default') and (source.split == 'train' or source.split+'.jsonl' in files) and source.max_records is None:
        mp = retry(lambda: hf_hub_download(source.repo_id, 'manifest.json', repo_type='dataset', revision=revision, token=token, etag_timeout=30))
        remote_manifest = json.loads(Path(mp).read_text())
        producer_run = remote_manifest.get('source_run')
        selected_file = source.split+'.jsonl' if source.split+'.jsonl' in remote_manifest.get('output_hashes', {}) else 'training.jsonl'
        downloaded = retry(lambda: hf_hub_download(source.repo_id, selected_file, repo_type='dataset', revision=revision, token=token, etag_timeout=30))
        expected = remote_manifest.get('output_hashes', {}).get(selected_file)
        if file_hash(Path(downloaded)) != expected:
            raise DissertationError('Hub manifest checksum mismatch: '+source.name)
        import shutil
        shutil.copyfile(downloaded, data)
        count = remote_manifest.get('splits',{}).get(source.split) if selected_file!='training.jsonl' else remote_manifest.get('accepted_examples')
    else:
        # Generic Hub datasets must already contain canonical or legacy pipeline records.
        # Native NVIDIA schemas go through Pipeline 3's dataset-specific adapters first.
        def stream_snapshot():
            temporary = snapshot/'training.jsonl.partial'
            count = 0
            stream = None
            try:
                stream = load_dataset(source.repo_id, name=source.config, split=source.split,
                                      revision=revision, token=token, streaming=True,
                                      **({'parse_agent_traces': False} if not any(f.endswith('.parquet') for f in files) else {}))
                with temporary.open('w', encoding='utf-8') as target:
                    from itertools import islice
                    rows = islice(stream, source.max_records) if source.max_records is not None else stream
                    for row in rows:
                        text = json.dumps(row, ensure_ascii=False, allow_nan=False)
                        if len(text.encode('utf-8')) > settings.pipeline4.max_line_bytes:
                            raise DissertationError('Remote record exceeds configured byte guard.')
                        target.write(text+'\n'); count += 1
                temporary.replace(data)
                return count
            finally:
                if stream is not None and hasattr(stream, 'close'):stream.close()
                temporary.unlink(missing_ok=True)
        count = retry(stream_snapshot)
    actual = file_hash(data)
    if source.sha256 and actual != source.sha256:
        raise DissertationError('Remote source checksum mismatch: '+source.name)
    cached = {'training_hash': actual, 'upstream_accepted': count, 'resolved_revision': revision,
              'producer_run': producer_run,
              'repo_id': source.repo_id, 'config': source.config, 'split': source.split,
              'license_metadata': {k:v for k,v in json.loads(pin.read_text()).items() if k != 'revision'},
              'evidence_scope': 'pinned Hub content; inherited test evidence is not locally verified'}
    dump_json(manifest_path, cached)
    return {**entry, **cached, 'training_path': str(data)}
