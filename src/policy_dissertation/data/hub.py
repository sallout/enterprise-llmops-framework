"""Opt-in, private, allowlisted publication of validated conversational exports."""
import json
import re
import tempfile
import time
from pathlib import Path
from huggingface_hub import HfApi
from policy_dissertation.errors import DissertationError
from policy_dissertation.data.adapters import adapt
from policy_dissertation.data.schemas import VERSION
from policy_dissertation.pipelines.policy_transformation.data_io import dump_json, atomic_text


def retry(operation, attempts=3):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt + 1 == attempts:
                raise DissertationError('Hugging Face operation failed after bounded retries; check access and connectivity.') from None
            time.sleep(min(2 ** attempt, 4))


def protect_exports(output, overwrite=False):
    names = ('training.jsonl', 'train.jsonl', 'validation.jsonl', 'test.jsonl', 'report.json', 'manifest.json')
    if not overwrite and any((Path(output)/name).exists() for name in names):
        raise DissertationError('Output already contains exported artifacts. Use a new output directory or --overwrite to resume/replace exports explicitly.')


def _publish(output, repo_id, settings, *, api=None, allow_incomplete_run=False):
    from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
    from policy_dissertation.pipelines.dataset_preparation.quality import scan_sensitive
    output = Path(output)
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    if not token:
        raise DissertationError('Publishing requires HF_TOKEN; local exports are preserved.')
    if not re.fullmatch(r'[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*', repo_id):
        raise DissertationError('Dataset repository must be OWNER/NAME.')
    report = json.loads((output/'report.json').read_text())
    data = output/'training.jsonl'
    incomplete = report.get('status') not in ('COMPLETED', 'COMPLETED_WITH_WARNINGS') or bool(report.get('errors')) or report.get('source_run_complete') is False
    if incomplete and not allow_incomplete_run:
        raise DissertationError('Only completed validated exports can be published; local artifacts are preserved.')
    if file_hash(data) != report.get('output_hashes', {}).get('training.jsonl'):
        raise DissertationError('Export checksum mismatch; publication refused.')
    with tempfile.TemporaryDirectory(prefix='llmops-publish-') as temporary:
        stage = Path(temporary)
        count = 0
        schema_versions = set()
        accepted_ids = set()
        with data.open(encoding='utf-8') as source, (stage/'training.jsonl').open('w', encoding='utf-8') as target:
            for line in source:
                raw = json.loads(line)
                checked = adapt(raw, raw.get('metadata', {}).get('pipeline'), {'stage': 'publication_validation'})
                schema_versions.add(raw.get('schema_version'))
                accepted_ids.add(checked['id'])
                secrets = [s.get_secret_value() for s in (settings.hf_token, settings.model.api_key) if s]
                scan_sensitive(line, settings.pipeline4, secrets)
                target.write(line)
                count += 1
        if file_hash(stage/'training.jsonl') != report['output_hashes']['training.jsonl']:
            raise DissertationError('Export changed during publication validation.')
        if count != report.get('accepted_examples') or count == 0:
            raise DissertationError('Publication count mismatch or empty export.')
        local_manifest = json.loads((output/'manifest.json').read_text()) if (output/'manifest.json').exists() else {}
        manifest = {'schema_version': next(iter(schema_versions)) if len(schema_versions) == 1 else 'mixed-supported-legacy', 'accepted_examples': count,
                    'output_hashes': {'training.jsonl': file_hash(stage/'training.jsonl')},
                    'source_run': {'status': report.get('status'), 'complete': not incomplete,
                                   'accepted_subset_only': bool(incomplete),
                                   'failed_units': report.get('failed_units'), 'pending_units': report.get('pending_units')},
                    'license': local_manifest.get('license', 'unspecified; no redistribution license inferred'),
                    'verification': 'Inherited evidence; no tests reexecuted during publication.'}
        allow_patterns=['training.jsonl','manifest.json','README.md']
        prepared = bool(report.get('training_ready'))
        if prepared:
            if len(accepted_ids)!=count:
                raise DissertationError('Prepared accepted export has duplicate IDs.')
            split_ids=set()
            manifest['splits']=report.get('splits',{})
            manifest['training_ready']=True
            manifest['tokenizer']=report.get('preparation',{}).get('tokenizer')
            for split in ('train','validation','test'):
                path=output/(split+'.jsonl')
                if file_hash(path)!=report['output_hashes'].get(path.name):
                    raise DissertationError('Prepared split checksum mismatch.')
                n=0
                with path.open() as stream, (stage/path.name).open('w') as target:
                    for line in stream:
                        raw=json.loads(line)
                        validated=adapt(raw,raw.get('metadata',{}).get('pipeline'),{'stage':'publication_split_validation'})
                        scan_sensitive(line,settings.pipeline4,secrets)
                        if validated['id'] not in accepted_ids or validated['id'] in split_ids or raw['metadata'].get('training_split')!=split:
                            raise DissertationError('Prepared split membership mismatch.')
                        split_ids.add(validated['id']);target.write(line);n+=1
                if n!=manifest['splits'].get(split):raise DissertationError('Prepared split count mismatch.')
                manifest['output_hashes'][path.name]=file_hash(stage/path.name)
                allow_patterns.append(path.name)
            if split_ids!=accepted_ids:raise DissertationError('Prepared splits do not cover accepted examples.')
        dump_json(stage/'manifest.json', manifest)
        atomic_text(stage/'README.md', '''---
pretty_name: Enterprise conversational training data
configs:
- config_name: default
  data_files:
  - split: train
    path: training.jsonl
---
# Enterprise conversational training data

Canonical schema: unified-qwen-sft-v1. Messages and optional tools are model-facing;
metadata and verification preserve provenance. This private export grants no new
license. Original source licenses and restrictions remain applicable. The publisher
must confirm rights for source-derived policy/code text. No semantic certification
or new runtime verification is implied. Only accepted records are included.
''')
        if prepared:
            text=(stage/'README.md').read_text().replace('  - split: train\n    path: training.jsonl', '\n'.join(f'  - split: {split}\n    path: {split}.jsonl' for split in ('train','validation','test')))
            atomic_text(stage/'README.md',text+'\ntraining.jsonl is an all-split audit union; use the declared split files for training/evaluation.\n')
        atomic_text(stage/'README.md', (stage/'README.md').read_text() +
                    f"\nExport schema: {manifest['schema_version']}. Source run status: {report.get('status')}. "
                    f"Accepted subset only: {bool(incomplete)}. Publication does not mark the producer run complete.\n")
        api = api or HfApi(token=token)
        # Validate all records before making any remote mutation.
        retry(lambda: api.create_repo(repo_id, repo_type='dataset', private=True, exist_ok=True))
        info = retry(lambda: api.dataset_info(repo_id))
        if info.private is not True:
            raise DissertationError('Publication target must be private; public repositories are refused.')
        commit = retry(lambda: api.upload_folder(repo_id=repo_id, repo_type='dataset', folder_path=stage,
                      allow_patterns=allow_patterns,
                      commit_message='Publish validated unified training export'))
        revision = commit.oid
        if not re.fullmatch(r'[0-9a-f]{40,64}', revision or ''):
            raise DissertationError('Hub did not return an immutable commit revision; local exports are preserved.')
    local = output/'manifest.json'
    manifest = json.loads(local.read_text()) if local.exists() else manifest
    manifest['publication'] = {'repo_id': repo_id, 'revision': revision, 'private': True,
                               'accepted_subset_only': bool(incomplete), 'source_run_status': report.get('status'),
                               'training_hash': report['output_hashes']['training.jsonl']}
    dump_json(local, manifest)
    return manifest['publication']


def publish(output, repo_id, settings, *, api=None, allow_incomplete_run=False):
    """Keep successful local exports intact even when optional publication fails."""
    try:
        return _publish(output, repo_id, settings, api=api, allow_incomplete_run=allow_incomplete_run)
    except Exception:
        path = Path(output)/'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest['last_publication_attempt'] = {'status': 'FAILED', 'reason': 'publication_failed; local exports retained'}
            dump_json(path, manifest)
        raise
