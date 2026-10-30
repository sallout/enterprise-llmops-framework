"""Restore allowlisted published enterprise exports without generating data."""
import json
import hashlib
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from policy_dissertation.data.adapters import adapt
from policy_dissertation.data.hub import protect_exports, retry
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
from policy_dissertation.pipelines.policy_transformation.data_io import dump_json
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock


def download_dataset(repo_id, output, settings, *, revision='main', overwrite=False,
                     api=None, downloader=None):
    """Validate pinned cached files before committing exports; never infer new test evidence."""
    if not re.fullmatch(r'[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*', repo_id):
        raise DissertationError('Dataset repository must be OWNER/NAME.')
    token = settings.hf_token.get_secret_value() if settings.hf_token else False
    api = api or HfApi(token=token)
    downloader = downloader or snapshot_download
    output = Path(output)
    with output_lock(output):
        protect_exports(output, overwrite)
        info = retry(lambda: api.dataset_info(repo_id, revision=revision))
        pinned = info.sha
        if not re.fullmatch(r'[0-9a-f]{40,64}', pinned or ''):
            raise DissertationError('Hub did not provide an immutable dataset revision.')
        # Hub cache retains completed transfers for bounded retry/restart. No checkpoints,
        # raw sources, model weights, credentials or arbitrary remote paths are downloaded.
        files = ['manifest.json', 'README.md', 'training.jsonl', 'train.jsonl', 'validation.jsonl', 'test.jsonl']
        folder = Path(retry(lambda: downloader(repo_id=repo_id, repo_type='dataset', revision=pinned,
                                              token=token, allow_patterns=files)))
        try:
            manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            hashes = manifest['output_hashes']
            ready = manifest.get('training_ready') is True
            names = ['training.jsonl'] + (['train.jsonl','validation.jsonl','test.jsonl'] if ready else [])
            counts = {}; accepted = {}; assigned = set()
            for name in names:
                path = folder/name
                if not path.is_file() or file_hash(path) != hashes.get(name):
                    raise DissertationError('Downloaded export checksum mismatch or required file missing.')
                ids = set(); count = 0
                with path.open(encoding='utf-8') as stream:
                    for line in stream:
                        if not line.strip(): continue
                        raw = json.loads(line)
                        checked = adapt(raw, raw.get('metadata',{}).get('pipeline'), {'stage':'download_validation'})
                        rid = checked['id']
                        if rid in ids: raise DissertationError('Downloaded export contains duplicate record IDs.')
                        ids.add(rid); count += 1
                        digest = hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                        if name == 'training.jsonl': accepted[rid] = digest
                        else:
                            if (rid not in accepted or rid in assigned or digest != accepted[rid] or
                                raw.get('metadata',{}).get('training_split') != name.removesuffix('.jsonl')):
                                raise DissertationError('Downloaded split membership/content mismatch.')
                            assigned.add(rid)
                counts[name] = count
            if not counts['training.jsonl'] or counts['training.jsonl'] != manifest['accepted_examples']:
                raise DissertationError('Downloaded accepted count mismatch or empty export.')
            if ready:
                if not isinstance(manifest.get('tokenizer'), dict) or not manifest['tokenizer']:
                    raise DissertationError('Prepared dataset is missing tokenizer provenance.')
                if assigned != set(accepted) or any(counts[s+'.jsonl'] != manifest['splits'].get(s) for s in ('train','validation','test')):
                    raise DissertationError('Downloaded split coverage/count mismatch.')
            complete = manifest.get('source_run',{}).get('complete',manifest.get('source_run_complete',False)) is True
            report = {'status':'COMPLETED' if complete and ready else 'COMPLETED_WITH_WARNINGS',
                      'accepted_examples':counts['training.jsonl'], 'training_ready':ready,
                      'source_run_complete':complete, 'splits':manifest.get('splits',{}),
                      'preparation':{'tokenizer':manifest.get('tokenizer')},
                      'output_hashes':{n:hashes[n] for n in names}, 'errors':[],
                      'warnings':([] if complete else ['inherited_incomplete_or_unknown_source_run']) +
                                 ([] if ready else ['accepted_export_requires_enterprise_preparation']),
                      'restoration':{'repo_id':repo_id,'revision':pinned,'records_schema_checked':counts['training.jsonl'],
                                     'verification':'Checksums, schema, IDs and split membership checked; original evidence inherited. No new semantic/runtime verification.'},
                      'llm_calls':0,'training_started':False}
        except DissertationError: raise
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise DissertationError('Downloaded dataset is not a compatible validated enterprise export.') from None
        if not ready and any((output/n).exists() for n in ('train.jsonl','validation.jsonl','test.jsonl')):
            raise DissertationError('Choose a new output directory for an unprepared export; existing prepared splits are protected.')
        if any((output/n).is_symlink() for n in names + ['manifest.json','README.md','report.json']):
            raise DissertationError('Refusing to replace symlinked export files.')
        # Commit only validated, allowlisted files; report is the completion marker.
        for name in names + ['manifest.json']:
            shutil.copyfile(folder/name, output/name)
        if (folder/'README.md').is_file(): shutil.copyfile(folder/'README.md',output/'README.md')
        dump_json(output/'report.json', report)
        return report
