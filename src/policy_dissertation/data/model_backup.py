"""Explicit private model-experiment backup; verify remote bytes before approval."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.data_io import dump_json
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
from policy_dissertation.pipelines.training.pipeline import verify_stage

VERSION = 'compression-backup-v1'
EXCLUDED_DIRS = {'.cache', '_arrow', '__pycache__', '.git'}
TEXT_SUFFIXES = {'.json', '.jsonl', '.txt', '.yaml', '.yml', '.md', '.jinja'}


def checksums(path, secrets=()):
    size = path.stat().st_size
    sha256 = hashlib.sha256(); git = hashlib.sha1()
    git.update(f'blob {size}\0'.encode())
    tail = b''
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            sha256.update(block); git.update(block)
            if path.suffix.lower() in TEXT_SUFFIXES:
                sample = tail + block
                if any(secret.encode() in sample for secret in secrets if secret):
                    raise DissertationError('A configured credential occurs in a text artifact; backup refused. Source files were preserved.')
                tail = sample[-max([len(s.encode()) for s in secrets if s] + [1]):]
    return {'bytes': size, 'sha256': sha256.hexdigest(), 'git_blob_sha1': git.hexdigest()}


def selected_files(root, committed=()):
    result = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        required = relative.as_posix() in committed
        if any(part in EXCLUDED_DIRS for part in relative.parts) and not required: continue
        if path.name.endswith('.lock') and not required: continue
        if path.name.startswith('.env') or path.suffix.lower() in ('.pem', '.key'): continue
        if relative.as_posix() == 'backup_manifest.json': continue
        if path.is_symlink():
            raise DissertationError('Symlinked experiment artifacts are not supported; backup refused.')
        if path.is_file():
            name = relative.as_posix()
            if any(c in name for c in '*?['):
                raise DissertationError('Experiment filenames containing glob characters are not supported.')
            result[name] = path
    return result


def backup_model(input_directory, output, settings, *, repo_id=None, api=None, progress=print):
    root = Path(input_directory).resolve(); output = Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise DissertationError('Backup receipt output must be separate from the compression directory.')
    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    if not token:
        raise DissertationError('Model backup requires HF_TOKEN with write permission; configure Env setup first.')
    if not (root/'report.json').is_file() or not (root/'checkpoint.json').is_file():
        raise DissertationError('Choose the completed compression experiment root containing report.json and checkpoint.json, not just model_4bit/.')
    source = json.loads((root/'report.json').read_text())
    if source.get('status') not in ('COMPLETED', 'COMPLETED_WITH_WARNINGS') or not source.get('final_model_ready') or source.get('errors'):
        raise DissertationError('Only completed compression experiments with a ready final model can be backed up.')
    fingerprint = source.get('fingerprint', '')
    if not re.fullmatch('[0-9a-f]{64}', fingerprint):
        raise DissertationError('Compression fingerprint is missing/invalid.')
    checkpoint = json.loads((root/'checkpoint.json').read_text())
    if checkpoint.get('fingerprint') != fingerprint:
        raise DissertationError('Compression report/checkpoint identity mismatch.')
    required_files = set()
    for stage in ('inputs', 'pruned', 'recovery', 'export'):
        if stage not in checkpoint.get('stages', {}):
            raise DissertationError('A committed compression stage is missing; complete-experiment backup refused.')
        folder = root/'model_4bit' if stage == 'export' else root/'_runs'/fingerprint/stage
        verify_stage(folder, checkpoint['stages'][stage])
        required_files.update((folder/name).relative_to(root).as_posix() for name in checkpoint['stages'][stage]['files'])
    if not (root/'model_4bit/config.json').is_file() or not any((root/'model_4bit').glob('*.safetensors')):
        raise DissertationError('Final model configuration/weights are missing.')
    api = api or HfApi(token=token)
    with output_lock(output):
        result = {'schema_version': VERSION, 'status': 'RUNNING', 'backup_verified': False,
            'compression_fingerprint': fingerprint, 'source_directory': str(root), 'errors': [],
            'started_at': datetime.now(timezone.utc).isoformat(),
            'note': 'Private experiment archive; saved model is in model_4bit/. Remote storage, not an offline guarantee.'}
        dump_json(output/'report.json', result)
        try:
            if repo_id is None:
                repo_id = api.whoami()['name'] + '/qwen3-enterprise-' + fingerprint[:12] + '-backup'
            if not re.fullmatch(r'[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*', repo_id):
                raise DissertationError('Model repository must be OWNER/NAME.')
            result['repo_id'] = repo_id
            progress('[1/4] Verifying committed stages and hashing experiment files')
            paths = selected_files(root, required_files)
            if not required_files <= paths.keys():
                raise DissertationError('A committed stage contains excluded credential/key files; backup refused.')
            secrets = [token, settings.model.api_key.get_secret_value() if settings.model.api_key else None]
            expected = {name: checksums(path, secrets) for name, path in paths.items()}
            if not expected: raise DissertationError('No backup files found.')
            dump_json(output/'file_manifest.json', expected)
            api.create_repo(repo_id, repo_type='model', private=True, exist_ok=True)
            if api.model_info(repo_id).private is not True:
                raise DissertationError('Destination repository is public; backup refused. Use a private repository.')
            progress('[2/4] Uploading private experiment archive; rerun identical command after interruption')
            api.upload_large_folder(repo_id=repo_id, repo_type='model', folder_path=root,
                                   allow_patterns=list(expected), num_workers=2)
            revision = api.model_info(repo_id).sha
            progress('[3/4] Verifying all selected remote files against local content hashes')
            remote = {entry.path: entry for entry in api.list_repo_tree(repo_id, repo_type='model', revision=revision,
                                                                      recursive=True) if hasattr(entry, 'size')}
            for i, (name, path) in enumerate(paths.items(), 1):
                entry = remote.get(name); record = expected[name]
                if entry is None or entry.size != record['bytes']:
                    raise DissertationError('Remote backup has missing or size-mismatched files; do not terminate storage.')
                lfs = getattr(entry, 'lfs', None)
                if lfs:
                    actual = lfs.get('sha256') if isinstance(lfs, dict) else lfs.sha256
                    wanted = record['sha256']
                else:
                    actual = entry.blob_id; wanted = record['git_blob_sha1']
                if actual != wanted or checksums(path) != record:
                    raise DissertationError('Remote content hash or local source changed; backup is unverified.')
                if i == len(paths) or i % 10 == 0: progress(f'  Verified {i}/{len(paths)} files')
            if set(selected_files(root, required_files)) != set(paths):
                raise DissertationError('Source file selection changed during backup.')
            progress('[4/4] Saving verified backup manifest and immutable revision')
            payload = {'schema_version': VERSION, 'compression_fingerprint': fingerprint, 'files': expected,
                       'verified_payload_revision': revision, 'private': True}
            committed = api.upload_file(repo_id=repo_id, repo_type='model',
                path_or_fileobj=json.dumps(payload, ensure_ascii=False, indent=2).encode(),
                path_in_repo='backup_manifest.json', commit_message='Record verified compression backup hashes')
            result.update(status='COMPLETED', backup_verified=True, revision=revision,
                manifest_revision=committed.oid, verified_files=len(expected),
                backup_bytes=sum(record['bytes'] for record in expected.values()),
                url='https://huggingface.co/'+repo_id,
                restoration='snapshot_download(repo_id=..., revision=revision, local_dir=...); model_4bit/ contains the final model')
            return result
        except BaseException as exc:
            result['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED'
            result['errors'].append(str(exc) if isinstance(exc, DissertationError) else 'backup_failed:'+type(exc).__name__)
            raise DissertationError('Backup not verified. Inspect the backup receipt report; source models are preserved. Rerun the same command to continue uploads.') from None
        finally:
            result['finished_at'] = datetime.now(timezone.utc).isoformat()
            dump_json(output/'report.json', result)
