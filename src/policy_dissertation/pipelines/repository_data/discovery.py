"""Safe source indexing with snapshot/commit references and static context selection."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib

from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.data_io import digest, dump_json, dump_jsonl
from .parsing import analyze
from .schemas import CodeUnit

LANGUAGES = {'.py': 'python', '.kt': 'kotlin', '.kts': 'kotlin', '.ts': 'typescript', '.tsx': 'typescript'}
IGNORED = {'.git', '.venv', 'venv', 'node_modules', 'vendor', 'build', 'dist', 'target', '.gradle',
           '.idea', '__pycache__', '.pytest_cache', '.next', 'coverage', 'generated', 'generated-sources',
           'migrations', 'alembic', '.turbo', '.cache', 'site-packages', '.mypy_cache', 'htmlcov', 'playwright-report'}
SECRET = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b|(?i:api[_-]?key|secret[_-]?key|password)\s*[:=]\s*["\'][^"\'\n]{12,}["\']')
GENERATED = re.compile(r'(?i)(?:automatically generated|auto-generated|@generated|do not edit)')
MANIFESTS = {'pyproject.toml', 'package.json', 'build.gradle.kts', 'settings.gradle.kts', 'libs.versions.toml'}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def has_secret(text: str, api_key: str = '') -> bool:
    return bool(SECRET.search(text) or (api_key and api_key in text))


def repository_roots(source: Path) -> list[Path]:
    markers = {'.git', 'pyproject.toml', 'package.json', 'settings.gradle.kts', 'settings.gradle'}
    if any((source/name).exists() for name in markers) or any(p.is_file() and p.suffix.lower() in LANGUAGES for p in source.iterdir()):
        return [source]
    return sorted(p for p in source.iterdir() if p.is_dir() and not p.is_symlink() and not p.name.startswith('.'))


def commit_reference(root: Path) -> str | None:
    # Read only the local .git directory; never invoke repository hooks/configuration.
    git = root/'.git'
    if not git.is_dir() or git.is_symlink():
        return None
    try:
        head = (git/'HEAD').read_text().strip()
        if re.fullmatch(r'[a-fA-F0-9]{40,64}', head):
            return head
        if head.startswith('ref: '):
            ref = head[5:]
            location = (git/ref).resolve()
            if not location.is_relative_to(git.resolve()):
                return None
            if location.is_file():
                value = location.read_text().strip()
                return value if re.fullmatch(r'[a-fA-F0-9]{40,64}', value) else None
            packed = git/'packed-refs'
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    fields = line.split()
                    if len(fields) == 2 and fields[1] == ref and re.fullmatch(r'[a-f0-9]{40,64}', fields[0]):
                        return fields[0]
    except OSError:
        pass
    return None


def index_repositories(source: Path, output: Path, max_tokens: int, *, max_file_bytes=250000,
                       limit_units: int | None = None, api_key: str = '') -> dict:
    source = source.resolve()
    if not source.is_dir():
        raise DissertationError('Repository input directory does not exist.')
    if max_tokens < 1 or (limit_units is not None and limit_units < 1):
        raise DissertationError('Source chunk and unit limits must be positive.')
    roots = repository_roots(source)
    repositories, all_units, exclusions = [], [], []
    counters = Counter()
    for root in roots:
        files, context, licenses = [], [], []
        def walk_error(error):
            counters['unreadable_directories'] += 1
            exclusions.append({'repository': root.name, 'path': str(error.filename), 'reason': 'unreadable_directory'})
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED and not d.startswith('.') and not (Path(directory)/d).is_symlink())
            for name in sorted(names):
                path = Path(directory)/name
                rel = path.relative_to(root).as_posix()
                if path.is_symlink() or name.startswith('.') or name.endswith(('.min.ts', '.d.ts')) or name.startswith(('secrets', 'credentials')):
                    counters['excluded_sensitive_or_generated_path'] += 1
                    continue
                relevant = path.suffix.lower() in LANGUAGES or name in MANIFESTS or name.startswith(('LICENSE', 'NOTICE'))
                if not relevant:
                    counters['excluded_other'] += 1
                    continue
                try:
                    if path.stat().st_size > max_file_bytes:
                        exclusions.append({'repository': root.name, 'path': rel, 'reason': 'file_size_limit'})
                        counters['excluded_large'] += 1
                        continue
                    raw = path.read_bytes()
                    text = raw.decode('utf-8-sig')
                except (OSError, UnicodeError):
                    counters['unreadable_files'] += 1
                    exclusions.append({'repository': root.name, 'path': rel, 'reason': 'unreadable_or_non_utf8'})
                    continue
                if '\x00' in text or has_secret(text, api_key):
                    exclusions.append({'repository': root.name, 'path': rel, 'reason': 'binary_or_credential_like_content'})
                    counters['excluded_sensitive_or_binary'] += 1
                    continue
                if name.startswith(('LICENSE', 'NOTICE')):
                    licenses.append({'file_path': rel, 'content_hash': sha(raw), 'excerpt': text[:600]})
                if name in MANIFESTS:
                    dependencies = {}
                    try:
                        parsed = json.loads(text) if name == 'package.json' else tomllib.loads(text) if name.endswith('.toml') else {}
                        dependencies = {k: parsed[k] for k in ('dependencies', 'devDependencies', 'project', 'dependency-groups') if k in parsed}
                    except (ValueError, tomllib.TOMLDecodeError):
                        pass
                    context.append({'file_path': rel, 'content_hash': sha(raw), 'declarations': dependencies,
                                    'excerpt': text[:max_tokens//2]})
                if path.suffix.lower() not in LANGUAGES:
                    continue
                if GENERATED.search(text[:1500]) or '/client/' in '/'+rel or name in ('routeTree.gen.ts', 'openapi.ts'):
                    counters['excluded_generated'] += 1
                    continue
                files.append({'path': rel, 'hash': sha(raw), 'text': text,
                              'language': LANGUAGES[path.suffix.lower()], 'suffix': path.suffix.lower()})
        # Content snapshot, not a fabricated Git commit for downloaded archives.
        snapshot = digest([[f['path'], f['hash']] for f in files] + [[c['file_path'], c['content_hash']] for c in context])
        repo_id, commit = digest(str(root.resolve())), commit_reference(root)
        repo_units = []
        for file in files:
            parsed = analyze(file['text'], file['language'], file['suffix'])
            counters['source_files'] += 1
            if not parsed['valid']:
                counters['files_with_syntax_diagnostics'] += 1
            symbols = parsed['symbols'] or [{'symbol': '<module>', 'kind': 'module', 'start_line': 1,
                                            'end_line': len(file['text'].splitlines()), 'valid': parsed['valid']}]
            lines = file['text'].splitlines(keepends=True)
            for symbol in symbols:
                code = ''.join(lines[symbol['start_line']-1:symbol['end_line']])
                counters['symbols_discovered'] += 1
                if not symbol['valid'] or not code.strip():
                    counters['invalid_symbols'] += 1
                    continue
                if len(code.strip()) < 20:
                    counters['too_short_symbols'] += 1
                    continue
                if len(code.encode()) > max_tokens:
                    counters['oversized_symbols'] += 1
                    continue
                unit_id = digest([repo_id, snapshot, file['path'], file['hash'], symbol])
                is_test = '/src/debug/' in '/'+file['path'] or file['path'].startswith('shared-test/') or bool(re.search(r'(^|/)(?:tests?|androidTest)(/|$)|(?:Test|Tests|\.test|\.spec)\.', file['path']))
                repo_units.append(CodeUnit(
                    unit_id=unit_id, repository_id=repo_id, repository_name=root.name,
                    repository_path=str(root.resolve()), commit=commit, snapshot_hash=snapshot,
                    file_path=file['path'], file_hash=file['hash'], symbol=symbol['symbol'], kind=symbol['kind'],
                    language=file['language'], start_line=symbol['start_line'], end_line=symbol['end_line'],
                    code=code, imports=parsed['imports'][:30], syntax_valid=True, is_test=is_test,
                    token_count=len(code.encode()), dependencies=context, licenses=licenses,
                ))
        for unit in repo_units:
            # Static matching, not a claim of complete call/dependency resolution.
            related = [u for u in repo_units if u.is_test and u.unit_id != unit.unit_id
                       and (unit.symbol in u.code or Path(unit.file_path).stem.replace('Test', '') in u.file_path)]
            unit.related_tests = [{'file_path': u.file_path, 'file_hash': u.file_hash, 'symbol': u.symbol,
                                   'start_line': u.start_line, 'end_line': u.end_line, 'code': u.code}
                                  for u in related[:2]]
            peers = [u for u in repo_units if u.file_path == unit.file_path and u.unit_id != unit.unit_id
                     and abs(u.start_line-unit.start_line) < 100]
            unit.local_context = [{'symbol': u.symbol, 'start_line': u.start_line, 'end_line': u.end_line,
                                   'file_path': u.file_path, 'code': u.code} for u in peers[:2]]
        # Avoid source/test overlap as separate training anchors by selecting production anchors.
        anchors = [u for u in repo_units if not u.is_test]
        anchors.sort(key=lambda u: (-bool(u.related_tests), 0 if 'function' in u.kind or u.kind == 'function' else 1, u.token_count, u.file_path, u.start_line))
        all_units.append(anchors)
        repositories.append({'repository_id': repo_id, 'name': root.name, 'path': str(root.resolve()),
                             'commit': commit, 'snapshot_hash': snapshot, 'source_files': len(files),
                             'eligible_units': len(anchors), 'license_files': licenses,
                             'commit_warning': None if commit else 'Git commit unavailable; snapshot hash used.'})
    # Round-robin selection ensures a small smoke test can cover both repositories.
    selected = []
    for index in range(max([len(units) for units in all_units], default=0)):
        for units in all_units:
            if index < len(units) and (limit_units is None or len(selected) < limit_units):
                selected.append(units[index])
    counters['eligible_units'] = sum(len(x) for x in all_units)
    counters['selected_units'] = len(selected)
    counters['excluded_by_unit_limit'] = counters['eligible_units']-len(selected)
    manifest = {'repositories': repositories, 'counts': dict(counters), 'exclusions': exclusions,
                'unit_ids': [u.unit_id for u in selected], 'limit_units': limit_units,
                'max_tokens': max_tokens, 'token_method': 'utf8_bytes_upper_bound_v1'}
    dump_jsonl(output/'units.jsonl', [u.model_dump() for u in selected])
    dump_json(output/'manifest.json', manifest)
    return manifest
