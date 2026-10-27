"""Atomic files and checksummed per-chunk checkpoints; no credentials are stored."""
import hashlib
import json
from pathlib import Path
import tempfile


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def dump_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def dump_jsonl(path: Path, values) -> None:
    atomic_text(path, ''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values))


def save_cache(path: Path, key: str, data) -> None:
    dump_json(path, {'key': key, 'digest': digest(data), 'data': data})


def load_cache(path: Path, key: str):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if value['key'] == key and value['digest'] == digest(value['data']):
            return value['data']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None
