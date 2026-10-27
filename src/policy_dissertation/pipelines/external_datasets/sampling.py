"""Seeded, bounded prefix-buffer sampling; explicitly not whole-corpus uniform sampling."""
import json
import random
from policy_dissertation.pipelines.policy_transformation.data_io import digest


def sampled(stream, seed, settings, on_read):
    rng = random.Random(seed)
    original = iter(stream)
    enumerated = enumerate(original)
    def read():
        ordinal, raw = next(enumerated)
        serialized = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        size = len(serialized.encode('utf-8'))
        on_read(size, raw)
        return {'ordinal': ordinal, 'raw_hash': digest(raw), 'size': size,
                'raw': raw if size <= settings.max_record_bytes else None,
                'rejection': 'oversized_source_record' if size > settings.max_record_bytes else None}
    buffer = []
    try:
        try:
            for _ in range(settings.shuffle_buffer):
                buffer.append(read())
        except StopIteration:
            pass
        while buffer:
            i = rng.randrange(len(buffer))
            yield buffer[i]
            try:
                buffer[i] = read()
            except StopIteration:
                buffer.pop(i)
    finally:
        close = getattr(original, 'close', None)
        if close:
            close()
