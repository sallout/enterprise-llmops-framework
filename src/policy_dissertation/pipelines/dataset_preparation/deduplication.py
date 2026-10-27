"""Exact and deterministic prefix-filtered shingle Jaccard comparison, backed by SQLite."""
import hashlib
import json
import math
import re
from policy_dissertation.pipelines.policy_transformation.data_io import digest


def features(record):
    text = '\n'.join(m['role']+'\n'+(m.get('content') or '')+json.dumps(m.get('tool_calls') or [],sort_keys=True) for m in record['messages'])
    if record.get('tools'):
        text += json.dumps(record['tools'], sort_keys=True)
    words = re.findall(r'\w+|[^\w\s]', text)
    width = min(5, len(words))
    values = sorted({hashlib.sha256(json.dumps(words[i:i+width],ensure_ascii=False).encode()).digest()
                     for i in range(max(1,len(words)-width+1))})
    exact = record['metadata'].get('preparation', {}).get('rendered_token_hash') or digest([record['messages'], record.get('tools')])
    # Do not collapse code indentation, operators, numbers, case, or mandatory language.
    code = record['domain'] == 'coding' or '```' in text or re.search(r'(?m)^\s*(?:def |class |function |const |return |fun )',text)
    formatted = record['messages'] if code else [m | {'content': re.sub(r'\s+', ' ', m.get('content') or '').strip()} for m in record['messages']]
    return exact, digest([formatted,record.get('tools')]), values


def unpack(blob):
    return {blob[i:i+32] for i in range(0,len(blob),32)}


def candidates(store, values, threshold):
    db = store.db
    db.execute('DELETE FROM candidates')
    n = len(values)
    prefix = values[:n-math.ceil(threshold*n-1e-10)+1]
    lower, upper = math.ceil(threshold*n-1e-10), math.floor(n/threshold+1e-10)
    for start in range(0,len(prefix),200):
        part = prefix[start:start+200]
        sql = 'INSERT OR IGNORE INTO candidates SELECT rid FROM prefixes WHERE n>=? AND n<=? AND token IN ('+','.join('?' for _ in part)+')'
        db.execute(sql, (lower,upper,*part))
    return prefix


def deduplicate(store, settings, progress):
    if store.complete('deduplication'):
        progress('  Reusing completed deduplication checkpoint')
        return
    for row in store.rows('payload IS NOT NULL'):
        rid = row['rid']
        if store.db.execute('SELECT 1 FROM dedup_done WHERE rid=?',(rid,)).fetchone():
            continue
        record = json.loads(row['payload'])
        exact, formatted, values = features(record)
        node = 'record:'+str(rid)
        with store.db:
            store.union(node,record['metadata']['component_group'])
            prefix = candidates(store, values, settings.near_threshold)
            duplicate = store.db.execute('SELECT rid FROM exacts WHERE key=?',(exact,)).fetchone()
            duplicate_rid, duplicate_kind = (store.canonical(duplicate[0]),'exact_duplicate') if duplicate else (None,None)
            current = set(values)
            for other in store.db.execute('SELECT r.* FROM records r JOIN candidates c ON c.rid=r.rid ORDER BY r.rid'):
                other_values = unpack(other['shingles'])
                similarity = len(current & other_values)/len(current | other_values)
                if similarity+1e-12 >= settings.near_threshold:
                    store.union(node,'record:'+str(other['rid']))
                    store.db.execute('INSERT OR IGNORE INTO near_pairs VALUES (?,?,?)',(other['rid'],rid,similarity))
                    if duplicate_rid is None and formatted == other['format_key'] and similarity+1e-12 >= settings.near_remove_threshold:
                        duplicate_rid, duplicate_kind = store.canonical(other['rid']),'near_format_duplicate'
            if duplicate_rid is not None:
                store.union(node,'record:'+str(duplicate_rid))
                previous = store.db.execute('SELECT priority FROM records WHERE rid=?',(duplicate_rid,)).fetchone()[0]
                if row['priority'] > previous:
                    store.db.execute('UPDATE records SET reason=?,duplicate_of=? WHERE rid=?',(duplicate_kind,rid,duplicate_rid))
                    store.db.execute('UPDATE exacts SET rid=? WHERE rid=?',(rid,duplicate_rid))
                else:
                    store.db.execute('UPDATE records SET reason=?,duplicate_of=? WHERE rid=?',(duplicate_kind,duplicate_rid,rid))
            winner = store.canonical(rid)
            store.db.execute('INSERT OR REPLACE INTO exacts VALUES (?,?)',(exact,winner))
            store.db.execute('UPDATE records SET exact_key=?,format_key=?,shingles=? WHERE rid=?',(exact,formatted,b''.join(values),rid))
            store.db.executemany('INSERT INTO prefixes VALUES (?,?,?)',((v,rid,len(values)) for v in prefix))
            store.db.execute('INSERT INTO dedup_done VALUES (?)',(rid,))
        if rid % 100 == 0:
            progress(f'  Compared {rid} source rows')
    # Roots include removed duplicates and unselected bridge records, retaining leakage constraints.
    with store.db:
        for row in store.rows('duplicate_of IS NOT NULL'):
            store.db.execute('UPDATE records SET duplicate_of=? WHERE rid=?',(store.canonical(row['rid']),row['rid']))
        for row in store.rows('payload IS NOT NULL'):
            store.db.execute('UPDATE records SET root=? WHERE rid=?',(store.find('record:'+str(row['rid'])),row['rid']))
    store.finish('deduplication')
