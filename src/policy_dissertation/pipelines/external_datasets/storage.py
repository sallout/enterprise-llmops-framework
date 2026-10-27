"""Transactional per-record decisions with bounded memory and exact deduplication."""
import json
import sqlite3
from pathlib import Path
from policy_dissertation.errors import DissertationError


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        if self.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise DissertationError('checkpoint_database_integrity_failed')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS decisions (
            source TEXT NOT NULL, position INTEGER NOT NULL, ordinal INTEGER NOT NULL,
            raw_hash TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, content_key TEXT,
            PRIMARY KEY(source,position));
          CREATE UNIQUE INDEX IF NOT EXISTS accepted_content ON decisions(content_key) WHERE kind='accepted';
          CREATE TABLE IF NOT EXISTS progress (source TEXT PRIMARY KEY, processed INTEGER NOT NULL, accepted INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS exhausted (source TEXT PRIMARY KEY);
        ''')

    def counts(self, source):
        row = self.db.execute('SELECT processed,accepted FROM progress WHERE source=?', (source,)).fetchone()
        return row if row else (0, 0)

    def exhausted(self, source):
        return self.db.execute('SELECT 1 FROM exhausted WHERE source=?', (source,)).fetchone() is not None

    def finish(self, source):
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO exhausted VALUES (?)', (source,))

    def commit(self, source, position, item, accepted, rejected, content_key):
        if content_key and self.db.execute("SELECT 1 FROM decisions WHERE kind='accepted' AND content_key=?", (content_key,)).fetchone():
            accepted = None
            rejected['reasons'] = ['duplicate_conversation']
        kind, payload = ('accepted', accepted) if accepted else ('rejected', rejected)
        with self.db:
            self.db.execute('INSERT INTO decisions VALUES (?,?,?,?,?,?,?)',
                (source, position, item['ordinal'], item['raw_hash'], kind,
                 json.dumps(payload, ensure_ascii=False), content_key if accepted else None))
            self.db.execute('INSERT INTO progress VALUES (?,1,?) ON CONFLICT(source) DO UPDATE SET processed=processed+1, accepted=accepted+excluded.accepted',
                            (source, int(kind == 'accepted')))
        return kind

    def replay_hash(self, source, position):
        return self.db.execute('SELECT raw_hash FROM decisions WHERE source=? AND position=?', (source, position)).fetchone()[0]

    def rows(self, kind):
        for row in self.db.execute('SELECT payload FROM decisions WHERE kind=? ORDER BY source,position', (kind,)):
            yield json.loads(row[0])

    def close(self):
        self.db.close()
