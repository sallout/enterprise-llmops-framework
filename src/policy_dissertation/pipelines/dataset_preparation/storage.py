"""Disk-backed records, union-find groups, and deduplication prefix index."""
import json
import sqlite3


class PreparationStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
          PRAGMA journal_mode=WAL;
          PRAGMA synchronous=FULL;
          CREATE TABLE IF NOT EXISTS records (
            rid INTEGER PRIMARY KEY, pipeline TEXT NOT NULL, line INTEGER NOT NULL, raw_hash TEXT NOT NULL,
            payload TEXT, reason TEXT, duplicate_of INTEGER, root TEXT, split TEXT, selected INTEGER DEFAULT 0,
            exact_key TEXT, format_key TEXT, shingles BLOB, priority INTEGER, domain TEXT, source TEXT,
            status TEXT, tokens INTEGER, rank TEXT, UNIQUE(pipeline,line));
          CREATE TABLE IF NOT EXISTS stages (name TEXT PRIMARY KEY);
          CREATE TABLE IF NOT EXISTS parents (node TEXT PRIMARY KEY, parent TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS exacts (key TEXT PRIMARY KEY, rid INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS prefixes (token BLOB NOT NULL,rid INTEGER NOT NULL,n INTEGER NOT NULL,PRIMARY KEY(token,rid));
          CREATE TABLE IF NOT EXISTS near_pairs (a INTEGER NOT NULL,b INTEGER NOT NULL, similarity REAL NOT NULL, PRIMARY KEY(a,b));
          CREATE TABLE IF NOT EXISTS dedup_done (rid INTEGER PRIMARY KEY);
          CREATE INDEX IF NOT EXISTS duplicates ON records(duplicate_of);
          CREATE INDEX IF NOT EXISTS roots ON records(root);
          CREATE INDEX IF NOT EXISTS selection_order ON records(pipeline,selected,rank);
          CREATE TEMP TABLE candidates (rid INTEGER PRIMARY KEY);
        ''')
        if self.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Invalid checkpoint database')

    def complete(self, name):
        return bool(self.db.execute('SELECT 1 FROM stages WHERE name=?', (name,)).fetchone())

    def finish(self, name):
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO stages VALUES (?)', (name,))

    def find(self, node):
        self.db.execute('INSERT OR IGNORE INTO parents VALUES (?,?)', (node,node))
        path, current = [], node
        while True:
            parent = self.db.execute('SELECT parent FROM parents WHERE node=?', (current,)).fetchone()[0]
            if parent == current:
                break
            path.append(current)
            current = parent
        for item in path:
            self.db.execute('UPDATE parents SET parent=? WHERE node=?', (current,item))
        return current

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            small, large = sorted((a,b))
            self.db.execute('UPDATE parents SET parent=? WHERE node=?', (small,large))

    def canonical(self, rid):
        while True:
            row = self.db.execute('SELECT duplicate_of FROM records WHERE rid=?', (rid,)).fetchone()
            if row[0] is None:
                return rid
            rid = row[0]

    def rows(self, where='1', params=()):
        return self.db.execute('SELECT * FROM records WHERE '+where+' ORDER BY rid', params)

    def count(self, where='1', params=()):
        return self.db.execute('SELECT count(*) FROM records WHERE '+where, params).fetchone()[0]

    def counts(self, field, where='1'):
        assert field in ('pipeline','domain','source','status','split','reason')
        return {row[0]:row[1] for row in self.db.execute(f'SELECT {field},count(*) FROM records WHERE {where} GROUP BY {field}')}

    def close(self):
        self.db.close()
