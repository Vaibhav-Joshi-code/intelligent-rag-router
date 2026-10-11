"""SQLite manifest, bounded prompt history, and idempotent completed requests.

One API process is supported. Per-session locks protect concurrent threads;
cross-process request coordination is intentionally not claimed.
"""
import hashlib
import json
import sqlite3  
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class Conflict(Exception):
    pass


class Busy(Exception):
    pass


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.guard, self.locks = threading.Lock(), {}
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS manifest (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, session TEXT NOT NULL,
                    question TEXT NOT NULL, answer TEXT NOT NULL, created REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS turns_session ON turns(session,id);
                CREATE TABLE IF NOT EXISTS responses (request_id TEXT PRIMARY KEY, session TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, response TEXT NOT NULL, created REAL NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def session_lock(self, session):
        with self.guard:
            if session not in self.locks:
                if len(self.locks) >= 1000:
                    inactive = next((key for key, value in self.locks.items() if value[1] == 0), None)
                    if inactive is None:
                        raise Busy('Too many active conversations.')
                    del self.locks[inactive]
                self.locks[session] = [threading.Lock(), 0]
            entry = self.locks[session]
            entry[1] += 1
        acquired = entry[0].acquire(blocking=False)
        try:
            if not acquired:
                raise Busy('A turn is already running in this conversation; retry shortly.')
            yield
        finally:
            if acquired:
                entry[0].release()
            with self.guard:
                entry[1] -= 1

    def manifest(self):
        with self.connect() as db:
            row = db.execute('SELECT value FROM manifest WHERE id=1').fetchone()
        if not row:
            raise ValueError('No active corpus. Run python ingest.py first.')
        return json.loads(row[0])


    def publish(self, value):
        with self.connect() as db:
            db.execute(
                'INSERT INTO manifest VALUES (1, ?) '
                'ON CONFLICT(id) DO UPDATE SET value=excluded.value',
                (json.dumps(value),)
            )

    def history(self, session, limit):
        with self.connect() as db:
            rows = db.execute('SELECT question,answer FROM turns WHERE session=? ORDER BY id DESC LIMIT ?',
                              (session, limit)).fetchall()
        return [{'role': role, 'content': row[i]} for row in reversed(rows)
                for i, role in enumerate(('human', 'ai'))]

    @staticmethod
    def fingerprint(payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def cached(self, payload):
        with self.connect() as db:
            row = db.execute('SELECT fingerprint,response FROM responses WHERE request_id=?',
                             (payload['request_id'],)).fetchone()
        if row and row[0] != self.fingerprint(payload):
            raise Conflict('This request_id already belongs to a different payload; use a new ID for a new turn.')
        return json.loads(row[1]) if row else None

    def complete(self, payload, response):
        with self.connect() as db:
            try:
                db.execute('INSERT INTO responses VALUES (?,?,?,?,?)',
                    (payload['request_id'], payload['session_id'], self.fingerprint(payload), json.dumps(response), time.time()))
            except sqlite3.IntegrityError as error:
                raise Conflict('A request with this ID already completed.') from error
            db.execute('INSERT INTO turns(session,question,answer,created) VALUES (?,?,?,?)',
                (payload['session_id'], payload['question'], response['answer'], time.time()))

    def clear(self, session):
        with self.session_lock(session), self.connect() as db:
            db.execute('DELETE FROM turns WHERE session=?', (session,))
            db.execute('DELETE FROM responses WHERE session=?', (session,))

    def maintenance(self, days=30):
        if days < 1:
            raise ValueError('Retention must be at least one day.')
        with self.connect() as db:
            for table in ('turns', 'responses'):
                db.execute(f'DELETE FROM {table} WHERE created < ?', (time.time() - days * 86400,))
