"""Optional Tiger Data (TimescaleDB) copy of Rewind's moments and events.

Only used if TIGER_DATABASE_URL is set in .env. Writes happen on their own thread, so a slow
database never holds up the camera. If anything goes wrong it prints once and turns itself off.
Rewind's own memory (rewind.db) works the same with or without it.
"""

import datetime as dt
import json
import queue
import threading

import config

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS rewind_keyframes (
           time timestamptz NOT NULL, keyframe int NOT NULL, detections jsonb)""",
    """CREATE TABLE IF NOT EXISTS rewind_events (
           time timestamptz NOT NULL, label text NOT NULL, kind text NOT NULL,
           keyframe int, x1 int, y1 int, x2 int, y2 int)""",
    "SELECT create_hypertable('rewind_keyframes', 'time', if_not_exists => TRUE)",
    "SELECT create_hypertable('rewind_events', 'time', if_not_exists => TRUE)",
]


def stamp(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc)


class Store:
    def __init__(self, url=None):
        url = config.TIGER_DATABASE_URL if url is None else url
        self.conn = None
        self.jobs = queue.Queue(maxsize=500)
        if not url:
            return
        try:
            import psycopg

            self.conn = psycopg.connect(url, autocommit=True, connect_timeout=5)
            with self.conn.cursor() as cur:
                for stmt in SCHEMA:
                    cur.execute(stmt)
            print("[store] Tiger Data connected")
        except Exception as e:
            print(f"[store] Tiger Data unavailable, running without it: {e}")
            self.conn = None
            return
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def enabled(self):
        return self.conn is not None

    def save_keyframe(self, index, t, items):
        self._put("INSERT INTO rewind_keyframes VALUES (%s, %s, %s)",
                  (stamp(t), index, json.dumps([list(d) for d in items])))

    def save_event(self, t, label, kind, keyframe, box):
        self._put("INSERT INTO rewind_events VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                  (stamp(t), label, kind, keyframe, *box))

    def _put(self, sql, args):
        if self.enabled:
            try:
                self.jobs.put_nowait((sql, args))
            except queue.Full:
                pass  # database far behind: drop rather than use up memory

    def _loop(self):
        while self.conn is not None:
            sql, args = self.jobs.get()
            try:
                with self.conn.cursor() as cur:
                    cur.execute(sql, args)
            except Exception as e:
                print(f"[store] write failed, turning Tiger Data off: {e}")
                self.conn = None
