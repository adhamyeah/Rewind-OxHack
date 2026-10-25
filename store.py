"""Optional Tiger Data (TimescaleDB) store. If anything goes wrong it prints once and turns itself off."""

import datetime as dt
import json

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


class Store:
    def __init__(self, url=config.TIGER_DATABASE_URL):
        self.conn = None
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

    @property
    def enabled(self):
        return self.conn is not None

    def save(self, kf, events):
        if not self.enabled:
            return
        try:
            ts = dt.datetime.fromtimestamp(kf.time, dt.timezone.utc)
            with self.conn.cursor() as cur:
                cur.execute("INSERT INTO rewind_keyframes VALUES (%s, %s, %s)",
                            (ts, kf.index, json.dumps([list(d) for d in kf.detections])))
                for ev in events:
                    cur.execute("INSERT INTO rewind_events VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                                (ts, ev.label, ev.kind, ev.keyframe, *ev.box))
        except Exception as e:
            print(f"[store] write failed, turning Tiger Data off: {e}")
            self.conn = None
