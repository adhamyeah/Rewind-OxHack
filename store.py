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
    # Which run a row came from (keyframe numbers restart at 0 every run). Older rows stay NULL.
    "ALTER TABLE rewind_keyframes ADD COLUMN IF NOT EXISTS session text",
    "ALTER TABLE rewind_events ADD COLUMN IF NOT EXISTS session text",
    # A continuous aggregate: Tiger Data keeps this 5-minute summary up to date by itself.
    """CREATE MATERIALIZED VIEW IF NOT EXISTS rewind_events_5min
       WITH (timescaledb.continuous) AS
       SELECT time_bucket('5 minutes', time) AS bucket, label, kind, count(*) AS n
       FROM rewind_events GROUP BY 1, 2, 3
       WITH NO DATA""",
]


class Store:
    def __init__(self, url=config.TIGER_DATABASE_URL):
        self.conn = None
        self.session = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")  # one id per run
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
                cur.execute("INSERT INTO rewind_keyframes (time, keyframe, detections, session) VALUES (%s, %s, %s, %s)",
                            (ts, kf.index, json.dumps([list(d) for d in kf.detections]), self.session))
                for ev in events:
                    cur.execute("INSERT INTO rewind_events (time, label, kind, keyframe, x1, y1, x2, y2, session) "
                                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                                (ts, ev.label, ev.kind, ev.keyframe, *ev.box, self.session))
        except Exception as e:
            print(f"[store] write failed, turning Tiger Data off: {e}")
            self.conn = None
