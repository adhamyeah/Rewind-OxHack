"""SQLite helpers shared by the detector (writes) and the server (reads).

The events table is the team's data contract: one row each time an item is
placed somewhere or goes from view.
"""

import sqlite3

DB_PATH = "rewind.db"

SCHEMA = """CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    item TEXT, event TEXT,          -- event: 'placed' or 'gone'
    zone TEXT, x1 INT, y1 INT, x2 INT, y2 INT,
    at REAL,                        -- unix time
    image TEXT,                     -- snapshot path, placed events only
    picked_up INT DEFAULT 0)"""


def open_db(path=DB_PATH):
    db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")   # lets the server read while the detector writes
    db.execute(SCHEMA)
    return db


def log_event(db, item, event, zone, box, at, image=None, picked_up=False):
    db.execute(
        "INSERT INTO events (item, event, zone, x1, y1, x2, y2, at, image, picked_up) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (item, event, zone, *box, at, image, int(picked_up)))
    db.commit()


def latest_rows(db):
    """Newest event per item, plus the photo from its last 'placed' event."""
    return db.execute("""
        SELECT e.*, (SELECT image FROM events p WHERE p.item = e.item AND p.event = 'placed'
                     ORDER BY p.id DESC LIMIT 1) AS photo
        FROM events e WHERE e.id IN (SELECT MAX(id) FROM events GROUP BY item)
        ORDER BY e.at DESC""").fetchall()
