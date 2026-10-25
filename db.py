"""SQLite helpers shared by main.py (writes) and server.py (reads).

One row each time an item is placed somewhere or goes from view.
"""

import sqlite3

DB_PATH = "rewind.db"

SCHEMA = """CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    item TEXT, event TEXT,          -- event: 'placed' or 'gone'
    zone TEXT, x1 INT, y1 INT, x2 INT, y2 INT,
    at REAL,                        -- unix time
    image TEXT,                     -- snapshot path, placed events only
    picked_up INT DEFAULT 0,
    description TEXT)               -- Gemini's "next to the blue mug", filled in a moment later"""


def open_db(path=DB_PATH):
    db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")   # lets the server read while main.py writes
    db.execute(SCHEMA)
    if "description" not in [c["name"] for c in db.execute("PRAGMA table_info(events)")]:
        db.execute("ALTER TABLE events ADD COLUMN description TEXT")   # older rewind.db files
    return db


def log_event(db, item, event, zone, box, at, image=None, picked_up=False):
    cur = db.execute(
        "INSERT INTO events (item, event, zone, x1, y1, x2, y2, at, image, picked_up) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (item, event, zone, *box, at, image, int(picked_up)))
    db.commit()
    return cur.lastrowid


def set_description(db, event_id, description):
    db.execute("UPDATE events SET description = ? WHERE id = ?", (description, event_id))
    db.commit()


def latest_rows(db):
    """Newest event per item, plus the photo and description from its last 'placed' event."""
    return db.execute("""
        SELECT e.*, p.image AS photo, p.description AS place
        FROM events e
        LEFT JOIN events p ON p.id = (SELECT MAX(id) FROM events
                                      WHERE item = e.item AND event = 'placed')
        WHERE e.id IN (SELECT MAX(id) FROM events GROUP BY item)
        ORDER BY e.at DESC""").fetchall()
