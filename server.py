"""Answers "where is it?" from rewind.db.

Run:  uvicorn server:app --reload   then open http://localhost:8000 in Chrome
"""

import os
import re
import time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import db

SNAP_DIR = "snapshots"

app = FastAPI()

# Other words people use for the same thing -> the name main.py logs
SYNONYMS = {"specs": "glasses", "spectacles": "glasses", "key": "keys",
            "mobile": "phone", "purse": "wallet", "tablets": "pills",
            "remote": "remote control", "cup": "mug"}


def ago(t, now=None):
    mins = int(((now or time.time()) - t) // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} minute{'s' * (mins != 1)} ago"
    hours = mins // 60
    return f"about {hours} hour{'s' * (hours != 1)} ago"


def answer(row):
    item, when = row["item"], ago(row["at"])
    where = row["place"] or row["zone"]          # Claude's description when we have it
    many = item.endswith("s")                    # keys, glasses -> plural
    are, them = ("are", "them") if many else ("is", "it")
    was = "were" if many else "was"
    if row["event"] == "placed":
        return f"Your {item} {are} {where}. I saw {them} there {when}."
    then = f"someone picked {them} up" if row["picked_up"] else f"{them} went out of view"
    return f"Your {item} {was} last {where}. Then {then}, {when}."


def as_json(row):
    if row["event"] == "placed":
        status = row["place"] or row["zone"]
    else:
        status = "picked up" if row["picked_up"] else "out of view"
    return {"item": row["item"], "status": status, "ago": ago(row["at"]),
            "photo": row["photo"], "answer": answer(row)}


def rows():
    con = db.open_db()
    try:
        return db.latest_rows(con)
    finally:
        con.close()


@app.get("/items")
def items():
    return [as_json(r) for r in rows()]


@app.get("/where")
def where(q: str):
    words = re.findall(r"[a-z]+", q.lower())
    text = " ".join(words + [SYNONYMS[w] for w in words if w in SYNONYMS])
    # longest names first, so "remote control" wins over a shorter overlapping name
    for row in sorted(rows(), key=lambda r: -len(r["item"])):
        if re.search(rf"\b{re.escape(row['item'])}\b", text):
            return as_json(row)
    return {"answer": "Sorry, I didn't catch which item. Try saying keys, glasses or wallet."}


# Routes must be declared above these mounts: the "/" mount catches every URL.
os.makedirs(SNAP_DIR, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=SNAP_DIR), name="snapshots")
app.mount("/", StaticFiles(directory="static", html=True), name="ui")
