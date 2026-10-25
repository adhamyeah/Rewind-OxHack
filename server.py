"""Answers "where is it?" from rewind.db.

Run:  uvicorn server:app --reload   then open http://localhost:8000 in Chrome
"""

import os
import re

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import db

SNAP_DIR = "snapshots"

app = FastAPI()

# Other words people use for the same thing -> the name main.py logs
SYNONYMS = {"specs": "glasses", "spectacles": "glasses", "key": "keys",
            "mobile": "phone", "purse": "wallet", "tablets": "pills",
            "remote": "remote control", "cup": "mug"}


def as_json(row):
    if row["event"] == "placed":
        status = row["place"] or row["zone"]
    else:
        status = "picked up" if row["picked_up"] else "out of view"
    return {"item": row["item"], "type": row["kind"], "status": status, "ago": db.ago(row["at"]),
            "photo": row["photo"], "answer": db.answer(row)}


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
    return {"answer": "Sorry, I didn't catch which item. Try saying phone, cup or bottle."}


# Routes must be declared above these mounts: the "/" mount catches every URL.
os.makedirs(SNAP_DIR, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=SNAP_DIR), name="snapshots")
app.mount("/", StaticFiles(directory="static", html=True), name="ui")
