import time

import pytest
from fastapi.testclient import TestClient

import db


@pytest.fixture
def client(tmp_path, monkeypatch):
    # run inside an empty folder with its own rewind.db and the real static/ UI
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Rewind</h1>")
    monkeypatch.chdir(tmp_path)

    con = db.open_db()
    now = time.time()
    box = (10, 20, 30, 40)
    db.log_event(con, "keys", "placed", "on the kitchen table", box, now - 120, "snapshots/k.jpg")
    db.log_event(con, "glasses", "placed", "by the kettle", box, now - 30, "snapshots/g1.jpg")
    db.log_event(con, "glasses", "placed", "on the top shelf", box, now - 10, "snapshots/g2.jpg")
    db.log_event(con, "phone", "placed", "on the sofa", box, now - 900, "snapshots/p.jpg")
    db.log_event(con, "phone", "gone", "on the sofa", box, now - 300, None, picked_up=True)
    db.log_event(con, "remote control", "placed", "on the arm of the chair", box, now, "snapshots/r.jpg")
    con.close()

    import server
    return TestClient(server.app)


def test_items_returns_latest_event_per_item(client):
    items = {i["item"]: i for i in client.get("/items").json()}
    assert set(items) == {"keys", "glasses", "phone", "remote control"}
    assert items["glasses"]["status"] == "on the top shelf"
    assert items["glasses"]["photo"] == "snapshots/g2.jpg"
    assert items["keys"]["ago"] == "2 minutes ago"


def test_gone_item_keeps_last_photo_and_says_picked_up(client):
    phone = next(i for i in client.get("/items").json() if i["item"] == "phone")
    assert phone["status"] == "picked up"
    assert phone["photo"] == "snapshots/p.jpg"
    assert phone["answer"] == "Your phone was last on the sofa. Then someone picked it up, 5 minutes ago."


def test_where_answers_with_plural_grammar(client):
    r = client.get("/where", params={"q": "Where are my keys?"}).json()
    assert r["answer"] == "Your keys are on the kitchen table. I saw them there 2 minutes ago."


def test_where_uses_synonyms(client):
    assert client.get("/where", params={"q": "where are my specs"}).json()["item"] == "glasses"
    assert client.get("/where", params={"q": "have you seen the remote"}).json()["item"] == "remote control"


def test_where_does_not_match_inside_other_words(client):
    # "monkeys" contains "keys" but isn't asking about them
    r = client.get("/where", params={"q": "monkeys"}).json()
    assert "item" not in r


def test_ui_is_served_at_root(client):
    assert "Rewind" in client.get("/").text


@pytest.mark.parametrize("secs, text", [
    (10, "just now"), (60, "1 minute ago"), (300, "5 minutes ago"),
    (3600, "about 1 hour ago"), (7300, "about 2 hours ago")])
def test_ago(secs, text):
    import server
    assert server.ago(1000.0, now=1000.0 + secs) == text
