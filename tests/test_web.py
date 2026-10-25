import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from answer import Answerer
from conftest import ITEMS, run
from ui import UIState
from web import App, clean_zones, start_server

KEYS = (100, 300, 160, 340, "keys", 0.5)
MUG = (190, 290, 250, 350, "mug", 0.6)


class StubCam:
    ended = False

    def __init__(self):
        self.frame = np.full((480, 640, 3), 90, dtype=np.uint8)

    def latest(self):
        return 1, self.frame


class StubDet:
    item_ms = 200.0

    def latest(self):
        return [KEYS]


class StubListener:
    enabled, listening = True, True

    def __init__(self, heard=" Where are my keys? ", fail=False):
        self.heard, self.fail, self.got = heard, fail, None

    def transcribe_upload(self, audio, content_type):
        self.got = (len(audio), content_type)
        if self.fail:
            raise RuntimeError("401 unauthorized sk_secret")
        return self.heard


class Quiet:
    enabled = True

    def say(self, text):
        pass

    def busy(self):
        return False


@pytest.fixture()
def server(mem, clock):
    run(mem, clock, [KEYS, MUG], 2.5)
    ui_state = UIState()
    answerer = Answerer(mem, Quiet(), ui_state, ITEMS)
    listener = StubListener()
    app = App(mem, ui_state, StubCam(), StubDet(), answerer, listener, Quiet())
    srv, url = start_server(app, 0)  # port 0 = any free port
    yield url, ui_state, listener, app
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, payload, raw=None, content_type="application/json"):
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data, {"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def error_of(fn):
    with pytest.raises(urllib.error.HTTPError) as e:
        fn()
    return e.value.code, json.loads(e.value.read() or b"{}").get("error", "")


def test_page_is_served_with_big_header(server):
    status, ctype, body = get(server[0] + "/")
    assert status == 200 and "text/html" in ctype
    assert b"<h1>Rewind" in body and b'type="range"' in body and b'id="zoneDraw"' in body


def test_state_lists_moments_items_and_events(server):
    state = json.loads(get(server[0] + "/api/state")[2])
    assert [k["labels"] for k in state["keyframes"]] == [["keys", "mug"]]
    assert state["present"] == ["keys", "mug"]
    assert [e["kind"] for e in state["events"]] == ["put down", "put down"]
    assert state["status"]["mic"] == "listening" and state["status"]["live_ms"] == 200
    assert state["jump"] is None and state["zones"] == {}


def test_moment_picture_and_gif(server):
    status, ctype, body = get(server[0] + "/frame/0.jpg?hl=keys")
    assert status == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8"
    status, ctype, body = get(server[0] + "/clip/0.gif?hl=keys&box=100,300,160,340&kind=put%20down")
    assert status == 200 and ctype == "image/gif" and body[:3] == b"GIF"


def test_missing_moment_is_404(server):
    for path in ("/frame/9.jpg", "/clip/9.gif"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(server[0] + path)
        assert e.value.code == 404


def test_ask_returns_answer_and_jump_once(server):
    data = post(server[0] + "/api/ask", {"question": "where are my keys?"})
    assert data["text"].startswith("Your keys are")
    assert data["jump"]["keyframe"] == 0 and data["jump"]["label"] == "keys" and data["jump"]["caption"] == data["text"]
    state = json.loads(get(server[0] + "/api/state")[2])
    assert state["jump"] is None and state["answer"]["text"] == data["text"]


def test_spoken_question_reaches_page_via_poll(server):
    url, ui_state, _, app = server
    app.answerer.heard("where's my mug")
    state = json.loads(get(url + "/api/state")[2])
    assert state["jump"]["label"] == "mug" and state["answer"]["heard"] == "where's my mug"


def test_empty_question_is_rejected(server):
    assert error_of(lambda: post(server[0] + "/api/ask", {"question": "  "}))[0] == 400


def test_live_stream_sends_multipart_jpeg(server):
    with urllib.request.urlopen(server[0] + "/stream.mjpg", timeout=5) as r:
        assert "multipart/x-mixed-replace" in r.headers["Content-Type"]
        chunk = r.read(400)
    assert b"--frame" in chunk and b"image/jpeg" in chunk


def test_talk_button_audio_is_answered_like_typing(server):
    url, ui_state, listener, _ = server
    post(url + "/api/talk", {"on": True})
    assert ui_state.talking()
    data = post(url + "/api/listen", None, raw=b"x" * 5000, content_type="audio/webm;codecs=opus")
    assert data["heard"] == "Where are my keys?" and data["text"].startswith("Your keys are")
    assert listener.got == (5000, "audio/webm;codecs=opus")
    assert not ui_state.talking()                 # let go: the always-on mic listens again


def test_talk_button_errors_are_friendly(server):
    url, _, listener, _ = server
    assert error_of(lambda: post(url + "/api/listen", None, raw=b"x" * 10, content_type="audio/webm"))[0] == 400
    listener.heard = "   "
    assert error_of(lambda: post(url + "/api/listen", None, raw=b"x" * 5000, content_type="audio/webm"))[0] == 422
    listener.fail = True
    code, msg = error_of(lambda: post(url + "/api/listen", None, raw=b"x" * 5000, content_type="audio/webm"))
    assert code == 502 and "type the question" in msg and "sk_secret" not in msg


def test_zones_can_be_drawn_from_the_page(server):
    url = server[0]
    zones = {"on the kitchen table": [[0, 0.6], [1, 0.6], [1, 1], [0, 1]]}
    assert post(url + "/api/zones", {"zones": zones})["zones"] == zones
    assert json.loads(get(url + "/api/zones")[2])["zones"] == zones
    data = post(url + "/api/ask", {"question": "keys"})
    assert "on the kitchen table" in json.loads(get(url + "/api/state")[2])["zones"]
    assert data["text"]  # still answers
    assert error_of(lambda: post(url + "/api/zones", {"zones": {"x": [[0, 0]]}}))[0] == 400


def test_clean_zones_clamps_and_rejects():
    assert clean_zones({"zones": {"  desk ": [[-1, 0], [2, 0], [0.5, 0.5]]}}) == {"desk": [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]}
    with pytest.raises(ValueError):
        clean_zones({"zones": {"": [[0, 0], [1, 0], [1, 1]]}})
