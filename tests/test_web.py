import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from answer import Answerer
from memory import Memory
from ui import UIState
from web import App, start_server


class StubCam:
    ended = False

    def __init__(self):
        self.frame = np.full((120, 160, 3), 90, dtype=np.uint8)

    def latest(self):
        return 1, self.frame


class StubDet:
    item_ms = 200.0
    dirty = False

    def latest(self):
        return []


class StubWorker:
    status, last_source, busy = "idle", "gemini 1.2s", False


@pytest.fixture()
def server():
    mem = Memory(160, 120)
    img = np.full((120, 160, 3), 50, dtype=np.uint8)
    mem.add_keyframe([(10, 10, 40, 40, "mug", 1.0), (60, 10, 90, 40, "pen", 1.0)], image=img)
    mem.add_keyframe([(10, 10, 40, 40, "mug", 1.0)], image=img)
    ui_state = UIState()
    spoken = []
    answerer = Answerer(mem, ui_state, speak=spoken.append, pick_label=lambda q: None)
    app = App(mem, ui_state, StubCam(), StubDet(), StubWorker(), answerer, gemini_on=True)
    srv, url = start_server(app, 0)  # port 0 = any free port
    yield url, ui_state
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_page_is_served_with_big_header(server):
    status, ctype, body = get(server[0] + "/")
    assert status == 200 and "text/html" in ctype
    assert b"<h1>Rewind" in body and b'type="range"' in body


def test_state_lists_keyframes_and_items(server):
    state = json.loads(get(server[0] + "/api/state")[2])
    assert [k["labels"] for k in state["keyframes"]] == [["mug", "pen"], ["mug"]]
    assert state["present"] == ["mug", "pen"]
    assert state["status"]["gemini"] is True and state["status"]["live_ms"] == 200
    assert state["jump"] is None


def test_keyframe_image_and_highlight(server):
    for path in ("/frame/0.jpg", "/frame/1.jpg?hl=mug"):
        status, ctype, body = get(server[0] + path)
        assert status == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8"


def test_missing_keyframe_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server[0] + "/frame/9.jpg")
    assert e.value.code == 404


def test_ask_returns_answer_and_jump(server):
    data = post(server[0] + "/api/ask", {"question": "where is the pen?"})
    assert "pen" in data["text"]
    assert data["jump"]["keyframe"] == 0 and data["jump"]["label"] == "pen"
    # the jump was consumed by the response, so the poll does not repeat it
    assert json.loads(get(server[0] + "/api/state")[2])["jump"] is None


def test_terminal_question_reaches_page_via_poll(server):
    url, ui_state = server
    ui_state.request_jump(1, (10, 10, 40, 40), "mug", "The mug is on the table.")
    jump = json.loads(get(url + "/api/state")[2])["jump"]
    assert jump == {"keyframe": 1, "label": "mug", "caption": "The mug is on the table."}


def test_empty_question_is_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server[0] + "/api/ask", {"question": "  "})
    assert e.value.code == 400


def test_live_stream_sends_multipart_jpeg(server):
    with urllib.request.urlopen(server[0] + "/stream.mjpg", timeout=5) as r:
        assert "multipart/x-mixed-replace" in r.headers["Content-Type"]
        chunk = r.read(400)
    assert b"--frame" in chunk and b"image/jpeg" in chunk


# ---- voice questions ----
def listen(url, audio, content_type="audio/webm"):
    req = urllib.request.Request(url + "/api/listen", audio, {"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def listen_error(url, audio):
    with pytest.raises(urllib.error.HTTPError) as e:
        listen(url, audio)
    return e.value.code, json.loads(e.value.read())["error"]


def make_server(transcribe):
    mem = Memory(160, 120)
    img = np.full((120, 160, 3), 50, dtype=np.uint8)
    mem.add_keyframe([(10, 10, 40, 40, "mug", 1.0)], image=img)
    ui_state = UIState()
    answerer = Answerer(mem, ui_state, speak=lambda t: None, pick_label=lambda q: None)
    app = App(mem, ui_state, StubCam(), StubDet(), StubWorker(), answerer, True, transcribe=transcribe)
    return start_server(app, 0)


def test_voice_question_is_answered_like_a_typed_one():
    seen = {}

    def fake(audio, content_type):
        seen["args"] = (len(audio), content_type)
        return " Where is the mug? "

    srv, url = make_server(fake)
    try:
        data = listen(url, b"x" * 5000, "audio/webm;codecs=opus")
    finally:
        srv.shutdown()
    assert data["heard"] == "Where is the mug?" and "mug" in data["text"]
    assert data["jump"]["label"] == "mug"
    assert seen["args"] == (5000, "audio/webm;codecs=opus")


def test_voice_empty_transcript_and_tiny_recording_are_friendly_errors():
    srv, url = make_server(lambda a, c: "   ")
    try:
        assert listen_error(url, b"x" * 5000)[0] == 422
        assert listen_error(url, b"x" * 10)[0] == 400
    finally:
        srv.shutdown()


def test_voice_transcription_failure_does_not_crash_the_server():
    def boom(a, c):
        raise RuntimeError("401 unauthorized sk_secret")

    srv, url = make_server(boom)
    try:
        code, msg = listen_error(url, b"x" * 5000)
        assert code == 502 and "type the question" in msg and "sk_secret" not in msg
        assert json.loads(get(url + "/api/state")[2])["present"] == ["mug"]  # still serving
    finally:
        srv.shutdown()


def test_voice_too_long_is_rejected():
    srv, url = make_server(lambda a, c: "mug")
    try:
        assert listen_error(url, b"x" * 5_000_001)[0] == 413
    finally:
        srv.shutdown()
