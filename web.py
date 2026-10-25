"""
Web page on http://127.0.0.1:<port>, using only Python's built-in HTTP server (no extra packages).

  /                      the page (static/index.html)
  /stream.mjpg           the LIVE view as a motion-JPEG stream (an <img> tag plays it)
  /frame/<i>.jpg         moment i as a still, with its boxes; ?hl=keys&box=x1,y1,x2,y2 adds the yellow highlight
  /clip/<i>.gif          moment i as the rewind gif (the seconds before it, then it); same ?hl= / &box= / &kind=
  /api/state             JSON the page polls: moments, items on the table, events, status, pending jump, last answer
  POST /api/ask          {"question": "..."} -> {"text": ..., "jump": {...} or null}
  POST /api/talk         {"on": true/false} the Talk button is held / let go (the always-on mic waits meanwhile)
  POST /api/listen       raw audio recorded in the browser -> speech-to-text -> same as /api/ask, plus "heard"
  GET/POST /api/zones    the named areas ("on the kitchen table"), drawn on the page

Everything slow (detection, Gemini, voice) runs on other threads; these handlers only read
memory and encode pictures. Bound to 127.0.0.1 so only this computer can open the page.
"""

import json
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import ui

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STREAM_FPS = 20
MAX_AUDIO_BYTES = 5_000_000   # ~ several minutes of speech; a question is a few seconds
MAX_ZONES = 20


def encode_jpeg(image, quality=82):
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode image")
    return jpg.tobytes()


def parse_box(text):
    try:
        box = [int(float(v)) for v in text.split(",")]
        return box if len(box) == 4 else None
    except (AttributeError, ValueError):
        return None


def clean_zones(data):
    """Zones posted by the page -> {name: [[x, y], ...]} with fractions 0..1, or raise ValueError."""
    zones = data.get("zones") if isinstance(data, dict) else None
    if not isinstance(zones, dict) or len(zones) > MAX_ZONES:
        raise ValueError("expected {\"zones\": {name: [[x, y], ...]}}")
    out = {}
    for name, points in zones.items():
        name = " ".join(str(name).split())[:60]
        if not name or not isinstance(points, list) or not 3 <= len(points) <= 50:
            raise ValueError(f"zone {name!r} needs a name and 3 or more corners")
        out[name] = [[min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))] for x, y in points]
    return out


class App:
    """Everything the handlers need, in one place."""

    def __init__(self, memory, ui_state, cam, det, answerer, listener=None, voice=None, knob=None):
        self.memory, self.ui_state, self.cam, self.det = memory, ui_state, cam, det
        self.answerer, self.listener, self.voice, self.knob = answerer, listener, voice, knob
        self.gifs = OrderedDict()   # (moment, highlight, box, kind) -> gif bytes; a few recent ones
        self.gif_lock = threading.Lock()
        self.warming = set()        # gifs being made ahead of time, so an answer's replay starts at once

    # ---- JSON for the page ----
    def state(self):
        mem = self.memory
        page = mem.page_state()
        self._warm_up_gifs(page["events"][-3:])
        describer = mem.describer
        listener = self.listener
        mic = ("listening" if listener and listener.listening else
               "Talk button only" if listener and listener.enabled else "off")
        page.update({
            "now": time.time(),
            "status": {
                "gemini": bool(describer and describer.enabled),
                "voice": bool(self.voice and getattr(self.voice, "enabled", False)),
                "mic": mic,
                "talk": bool(listener and listener.enabled),
                "live_ms": round(self.det.item_ms),
            },
            "zones": self.zones(),
            "jump": self.ui_state.take_web_jump(),
            "answer": self.ui_state.answer,
            "knob": self.knob.position(len(page["keyframes"])) if self.knob and self.knob.ok else None,
        })
        return page

    # ---- pictures ----
    def live_jpeg(self, frame):
        mem = self.memory
        ruled_out = [(*s["veto_box"], name) for name, s in list(mem.items.items())
                     if s["veto_none"] and s["veto_box"] and time.time() < s["veto_until"]]
        boxes = ui.live_boxes(self.det.latest(), mem.known_items(), mem.gemini_only_items(), ruled_out)
        return encode_jpeg(ui.draw_detections(frame.copy(), boxes), quality=75)

    def keyframe_jpeg(self, index, highlight="", box=None):
        kf = self.memory.keyframe(index)
        return None if kf is None else encode_jpeg(ui.moment_picture(kf, highlight, box))

    def _gif_key(self, kf, highlight, box, kind):
        return (kf.index, highlight, tuple(box or ()), kind, len(kf.clip), kf.last_pass)

    def _warm_up_gifs(self, events):
        """Make the gifs of the latest events in the background: those are what answers replay."""
        for ev in events:
            kf = self.memory.keyframe(ev["keyframe"]) if ev["keyframe"] is not None else None
            if kf is None:
                continue
            key = self._gif_key(kf, ev["item"], ev["box"], ev["kind"])
            with self.gif_lock:
                if key in self.gifs or key in self.warming:
                    continue
                self.warming.add(key)

            def make(ev=ev, key=key):
                try:
                    self.keyframe_gif(ev["keyframe"], ev["item"], ev["box"], ev["kind"])
                except Exception as e:
                    print(f"[web] gif failed: {e}")
                finally:
                    with self.gif_lock:
                        self.warming.discard(key)

            threading.Thread(target=make, daemon=True).start()

    def keyframe_gif(self, index, highlight="", box=None, kind=""):
        kf = self.memory.keyframe(index)
        if kf is None:
            return None
        key = self._gif_key(kf, highlight, box, kind)
        with self.gif_lock:
            if key in self.gifs:
                self.gifs.move_to_end(key)
                return self.gifs[key]
        gif = ui.replay_gif(kf, highlight, box, kind)
        with self.gif_lock:
            self.gifs[key] = gif
            while len(self.gifs) > 24:
                self.gifs.popitem(last=False)
        return gif

    # ---- zones ----
    def zones(self):
        return {name: np.asarray(poly, dtype=float).round(4).tolist() for name, poly in self.memory.zones.items()}

    def save_zones(self, data):
        from zones import save_zones

        zones = clean_zones(data)
        save_zones({name: np.array(points, dtype=np.float32) for name, points in zones.items()})
        self.memory.reload_zones()
        print(f"[zones] saved: {', '.join(zones) or 'none'}")
        return zones


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal for Rewind's own messages
            pass

        def _send(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=200):
            self._send(json.dumps(obj).encode(), "application/json", status)

        def _body_json(self):
            size = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(size) or b"{}")

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            hl = query.get("hl", [""])[0]
            box = parse_box(query.get("box", [""])[0])
            try:
                if url.path in ("/", "/index.html"):
                    with open(os.path.join(STATIC, "index.html"), "rb") as f:
                        self._send(f.read(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self._json(app.state())
                elif url.path == "/api/zones":
                    self._json({"zones": app.zones()})
                elif url.path.startswith("/frame/") and url.path.endswith(".jpg"):
                    jpg = app.keyframe_jpeg(int(url.path[len("/frame/"):-len(".jpg")]), hl, box)
                    if jpg is None:
                        self._send(b"no such moment", "text/plain", 404)
                    else:
                        self._send(jpg, "image/jpeg")
                elif url.path.startswith("/clip/") and url.path.endswith(".gif"):
                    kind = query.get("kind", [""])[0]
                    gif = app.keyframe_gif(int(url.path[len("/clip/"):-len(".gif")]), hl, box, kind)
                    if gif is None:
                        self._send(b"no such moment", "text/plain", 404)
                    else:
                        self._send(gif, "image/gif")
                elif url.path == "/stream.mjpg":
                    self._stream()
                else:
                    self._send(b"not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser went away (closed tab, switched to a moment)
            except Exception as e:
                print(f"[web] {url.path} failed: {e}")
                try:
                    self._send(b"error", "text/plain", 500)
                except Exception:
                    pass

        def do_POST(self):
            try:
                if self.path == "/api/listen":
                    return self._listen()
                if self.path == "/api/talk":
                    app.ui_state.talk(bool(self._body_json().get("on")))
                    return self._json({"ok": True})
                if self.path == "/api/zones":
                    try:
                        return self._json({"zones": app.save_zones(self._body_json())})
                    except ValueError as e:
                        return self._json({"error": str(e)}, 400)
                if self.path != "/api/ask":
                    return self._send(b"not found", "text/plain", 404)
                question = str(self._body_json().get("question", "")).strip()
                if not question:
                    return self._json({"error": "empty question"}, 400)
                reply = app.answerer.ask(question)       # speaks on its own thread
                app.ui_state.take_web_jump()             # the reply carries the jump; don't repeat it on the poll
                self._json({"text": reply["text"], "jump": reply["jump"] and dict(reply["jump"], caption=reply["text"])})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                print(f"[web] {self.path} failed: {e}")
                self._json({"error": str(e)}, 500)

        def _listen(self):
            """Audio from the browser's microphone -> text -> the same answer path as typing."""
            size = int(self.headers.get("Content-Length", 0))
            try:
                if size > MAX_AUDIO_BYTES:
                    left = min(size, 8 * MAX_AUDIO_BYTES)  # read and drop the upload so the browser sees our reply
                    while left > 0:
                        left -= len(self.rfile.read(min(65536, left)))
                    return self._json({"error": "That recording is too long."}, 413)
                audio = self.rfile.read(size)
                if len(audio) < 1000:
                    return self._json({"error": "I didn't hear anything. Hold the button while you speak."}, 400)
                if app.listener is None or not app.listener.enabled:
                    return self._json({"error": "Speech-to-text is off (no ElevenLabs key). Type the question instead."}, 503)
                try:
                    heard = app.listener.transcribe_upload(audio, self.headers.get("Content-Type", "audio/webm")).strip()
                except Exception as e:
                    print(f"[listen] speech-to-text failed: {type(e).__name__}")
                    return self._json({"error": "I couldn't turn that into text. You can type the question instead."}, 502)
                if not heard:
                    return self._json({"error": "I didn't catch that. Try again a bit closer to the mic."}, 422)
                print(f"[listen] heard (Talk button): {heard}")
                reply = app.answerer.ask(heard, heard=heard, source="talk")
                app.ui_state.take_web_jump()
                self._json({"heard": heard, "text": reply["text"],
                            "jump": reply["jump"] and dict(reply["jump"], caption=reply["text"])})
            finally:
                app.ui_state.talk(False)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_id, gap = -1, 1.0 / STREAM_FPS
            while True:
                frame_id, frame = app.cam.latest()
                if frame is None or frame_id == last_id:
                    if app.cam.ended:
                        return
                    time.sleep(0.01)
                    continue
                last_id = frame_id
                jpg = app.live_jpeg(frame)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(gap)

    return Handler


def start_server(app, port=8000):
    """Start the web server on a background thread. Returns (server, url)."""
    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
