"""
Web UI on http://127.0.0.1:<port> using only Python's built-in HTTP server (no extra packages).

  /                  the page (static/index.html)
  /stream.mjpg       the LIVE view as a motion-JPEG stream (an <img> tag plays it)
  /frame/<i>.jpg     keyframe i with its boxes; ?hl=keys adds the yellow highlight on that item
  /api/state         JSON the page polls: keyframes, items on the table, events, status, pending jump
  POST /api/ask      {"question": "..."} -> {"text": ..., "jump": {...} or null}
  POST /api/listen   raw audio recorded in the browser -> ElevenLabs speech-to-text -> same as /api/ask,
                     plus "heard": what it understood

Everything slow (Gemini, voice) already runs on other threads, so these handlers only read
memory and encode JPEGs. Bound to 127.0.0.1 so only this computer can open it.
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

import ui
from listen import transcribe as default_transcribe

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STREAM_FPS = 20
MAX_AUDIO_BYTES = 5_000_000   # ~ several minutes of speech; a question is a few seconds


def encode_jpeg(image, quality=82):
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode image")
    return jpg.tobytes()


class App:
    """Everything the handlers need, in one place."""

    def __init__(self, memory, ui_state, cam, det, worker, answerer, gemini_on, transcribe=default_transcribe):
        self.memory, self.ui_state, self.cam = memory, ui_state, cam
        self.det, self.worker, self.answerer, self.gemini_on = det, worker, answerer, gemini_on
        self.transcribe = transcribe

    def state(self):
        mem = self.memory
        with mem.lock:
            keyframes = [{"i": k.index, "t": k.time,
                          "time": time.strftime("%H:%M:%S", time.localtime(k.time)),
                          "labels": sorted({d[4] for d in k.detections if d[4] != "person"})}
                         for k in mem.keyframes]
            events = [{"t": e.time, "label": e.label, "kind": e.kind, "keyframe": e.keyframe}
                      for e in mem.events[-10:]]
        return {
            "now": time.time(),
            "keyframes": keyframes,
            "present": mem.present_items(),
            "events": events,
            "status": {"memory": self.worker.status, "last": self.worker.last_source,
                       "live_ms": round(self.det.item_ms), "gemini": self.gemini_on},
            "jump": self.ui_state.take_web_jump(),
        }

    def keyframe_jpeg(self, index, highlight=""):
        with self.memory.lock:
            if not 0 <= index < len(self.memory.keyframes):
                return None
            kf = self.memory.keyframes[index]
            image, dets = kf.image.copy(), list(kf.detections)
        ui.draw_detections(image, dets)
        for d in dets:
            if highlight and d[4] == highlight:
                ui.draw_highlight(image, d[:4], highlight)
        return encode_jpeg(image)

    def live_jpeg(self, frame):
        last = self.memory.keyframes[-1].detections if self.memory.keyframes else None
        boxes = ui.pick_live_boxes(self.det.latest(), last, self.det.dirty, self.worker.busy)
        return encode_jpeg(ui.draw_detections(frame.copy(), boxes), quality=75)


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

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    with open(os.path.join(STATIC, "index.html"), "rb") as f:
                        self._send(f.read(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self._json(app.state())
                elif url.path.startswith("/frame/") and url.path.endswith(".jpg"):
                    index = int(url.path[len("/frame/"):-len(".jpg")])
                    hl = parse_qs(url.query).get("hl", [""])[0]
                    jpg = app.keyframe_jpeg(index, hl)
                    if jpg is None:
                        self._send(b"no such keyframe", "text/plain", 404)
                    else:
                        self._send(jpg, "image/jpeg")
                elif url.path == "/stream.mjpg":
                    self._stream()
                else:
                    self._send(b"not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser went away (closed tab, switched to a keyframe)
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
                if self.path != "/api/ask":
                    return self._send(b"not found", "text/plain", 404)
                size = int(self.headers.get("Content-Length", 0))
                question = str(json.loads(self.rfile.read(size) or b"{}").get("question", "")).strip()
                if not question:
                    return self._json({"error": "empty question"}, 400)
                text = app.answerer.ask(question)  # waits briefly for memory, speaks on its own thread
                self._json({"text": text, "jump": app.ui_state.take_web_jump()})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                print(f"[web] ask failed: {e}")
                self._json({"error": str(e)}, 500)

        def _listen(self):
            """Audio from the browser's microphone -> text -> the same answer path as typing."""
            size = int(self.headers.get("Content-Length", 0))
            if size > MAX_AUDIO_BYTES:
                left = min(size, 8 * MAX_AUDIO_BYTES)  # read and drop the upload so the browser sees our reply
                while left > 0:
                    left -= len(self.rfile.read(min(65536, left)))
                return self._json({"error": "That recording is too long."}, 413)
            audio = self.rfile.read(size)
            if len(audio) < 1000:
                return self._json({"error": "I didn't hear anything. Hold the button while you speak."}, 400)
            try:
                heard = app.transcribe(audio, self.headers.get("Content-Type", "audio/webm")).strip()
            except Exception as e:
                print(f"[listen] speech-to-text failed: {e}")
                return self._json({"error": "I couldn't turn that into text. You can type the question instead."}, 502)
            if not heard:
                return self._json({"error": "I didn't catch that. Try again a bit closer to the mic."}, 422)
            print(f"[listen] heard: {heard}")
            text = app.answerer.ask(heard)
            self._json({"heard": heard, "text": text, "jump": app.ui_state.take_web_jump()})

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
    import threading

    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
