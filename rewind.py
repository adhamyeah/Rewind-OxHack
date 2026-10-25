"""
Rewind - Ctrl+Z for the physical world.

Default: a web page on http://127.0.0.1:8000 with the live view, a slider through the table's
history (0..n-1 = keyframes, n = LIVE) and a box to ask "where are my keys?".
`--window` uses the old OpenCV window instead (trackbar, keys , . l q).
You can also type questions in the terminal either way.
"""

import argparse
import threading
import time
import webbrowser

import cv2

import config
import ui
from answer import Answerer
from camera import Camera
from keyframe_worker import KeyframeWorker
from knob import Knob
from live_detector import LiveDetector, pick_device
from memory import Memory
from store import Store
from web import App, start_server


def terminal_questions(answerer):
    """Read questions from the terminal on their own thread (input() blocks)."""
    while True:
        try:
            q = input("ask> ").strip()
        except EOFError:
            return
        if q:
            answerer.ask(q)


def main():
    ap = argparse.ArgumentParser(description="Rewind")
    ap.add_argument("--source", default="0", help="camera index or a video file")
    ap.add_argument("--no-gemini", action="store_true", help="use YOLO boxes for memory, skip Gemini")
    ap.add_argument("--knob", default="", help="serial port of the potentiometer, e.g. /dev/cu.usbmodem1101")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = until q)")
    ap.add_argument("--window", action="store_true", help="use the old OpenCV window instead of the web page")
    ap.add_argument("--port", type=int, default=8000, help="port for the web page")
    ap.add_argument("--no-browser", action="store_true", help="don't open the web page automatically")
    args = ap.parse_args()

    use_gemini = not args.no_gemini and bool(config.GEMINI_API_KEY)
    if not args.no_gemini and not use_gemini:
        print("[rewind] no GEMINI_API_KEY: memory will use YOLO boxes")
    print("[rewind] loading models (first run downloads them, give it a minute)")

    memory = Memory()
    ui_state = ui.UIState()
    store = Store()
    worker = KeyframeWorker(memory, store, use_gemini=use_gemini,
                            on_pending=lambda: det.force_keyframe())  # det is created just below
    try:
        cam = Camera(args.source)
    except RuntimeError as e:
        print(f"[rewind] {e}")
        worker.stop()
        return
    det = LiveDetector(cam, pick_device(), on_keyframe=worker.submit)
    cam.start()
    det.start()
    knob = Knob(args.knob) if args.knob else None

    def memory_ready():
        # settled scene, no keyframe being processed, and no item waiting to be confirmed gone
        return not det.dirty and not worker.busy and not memory.pending_misses()

    answerer = Answerer(memory, ui_state, ready=memory_ready)
    threading.Thread(target=terminal_questions, args=(answerer,), daemon=True).start()

    if args.window:
        window_loop(args, memory, ui_state, worker, cam, det, knob)
    else:
        web_loop(args, memory, ui_state, worker, cam, det, answerer, use_gemini)


def web_loop(args, memory, ui_state, worker, cam, det, answerer, use_gemini):
    app = App(memory, ui_state, cam, det, worker, answerer, use_gemini)
    try:
        server, url = start_server(app, args.port)
    except OSError as e:
        print(f"[rewind] could not start the web page on port {args.port}: {e}\n"
              f"[rewind] try --port {args.port + 1}, or use --window")
        det.stop(), cam.stop(), worker.stop()
        return
    print(f"[rewind] open {url}   (Ctrl+C to quit)")
    if not args.no_browser:
        webbrowser.open(url)
    started = time.perf_counter()
    try:
        while not cam.ended:
            time.sleep(0.2)
            if args.seconds and time.perf_counter() - started > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        det.stop()
        cam.stop()
        worker.stop()


def window_loop(args, memory, ui_state, worker, cam, det, knob):
    cv2.namedWindow(config.WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("time", config.WINDOW, 0, 1, lambda v: None)
    position = 0          # trackbar position; == len(keyframes) means LIVE
    following_live = True
    started = time.perf_counter()
    try:
        while True:
            n = len(memory.keyframes)
            cv2.setTrackbarMax("time", config.WINDOW, max(1, n))

            jump = ui_state.take_jump()
            if jump is not None:
                position, following_live = jump, False
                cv2.setTrackbarPos("time", config.WINDOW, position)
            else:
                tb = cv2.getTrackbarPos("time", config.WINDOW)
                kpos = knob.position(n) if knob and knob.ok else None
                if kpos is not None and kpos != position:
                    position, following_live = kpos, kpos >= n
                    cv2.setTrackbarPos("time", config.WINDOW, position)
                elif tb != position:
                    position, following_live = tb, tb >= n
                    ui_state.clear()
            if following_live:
                position = n
                cv2.setTrackbarPos("time", config.WINDOW, n)

            _, live = cam.latest()
            if live is None:
                if cam.ended:
                    break
                time.sleep(0.01)
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            if position >= n or position >= len(memory.keyframes):
                last = memory.keyframes[-1].detections if memory.keyframes else None
                boxes = ui.pick_live_boxes(det.latest(), last, det.dirty, worker.busy)
                view = ui.draw_detections(live.copy(), boxes)
                mode = "LIVE"
            else:
                kf = memory.keyframes[position]
                view = ui.draw_detections(kf.image.copy(), kf.detections)
                hl = ui_state.highlight
                if hl:
                    ui.draw_highlight(view, hl[0], hl[1])
                mode = f"REWIND {position + 1}/{n}  {time.strftime('%H:%M:%S', time.localtime(kf.time))}"
            status = f"{mode} | memory: {worker.status} (last: {worker.last_source}) | live {det.item_ms:.0f} ms | keyframes {n}"
            ui.draw_text_bar(view, status)
            if ui_state.caption:
                ui.draw_text_bar(view, ui_state.caption, top=False, colour=(0, 255, 255))
            cv2.imshow(config.WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(config.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(","):
                position, following_live = max(0, position - 1), False
                cv2.setTrackbarPos("time", config.WINDOW, position)
            elif key == ord("."):
                position = min(n, position + 1)
                following_live = position >= n
                cv2.setTrackbarPos("time", config.WINDOW, position)
            elif key in (ord("l"), ord("L")):
                following_live = True
            if args.seconds and time.perf_counter() - started > args.seconds:
                break
    finally:
        det.stop()
        cam.stop()
        worker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
