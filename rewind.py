"""
Rewind - Ctrl+Z for the physical world.

Trackbar 0..n-1 = keyframes (the table's history), n = LIVE.
Type a question in the terminal ("where are my keys?"); the window jumps to that moment.
Keys: , = back one keyframe, . = forward, l = live, SPACE = look now (send this moment to Gemini), q = quit.
"""

import argparse
import socket
import threading
import time

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


def terminal_questions(answerer):
    """Read questions from the terminal on their own thread (input() blocks)."""
    while True:
        try:
            q = input("ask> ").strip()
        except EOFError:
            return
        if q:
            answerer.ask(q)


def claim_single_instance(port=47_311):
    """Only one Rewind at a time: a second one would get black frames from the busy camera.

    Holding a local port is the lock; Windows and macOS free it when the process exits,
    even if it crashes. Returns the socket (keep it alive) or None if Rewind already runs.
    """
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):       # Windows: refuse to share the port
        lock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        lock.bind(("127.0.0.1", port))
    except OSError:
        lock.close()
        return None
    return lock


def main():
    instance_lock = claim_single_instance()
    if instance_lock is None:
        raise SystemExit("[rewind] Rewind is already running in another window or terminal.\n"
                         "         Close that one first (press q in its window), then start again.")
    ap = argparse.ArgumentParser(description="Rewind")
    ap.add_argument("--source", default="0", help="camera index or a video file")
    ap.add_argument("--no-gemini", action="store_true", help="use YOLO boxes for memory, skip Gemini")
    ap.add_argument("--knob", default="", help="serial port of the potentiometer, e.g. /dev/cu.usbmodem1101")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = until q)")
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
        raise SystemExit(f"[rewind] {e}")
    det = LiveDetector(cam, pick_device(), on_keyframe=worker.submit)
    cam.start()
    det.start()
    knob = Knob(args.knob) if args.knob else None

    def memory_ready():
        # settled scene, no keyframe being processed, and no item waiting to be confirmed gone
        return not det.dirty and not worker.busy and not memory.pending_misses()

    answerer = Answerer(memory, ui_state, ready=memory_ready)
    threading.Thread(target=terminal_questions, args=(answerer,), daemon=True).start()

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
                if config.OPEN_VOCAB:   # Gemini names the objects: always show its latest boxes
                    boxes = ui.pick_live_boxes_open(det.latest(), memory.keyframes[-1] if memory.keyframes else None,
                                                    time.time())
                else:
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
            elif key == ord(" "):       # show Gemini what's in view right now
                det.request_look()
                print("[rewind] looking now...")
            if args.seconds and time.perf_counter() - started > args.seconds:
                break
    finally:
        det.stop()
        cam.stop()
        worker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
