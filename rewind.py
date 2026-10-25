"""
Rewind - Ctrl+Z for the physical world.

  python rewind.py                 # default webcam; opens http://127.0.0.1:8000 in your browser
  python rewind.py --source 1      # second camera (e.g. a phone webcam)

The page shows the live camera, a slider back through the table's history (each moment
replays as a little rewind gif), and a box to ask "where are my keys?". You can also just
ask out loud (the microphone is always listening for questions), hold Talk on the page,
or type a question in this terminal.

How the pieces fit:
  phase1_camera_gate.py  camera + fast item/person detection (change-gated)
  memory.py              what was put down where, when it left, the saved moments (+ Gemini via describe.py)
  zones.py               named areas ("on the kitchen table"), drawn on the page
  answer.py              question -> item -> answer + the moment to replay
  speak.py / listen.py   ElevenLabs voice out / speech in
  web.py + static/       the page;  ui.py  drawing and the gif
  store.py, knob.py      optional extras (Tiger Data copy, a physical rewind knob)
"""

import argparse
import threading
import time
import webbrowser


def terminal_questions(answerer):
    """Read questions from the terminal on their own thread (input() blocks)."""
    while True:
        try:
            q = input("ask> ").strip()
        except (EOFError, OSError):
            return
        if q:
            answerer.ask(q)


class PrintVoice:
    """--no-voice: answers are printed instead of spoken."""
    enabled = False

    @staticmethod
    def say(text):
        print(f"[voice] {text}")

    @staticmethod
    def busy():
        return False


def main():
    ap = argparse.ArgumentParser(description="Rewind")
    ap.add_argument("--source", default="0", help="camera index (0, 1, ...) or a video file")
    ap.add_argument("--port", type=int, default=8000, help="port for the web page")
    ap.add_argument("--no-browser", action="store_true", help="don't open the web page automatically")
    ap.add_argument("--no-gemini", action="store_true", help="don't send pictures to Gemini (offline / save quota)")
    ap.add_argument("--no-voice", action="store_true", help="print answers instead of speaking them")
    ap.add_argument("--quiet", action="store_true", help="only speak when asked, not on every put-down")
    ap.add_argument("--no-listen", action="store_true", help="no always-on microphone (the page's Talk button still works)")
    ap.add_argument("--wake", action="store_true", help='noisy room: only answer spoken questions that start with "Rewind"')
    ap.add_argument("--knob", default="", help="serial port of the rewind knob, e.g. COM3 or /dev/cu.usbmodem1101")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = until Ctrl+C)")
    args = ap.parse_args()

    from answer import Answerer
    from knob import Knob
    from listen import Listener
    from memory import Memory, plural
    from phase1_camera_gate import IMGSZ, ITEMS, MODEL, PERSON_MODEL, Camera, Detector, pick_device
    from speak import Voice
    from store import Store
    from ui import UIState
    from web import App, start_server

    device = pick_device()
    print(f"[rewind] device={device}  imgsz={IMGSZ}  items={MODEL}  people={PERSON_MODEL}")
    voice = PrintVoice() if args.no_voice else Voice()

    def announce(item, phrase):  # Gemini confirmed a put-down
        voice.say(f"Got it. Your {item} {'are' if plural(item) else 'is'} {phrase}.")

    memory = Memory(use_gemini=not args.no_gemini, on_confirmed=None if args.quiet else announce,
                    items=ITEMS, store=Store())
    ui_state = UIState()
    answerer = Answerer(memory, voice, ui_state, ITEMS, wake_word=args.wake)
    # The always-on mic ignores the room while Rewind talks, or while someone holds the page's Talk button.
    listener = Listener(answerer.heard, is_busy=lambda: voice.busy() or ui_state.talking())

    try:
        cam = Camera(args.source)
    except RuntimeError as e:
        print(f"[rewind] {e}")
        return
    print("[rewind] loading models (first run downloads them, give it a minute)")
    det = Detector(cam, device, on_pass=memory.update, on_settle=memory.settled)
    cam.start()  # only start reading once the models are ready
    if not args.no_listen:
        listener.start()
    knob = Knob(args.knob) if args.knob else None

    app = App(memory, ui_state, cam, det, answerer, listener, voice, knob)
    try:
        server, url = start_server(app, args.port)
    except OSError as e:
        print(f"[rewind] could not start the web page on port {args.port}: {e}\n"
              f"[rewind] try --port {args.port + 1}")
        det.stop()
        cam.stop()
        return
    print(f"[rewind] open {url}   (Ctrl+C here to quit; you can also type questions below)")
    if not args.no_browser:
        webbrowser.open(url)
    threading.Thread(target=terminal_questions, args=(answerer,), daemon=True).start()

    started = time.perf_counter()
    try:
        while not cam.ended:
            time.sleep(0.2)
            if args.seconds and time.perf_counter() - started > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("[rewind] stopping")
        server.shutdown()
        det.stop()
        cam.stop()
        if knob:
            knob.stop()


if __name__ == "__main__":
    main()
