"""
Checks that this machine is ready to run Rewind. Prints a tick or cross for each check,
with a hint on how to fix failures. Never crashes.

  python doctor.py              # local checks only
  python doctor.py --network    # also makes one real Gemini and one real ElevenLabs call
"""

import argparse
import importlib
import sys

OK, BAD = "✔", "✘"
failures = 0


def check(name, fn, hint):
    """Run fn(); it returns a detail string (or None). Any exception counts as a failure."""
    global failures
    try:
        detail = fn()
        print(f" {OK} {name}" + (f"  ({detail})" if detail else ""))
    except Exception as e:
        failures += 1
        print(f" {BAD} {name}: {str(e)[:120]}\n      fix: {hint}")


def mask(value):
    return "(not set)" if not value else value[:3] + "..." + value[-2:] if len(value) > 8 else "set"


def import_check(module, package):
    def run():
        m = importlib.import_module(module)
        return getattr(m, "__version__", "")
    check(f"import {module}", run, f"pip install {package}")


def need(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true", help="also test Gemini and ElevenLabs live")
    args = ap.parse_args()

    print("Python")
    check("Python 3.10+", lambda: (need(sys.version_info >= (3, 10), sys.version.split()[0]), sys.version.split()[0])[1],
          "install Python 3.10 or newer and recreate .venv")

    print("Packages")
    for module, package in [("cv2", "opencv-python"), ("ultralytics", "ultralytics"), ("google.genai", "google-genai"),
                            ("elevenlabs", "elevenlabs"), ("sounddevice", "sounddevice"), ("PIL", "pillow")]:
        import_check(module, package)
    print("Optional packages")
    for module, package, what in [("dotenv", "python-dotenv", ".env loading (Rewind has its own fallback)"),
                                  ("psycopg", "psycopg[binary]", "Tiger Data copy"),
                                  ("serial", "pyserial", "the rewind knob")]:
        try:
            importlib.import_module(module)
            print(f" {OK} import {module}")
        except Exception:
            print(f"   - {module} not installed ({what} stays off): pip install {package}")

    try:
        import config
    except Exception as e:
        print(f" {BAD} config.py failed to import: {e}")
        return 1

    print(".env keys (masked)")
    for name in ["GEMINI_API_KEY", "ELEVENLABS_API_KEY"]:
        value = getattr(config, name)
        check(f"{name} = {mask(value)}", lambda v=value: need(v, "empty"),
              f"copy .env.example to .env and set {name} (without it that part of Rewind stays off)")
    for name in ["ELEVENLABS_VOICE_ID", "TIGER_DATABASE_URL"]:
        print(f"   - {name} = {mask(getattr(config, name))} (optional)")

    print("Camera")
    for index in (0, 1):
        def cam(i=index):
            import cv2

            cap = cv2.VideoCapture(i)
            try:
                ok, frame = cap.read()
                need(cap.isOpened() and ok, "cannot read a frame")
                return f"{frame.shape[1]}x{frame.shape[0]}"
            finally:
                cap.release()
        check(f"camera index {index}", cam, "plug the camera in, allow camera access for your terminal, or try the other index")

    print("Microphone")
    def microphone():
        import sounddevice

        return sounddevice.query_devices(kind="input")["name"]
    check("default input device", microphone, "plug in / allow a microphone (or run rewind.py --no-listen and use the Talk button)")

    print("YOLO weights")
    def weights():
        from ultralytics import YOLO, YOLOWorld

        import phase1_camera_gate as gate

        YOLOWorld(gate.MODEL)
        YOLO(gate.PERSON_MODEL)
        return f"{gate.MODEL} + {gate.PERSON_MODEL}"
    check("weights load", weights, "git pull (weights are committed) or see README 'Weights'")

    print("Services")
    if config.TIGER_DATABASE_URL:
        def tiger():
            import psycopg

            with psycopg.connect(config.TIGER_DATABASE_URL, connect_timeout=5) as c:
                c.execute("SELECT 1")
        check("Tiger Data connect", tiger, "check TIGER_DATABASE_URL (copy it from the Tiger Cloud console)")
    else:
        print("   - Tiger Data: no URL set, skipped")

    if args.network:
        def gemini():
            import describe

            reply = describe.Gemini(config.GEMINI_API_KEY).ask_text("Reply with the word OK.")
            need(reply, "empty reply")
            return f"{describe.MODEL}: {reply.strip()[:20]}"
        check("Gemini call", gemini, "check GEMINI_API_KEY (pip install -U google-genai for new 'AQ.' keys), or wait if rate-limited")

        def eleven():
            import speak

            voice = speak.Voice()
            need(voice.enabled, "voice is off")
            need(len(voice.synthesize("Rewind is ready.")) > 1000, "no audio returned")
        check("ElevenLabs call", eleven, "check ELEVENLABS_API_KEY (and ELEVENLABS_VOICE_ID if you set one)")
    else:
        print("   - Gemini / ElevenLabs live calls skipped (add --network)")

    print(f"\n{'All good.' if not failures else f'{failures} check(s) need attention.'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # the doctor itself must never crash
        print(f"doctor hit an unexpected error: {e}")
