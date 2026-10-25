"""
Secrets and service settings, read from the git-ignored .env file (see .env.example).

Everything here is optional: with a key missing, that part of Rewind turns itself off and the
rest keeps working. The detection, memory, voice and microphone settings live at the top of
their own files (phase1_camera_gate.py, memory.py, speak.py, listen.py, describe.py).
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path=os.path.join(HERE, ".env")):
    """Put the .env values into the environment (python-dotenv if installed, else a simple reader)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
        return
    except ImportError:
        pass
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                name, sep, value = line.strip().partition("=")
                if sep and name.strip() and not name.startswith("#"):
                    os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


load_env()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")      # empty = speak.py's default voice
TIGER_DATABASE_URL = os.getenv("TIGER_DATABASE_URL", "")        # empty = no Tiger Data copy
