"""
Every tunable setting for Rewind lives here.
Secrets come from a git-ignored .env file (see .env.example).
"""

import os

try:  # python-dotenv is optional: without it, real environment variables still work
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# --- secrets / services (all optional: each one fails gracefully) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
TIGER_DATABASE_URL = os.getenv("TIGER_DATABASE_URL", "")

# --- what to look for (from phase1: works on real glasses) ---
ITEMS = ["keys", "glasses", "wallet", "phone", "remote control", "mug", "headphones", "pen", "water bottle"]
# Other ways people say each item. normalise_label() maps them back to ITEMS.
SYNONYMS = {
    "keys": ["keys", "key", "car keys", "house keys", "keyring", "key ring"],
    "glasses": ["glasses", "specs", "spectacles", "sunglasses", "eyeglasses"],
    "wallet": ["wallet", "purse"],
    "phone": ["phone", "mobile", "smartphone", "iphone", "cell phone"],
    "remote control": ["remote control", "remote", "clicker"],
    "mug": ["mug", "cup", "coffee cup"],
    "headphones": ["headphones", "airpods", "earbuds", "earphones"],
    "pen": ["pen", "pencil"],
    "water bottle": ["water bottle", "bottle", "flask", "water flask"],
}
# Items whose name is plural, so answers say "were" instead of "was".
PLURAL_ITEMS = {"keys", "glasses", "headphones"}

# --- item model (measured on 128 labelled photos, see phase1_camera_gate.py) ---
MODEL = "yolov8m-worldv2.pt"
IMGSZ = 640
CONFIDENCE = 0.15

# --- people (separate model) ---
USE_PERSON_MODEL = True
PERSON_MODEL = "yolo11n.pt"
PERSON_CONFIDENCE = 0.5
MIN_PERSON_AREA = 0.03

# --- camera ---
CAMERA_W, CAMERA_H = 640, 480
USE_MAC_GPU = False

# --- live change gate (from phase1) ---
GATE_W, GATE_H = 480, 360
BLUR = 9
DIFF_THRESHOLD = 5.0
DRIFT = 0.1
RECHECK_SECONDS = 2.0
PASSES_PER_SECOND = 5

# --- keyframes: "the scene moved, then went still" ---
MOTION_PIXEL_DIFF = 25      # a pixel counts as moving if it changed by more than this (0-255)
MOTION_FRACTION = 0.01      # scene is "moving" if more than 1% of pixels moved
SETTLE_SECONDS = 1.0        # scene must stay still this long before a keyframe fires

# --- memory ---
MISSES_TO_LEAVE = 2         # keyframes an item must be missing from before it is "gone"
NEAR_FRACTION = 0.3         # items closer than this (fraction of frame width, centre to centre) are "next to" each other
MOVE_FRACTION = 0.05        # centre must shift by more than 5% of frame width to count as "moved"

# --- Gemini ---
GEMINI_PROMPT = (
    "You are looking at an overhead photo of a table. Find these items if present: "
    + ", ".join(ITEMS)
    + ". Return JSON only: a list of objects like "
    '{"label": "keys", "box_2d": [ymin, xmin, ymax, xmax]} '
    "with coordinates normalised 0-1000 (y first). Use only the labels listed. "
    "Return [] if none are visible."
)
GEMINI_MIN_GAP = 1.0        # seconds between live Gemini calls (batch scripts add their own ~4 s)
ASK_WAIT_SECONDS = 8.0      # a question waits up to this long for memory to catch up with the table
GEMINI_COOLDOWN = 600.0     # after a Gemini failure (e.g. quota), skip Gemini this long and use YOLO

# --- files ---
KEYFRAME_DIR = "keyframes"
ANSWER_WAV = "answer.wav"
WINDOW = "Rewind (q to quit)"
