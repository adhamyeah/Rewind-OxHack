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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")            # questions ("which item is this about?")
GEMINI_SCAN_MODEL = os.getenv("GEMINI_SCAN_MODEL", "gemini-3.8-flash")  # whole-scene object finding at each keyframe
GEMINI_CHECK_MODEL = os.getenv("GEMINI_CHECK_MODEL", "gemini-3.5-flash-lite")  # quick "is that really a pen?" checks
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")
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
# What the fast YOLO model looks for. Not "remote control" or "wallet": YOLO kept putting those names
# on water bottles and phones (a flat black phone looks like a wallet). Gemini still looks for every
# item in ITEMS, so a real remote or wallet is still found.
YOLO_SKIP = {"remote control", "wallet"}
YOLO_ITEMS = [item for item in ITEMS if item not in YOLO_SKIP]
SAME_OBJECT_IOU = 0.7       # two YOLO labels on (almost) the same box = one object; keep the stronger label

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
MAX_MOTION_WAIT = 5.0       # a keyframe fires after this long even if the scene never goes still (a face keeps moving)
SETTLE_SECONDS = 1.0        # scene must stay still this long before a keyframe fires

# --- memory ---
MISSES_TO_LEAVE = 2         # keyframes an item must be missing from before it is "gone"
NEAR_FRACTION = 0.3         # items closer than this (fraction of frame width, centre to centre) are "next to" each other
HELD_OVERLAP = 0.7          # item box this much inside a person's box = held/worn, not on the table
HELD_UPPER_FRACTION = 0.85  # ...and its centre in the top 85% of that person (face, chest, hands); lower = table
MOVE_FRACTION = 0.05        # centre must shift by more than 5% of frame width to count as "moved"

# --- Gemini ---
# Scan (one per keyframe). People's heads are blurred out before the photo is sent (see
# PERSON_BLUR_FRACTION), so Gemini never sees faces, but hands and the table stay visible: items
# in someone's hand are found like any other and memory marks them "with you".
PERSON_BLUR_FRACTION = 0.4    # blur the top 40% of each person box (head and shoulders), not the whole body
GEMINI_PROMPT = (
    "This is a still photo from a camera in someone's home. Blurred areas are people's faces. "
    "Find each of these objects that is clearly visible, including ones held in a hand: "
    + ", ".join(f'"{item}"' for item in ITEMS)
    + ". "
    "Return ONLY a JSON list, no other text. Each entry: "
    '{"label": one of the names above exactly, "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000}. '
    "Return [] if none of them are visible."
)
# Check: YOLO saw an item that Gemini's scan didn't list. Gemini looks at the photo with that item
# in an orange box and says whether it's real (NONE = not that item, HELD = in someone's hand).
GEMINI_CHECK_PROMPT = (
    "This photo is from a camera in someone's home. Blurred areas are people. A detector thinks "
    "there is a {item} inside the orange box. Reply with exactly one word:\n"
    "NONE - if the orange box does not contain a {item}.\n"
    "HELD - if the {item} is in someone's hand or being worn.\n"
    "YES - if a {item} is in the orange box, put down somewhere."
)
MAX_CHECKS_PER_KEYFRAME = 3   # Gemini checks per keyframe at most; any left over are checked on the next one
VETO_NONE_SECONDS = 120       # Gemini said "that's not a pen": ignore YOLO's pen at that spot for 2 minutes
VETO_HELD_SECONDS = 8         # Gemini said "it's in a hand": look again after 8 seconds
CHECK_MOVE_FRACTION = 0.08    # a verdict holds until the box moves more than 8% of the frame width
GEMINI_THINKING = os.getenv("GEMINI_THINKING", "low")  # Gemini 3 'thinking' level for the scan; default thinking took up to 16 s, 'low' ~2 s. "" = model default
GEMINI_MIN_GAP = 1.0        # seconds between live Gemini calls (batch scripts add their own ~4 s)
ASK_WAIT_SECONDS = 8.0      # a question waits up to this long for memory to catch up with the table
GEMINI_COOLDOWN = 600.0     # after Gemini says the QUOTA is used up, skip Gemini this long and use YOLO
GEMINI_RETRY_SECONDS = 15.0 # after any other failure (503 overloaded, network blip), try Gemini again after this long

# --- files ---
KEYFRAME_DIR = "keyframes"
ANSWER_WAV = "answer.wav"
WINDOW = "Rewind (q to quit)"
