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
PLURAL_ITEMS = {"keys", "glasses", "headphones", "scissors", "sunglasses", "earphones"}

# --- open vocabulary (branch 007): Gemini can name anything; ITEMS are just the starting list ---
OPEN_VOCAB = True           # False = only ITEMS are ever detected (the original behaviour)
MAX_LEARNED = 40            # names learned from Gemini that YOLO-World also looks for (most recent kept)
# names Gemini may use that are never "items": people (the person model boxes them), body parts,
# the room's structure and names too vague to mean anything. Everything else IS named and tracked.
NOT_ITEMS = {"person", "people", "man", "woman", "boy", "girl", "child", "face", "head", "hair",
             "eye", "eyes", "nose", "mouth", "beard", "hand", "hands", "arm", "finger", "body",
             "wall", "floor", "ceiling", "window", "background", "shadow", "room", "sky",
             "object", "thing", "item", "stuff", "unknown"}

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
MAX_KEYFRAME_WAIT = 5.0     # ...but if it keeps changing (a person in view, a thing held up), fire anyway after this

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
# open vocabulary: EVERY object in the frame, each named for what it really is. Measured on
# real keyframes: this found 15 objects in 3 frames where "only loseable things" found 6.
# No list of known names here: given one, Gemini squeezed things into it and even invented
# "keys" and "glasses" that weren't there. vocab.canonical() keeps names consistent instead.
GEMINI_OPEN_PROMPT = (
    "List EVERY object visible in this photo, large and small, near and far: furniture, devices, "
    "bags, bottles, cups, papers, signs, posters, decorations, clothing, accessories, anything a "
    "person is holding, and each person (label them 'person'). Do not list walls, floor, ceiling "
    "or the sky. Be exhaustive: if you can see it and name it, include it, up to 40 objects. "
    "Name each one specifically in 1-3 common lowercase words, by what it actually is "
    "(for example 'black puffer jacket' -> 'jacket', 'smartphone' -> 'phone'). "
    'Return JSON only: a list like [{"label": "phone", "box_2d": [ymin, xmin, ymax, xmax]}] '
    "with coordinates normalised 0-1000 (y first)."
)
# forces Gemini's reply into exactly that shape
GEMINI_OPEN_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "label": {"type": "string"}, "box_2d": {"type": "array", "items": {"type": "integer"}}},
    "required": ["label", "box_2d"]}}
GEMINI_MIN_GAP = 1.0        # seconds between live Gemini calls (batch scripts add their own ~4 s)
ASK_WAIT_SECONDS = 8.0      # a question waits up to this long for memory to catch up with the table
GEMINI_COOLDOWN = 600.0     # after a quota error, skip Gemini this long and use YOLO
GEMINI_RETRY_SECONDS = 30.0 # after any other Gemini failure (network, busy server), try again this soon
GEMINI_BOX_SECONDS = 20.0   # open vocabulary: the live view keeps showing Gemini's boxes this long (a look can take 15 s)

# --- files ---
KEYFRAME_DIR = "keyframes"
ANSWER_WAV = "answer.wav"
WINDOW = "Rewind (q to quit)"
