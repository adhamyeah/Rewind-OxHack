"""
Gemini as the memory detector: open-vocabulary, so it finds items YOLO-World misses.

Two jobs (both from Adham's detection branch):
  SCAN   gemini_detect(): one look at the whole settled scene, every item with a box.
  CHECK  gemini_check():  YOLO saw an item the scan didn't list; Gemini looks at it in an
         orange box and says YES / NONE (that isn't a pen) / HELD (it's in a hand).
The keyframe worker blurs people out before either call, so Gemini never sees faces.

Detections come back in the same tuple as everywhere else:
(x1, y1, x2, y2, label, confidence) in PIXELS. Gemini has no confidence, so we use 1.0.
"""

import json
import re
import time

import cv2

import config

_client = None
_last_call = 0.0

ORANGE = (0, 140, 255)  # BGR: the box Gemini is told to look inside


def normalise_label(text):
    """Map free text ("my specs", "car keys") to one of config.ITEMS, or None."""
    text = (text or "").lower()
    # Longest phrases first, so "remote control" wins over "remote" and "car keys" over "keys".
    pairs = sorted(((s, item) for item, syns in config.SYNONYMS.items() for s in syns),
                   key=lambda p: -len(p[0]))
    for phrase, item in pairs:
        # \b makes "pen" match "pen" but not "open"; the optional s allows plurals.
        if re.search(rf"\b{re.escape(phrase)}s?\b", text):
            return item
    return None


def blur_people(img, people):
    """Blur every person box in `img` (in place). Gemini never sees faces."""
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in people:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (31, 31), 0)
    return img


def _get_client():
    """Create the Gemini client once. Returns None if there is no key or no package."""
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            return None
        try:
            from google import genai

            _client = genai.Client(api_key=config.GEMINI_API_KEY)
        except Exception as e:
            print(f"[gemini] unavailable: {e}")
            return None
    return _client


def _image_part(jpg_bytes):
    from google.genai import types

    return types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg")


def _wait_turn():
    """Free-tier limits: never call in a tight loop."""
    global _last_call
    wait = config.GEMINI_MIN_GAP - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _load_json(text):
    """json.loads that also copes with ```json fences and stray words around the JSON."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    match = re.search(r"[\[{].*[\]}]", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


def parse_gemini_json(text, width, height):
    """Turn Gemini's JSON reply into pixel detections. Never raises on odd data."""
    data = _load_json(text)
    if isinstance(data, dict):  # sometimes wrapped as {"objects": [...]}
        data = next((v for v in data.values() if isinstance(v, list)), [])
    dets = []
    for obj in data if isinstance(data, list) else []:
        try:
            label = normalise_label(obj["label"])
            if label is None:
                continue  # not one of our items
            # Gemini order is [ymin, xmin, ymax, xmax] on a 0-1000 grid: y comes FIRST.
            ymin, xmin, ymax, xmax = (min(1000.0, max(0.0, float(v))) for v in obj["box_2d"])
        except (KeyError, TypeError, ValueError):
            continue
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        x1, x2 = int(xmin / 1000 * width), int(xmax / 1000 * width)
        y1, y2 = int(ymin / 1000 * height), int(ymax / 1000 * height)
        if x2 > x1 and y2 > y1:
            dets.append((x1, y1, x2, y2, label, 1.0))
    return dets


def gemini_detect(frame, client=None):
    """SCAN: ask Gemini for item boxes in `frame` (people already blurred by the caller).
    Raises on failure (the caller falls back to YOLO)."""
    client = client or _get_client()
    if client is None:
        raise RuntimeError("Gemini is not configured")
    _wait_turn()
    h, w = frame.shape[:2]
    ok, jpg = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("could not encode frame")
    reply = client.models.generate_content(
        model=config.GEMINI_SCAN_MODEL,
        contents=[_image_part(jpg.tobytes()), config.GEMINI_PROMPT],
        config={"response_mime_type": "application/json"},
    )
    return parse_gemini_json(reply.text, w, h)


def read_verdict(text):
    """Gemini's check reply -> "ok", "none", "held", or None if it can't be read."""
    words = (text or "").strip().split()
    word = words[0].strip(".,:;-!\"'*").upper() if words else ""
    return {"YES": "ok", "NONE": "none", "NO": "none", "HELD": "held"}.get(word)


def gemini_check(frame, box, item, client=None):
    """CHECK: is there really a `item` in `box`? `frame` has people blurred already.
    Returns "ok" / "none" / "held", or None if the reply was unreadable. Raises on API failure."""
    client = client or _get_client()
    if client is None:
        raise RuntimeError("Gemini is not configured")
    img = frame.copy()
    cv2.rectangle(img, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:4]), ORANGE, 3)
    ok, jpg = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("could not encode frame")
    _wait_turn()
    reply = client.models.generate_content(
        model=config.GEMINI_CHECK_MODEL,
        contents=[_image_part(jpg.tobytes()), config.GEMINI_CHECK_PROMPT.format(item=item)],
    )
    return read_verdict(reply.text)


def gemini_pick_label(question, client=None):
    """Fallback when keywords fail: ask Gemini which item a question is about, or None."""
    client = client or _get_client()
    if client is None:
        return None
    try:
        reply = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=f"Which one of {config.ITEMS} is this question about? "
                     f"Reply with just the item name, or NONE.\nQuestion: {question}",
        )
        return normalise_label(reply.text)
    except Exception as e:
        print(f"[gemini] could not interpret question: {e}")
        return None
