"""
Gemini as the memory detector: open-vocabulary, so it finds items YOLO-World misses.

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


def parse_gemini_json(text, width, height):
    """Turn Gemini's JSON reply into pixel detections. Never raises on odd data."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
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
    """Ask Gemini for item boxes in `frame`. Raises on failure (the caller falls back)."""
    global _last_call
    client = client or _get_client()
    if client is None:
        raise RuntimeError("Gemini is not configured")
    wait = config.GEMINI_MIN_GAP - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)  # free-tier limits: never call in a tight loop
    _last_call = time.time()

    h, w = frame.shape[:2]
    ok, jpg = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("could not encode frame")
    reply = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[_image_part(jpg.tobytes()), config.GEMINI_PROMPT],
        config={"response_mime_type": "application/json"},
    )
    return parse_gemini_json(reply.text, w, h)


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
