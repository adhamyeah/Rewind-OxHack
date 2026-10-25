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
from vocab import VOCAB

_client = None
_last_call = 0.0
_used_up = set()   # models whose quota ran out this session: skipped until Rewind restarts


def is_quota_error(e):
    return "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)


def _generate(client, **kw):
    """generate_content on the first model that still has quota.

    Each Gemini model has its own free daily limit, so when one runs out we move on to the
    next in config.GEMINI_MODELS instead of losing detection. Raises the last quota error
    once every model is used up (the worker then falls back to YOLO).
    """
    last = None
    for model in config.GEMINI_MODELS:
        if model in _used_up:
            continue
        try:
            return client.models.generate_content(model=model, **kw)
        except Exception as e:
            if not is_quota_error(e):
                raise
            _used_up.add(model)
            print(f"[gemini] {model}: quota used up, switching model")
            last = e
    raise last or RuntimeError("429 RESOURCE_EXHAUSTED: every Gemini model's quota is used up")


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


def parse_gemini_json(text, width, height, vocab=None):
    """Turn Gemini's JSON reply into pixel detections. Never raises on odd data.

    With a vocab (open vocabulary) any thing is kept under its tracked name;
    without one, only config.ITEMS are kept.
    """
    to_label = vocab.canonical if vocab is not None else normalise_label
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return []
    if isinstance(data, dict):  # sometimes wrapped as {"objects": [...]}
        data = next((v for v in data.values() if isinstance(v, list)), [])
    dets = []
    for obj in data if isinstance(data, list) else []:
        try:
            label = to_label(obj["label"])
            if label is None:
                continue  # not one of our items (or a person / part of the room)
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


def _default_vocab():
    return VOCAB if config.OPEN_VOCAB else None


def gemini_detect(frame, client=None, vocab=None):
    """Ask Gemini for item boxes in `frame`. Raises on failure (the caller falls back).

    Open vocabulary: Gemini may name anything; new names are learned into `vocab`.
    """
    global _last_call
    vocab = vocab if vocab is not None else _default_vocab()
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
    prompt = config.GEMINI_OPEN_PROMPT if vocab else config.GEMINI_PROMPT
    reply = _generate(
        client,
        contents=[_image_part(jpg.tobytes()), prompt],
        config={"response_mime_type": "application/json", "response_schema": config.GEMINI_OPEN_SCHEMA}
        if vocab else {"response_mime_type": "application/json"},
    )
    dets = parse_gemini_json(reply.text, w, h, vocab)
    if vocab is not None:
        new = vocab.learn([d[4] for d in dets])
        if new:
            print(f"[vocab] learned: {', '.join(new)}")
    return dets


def gemini_pick_label(question, client=None, vocab=None):
    """Fallback when keywords fail: ask Gemini which item a question is about, or None.

    Open vocabulary: a thing we haven't seen yet still gets a name, so the answer can
    say "I haven't seen the stapler yet" instead of "I don't know that item".
    """
    vocab = vocab if vocab is not None else _default_vocab()
    client = client or _get_client()
    if client is None:
        return None
    if vocab is not None:
        prompt = (f"Things a home camera has seen: {', '.join(vocab.names()) or 'nothing yet'}. Which thing is this "
                  "question about? Reply with the matching name from that list; if it's about a "
                  "thing not in the list, reply with its plain name in 1-3 words; if it isn't about "
                  f"a thing, reply NONE.\nQuestion: {question}")
    else:
        prompt = (f"Which one of {config.ITEMS} is this question about? "
                  f"Reply with just the item name, or NONE.\nQuestion: {question}")
    try:
        reply = _generate(client, contents=prompt)
        text = (reply.text or "").strip()
        if vocab is None:
            return normalise_label(text)
        return None if text.upper().startswith("NONE") else vocab.canonical(text)
    except Exception as e:
        print(f"[gemini] could not interpret question: {e}")
        return None
