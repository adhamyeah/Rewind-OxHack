"""Data cleaning for everything Rewind captures, before it's used.

1. Frames: remove sensor grain and fix light without blurring (Enhancer), and
   skip frames not worth a Gemini look: blurred by motion, too dark, washed out.
2. Microphone audio: remove the DC offset and low hum, subtract the room's
   background noise, trim silence at both ends, and even out the volume.
3. Gemini's detections: check every field, tidy names ("A Cellphone." -> "phone"),
   keep only known types, fix and clamp boxes, drop impossible ones, merge
   duplicates, and keep an object's name steady from one look to the next.
"""

import re

import cv2
import numpy as np

# ---------- 1. frames ----------

STILL_WEIGHT = 0.2                  # still pixels: newest frame counts 20% (averages ~5 frames of grain away)
MOVING_DIFF = 20                    # a pixel that changed more than this (grey levels) really moved: no averaging, no smear
CONTRAST_LIMIT = 1.5                # local contrast boost; higher also boosts leftover grain
SHARPEN = 0.5                       # how much real edges are sharpened
SHARPEN_ABOVE = 6                   # detail weaker than this (grey levels) is grain: not sharpened
TARGET_BRIGHTNESS = 120             # darker frames get their shadows lifted
TOO_DARK, TOO_BRIGHT = 25, 235      # average brightness outside this: lens covered / camera blinded
BLURRY_SHARE = 0.5                  # sharpness under half the recent normal = motion blur


class Enhancer:
    """Cleans each frame without blurring it.

    Noise: each pixel is averaged over time only while it stays still. A pixel
    that really changes (a hand, an object moving) takes the new frame at once,
    so nothing smears, and there's no spatial blur to soften edges.
    Light: shadows are lifted if the frame is dark, local contrast is boosted,
    and a light sharpening pass keeps edges crisp.
    """

    def __init__(self):
        self.avg = None
        self.clahe = cv2.createCLAHE(clipLimit=CONTRAST_LIMIT, tileGridSize=(8, 8))
        self.dot = np.ones((3, 3), np.uint8)
        self.pad = np.ones((5, 5), np.uint8)

    def __call__(self, frame):
        if self.avg is None or self.avg.shape != frame.shape:
            self.avg = frame.astype(np.float32)
        else:
            change = cv2.cvtColor(cv2.absdiff(frame, cv2.convertScaleAbs(self.avg)), cv2.COLOR_BGR2GRAY)
            moved = (change >= MOVING_DIFF).astype(np.uint8)
            # real movement is a blob (a hand); noise is lone dots: drop the dots, pad the blobs' edges
            moved = cv2.dilate(cv2.morphologyEx(moved, cv2.MORPH_OPEN, self.dot), self.pad)
            cv2.accumulateWeighted(frame, self.avg, STILL_WEIGHT, mask=1 - moved)   # still: average grain away
            cv2.accumulateWeighted(frame, self.avg, 1.0, mask=moved)                # moved: take it as it is now
        img = cv2.convertScaleAbs(self.avg)
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        mean = float(l.mean())
        if mean < TARGET_BRIGHTNESS - 20:                    # too dark: lift the shadows
            gamma = np.log(TARGET_BRIGHTNESS / 255) / np.log(max(mean, 1.0) / 255)
            l = cv2.LUT(l, (((np.arange(256) / 255) ** gamma) * 255).astype(np.uint8))
        l = self.clahe.apply(l)                              # local contrast: edges stand out
        # sharpen real edges only: flat areas (walls, table) are left alone so grain isn't boosted
        soft = cv2.GaussianBlur(l, (0, 0), 1.2)
        edges = cv2.compare(cv2.absdiff(l, soft), SHARPEN_ABOVE, cv2.CMP_GE)
        cv2.copyTo(cv2.addWeighted(l, 1 + SHARPEN, soft, -SHARPEN, 0), edges, l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


class FrameCheck:
    """Says whether a raw frame is worth a look, and why not if it isn't."""

    def __init__(self):
        self.recent = []                                 # sharpness of recent good frames

    def __call__(self, frame):
        grey = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        brightness = float(grey.mean())
        if brightness < TOO_DARK:
            return False, "too dark (lens covered?)"
        if brightness > TOO_BRIGHT:
            return False, "too bright"
        sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        normal = float(np.median(self.recent)) if len(self.recent) >= 5 else 0.0
        if normal and sharpness < BLURRY_SHARE * normal:
            return False, "blurred by movement"
        self.recent = (self.recent + [sharpness])[-30:]
        return True, ""


# ---------- 2. audio ----------

HUM_BELOW_HZ = 80                   # mains hum and desk rumble live below this
NOISE_FRAME = 512                   # samples per analysis frame (32 ms at 16 kHz)
NOISE_OVERSUBTRACT = 1.5            # remove a bit more than the measured noise...
NOISE_FLOOR = 0.1                   # ...but never below 10% of the original (avoids "watery" artefacts)
TARGET_PEAK = 0.9                   # loudest sample ends at 90% of full scale
MAX_GAIN = 20.0                     # don't blow quiet noise up into "speech"


def _frames(x):
    """Overlapping windowed frames (50% overlap) for short-time spectra."""
    hop = NOISE_FRAME // 2
    count = max(1, 1 + (len(x) - NOISE_FRAME) // hop)
    x = np.pad(x, (0, max(0, (count - 1) * hop + NOISE_FRAME - len(x))))
    idx = np.arange(NOISE_FRAME)[None, :] + hop * np.arange(count)[:, None]
    return x[idx] * np.hanning(NOISE_FRAME), hop


def noise_profile(quiet_samples):
    """Average spectrum of the room when nobody is talking."""
    if quiet_samples is None or len(quiet_samples) < NOISE_FRAME:
        return None
    frames, _ = _frames(quiet_samples.astype(np.float32))
    return np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0)


def clean_audio(samples, rate, noise=None):
    """int16 speech -> cleaned int16 speech, ready to send."""
    x = samples.astype(np.float32)
    x -= x.mean()                                        # DC offset
    frames, hop = _frames(x)
    spec = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(NOISE_FRAME, 1 / rate)
    spec[:, freqs < HUM_BELOW_HZ] = 0                    # hum and rumble
    if noise is not None:                                # spectral subtraction of the room's noise
        mag = np.abs(spec)
        cleaned = np.maximum(mag - NOISE_OVERSUBTRACT * noise[None, :], NOISE_FLOOR * mag)
        spec *= cleaned / np.maximum(mag, 1e-9)
    out = np.zeros((len(frames) - 1) * hop + NOISE_FRAME, np.float32)
    window_sum = np.zeros_like(out)
    for i, f in enumerate(np.fft.irfft(spec, n=NOISE_FRAME, axis=1)):   # overlap-add back to sound
        out[i * hop:i * hop + NOISE_FRAME] += f * np.hanning(NOISE_FRAME)
        window_sum[i * hop:i * hop + NOISE_FRAME] += np.hanning(NOISE_FRAME) ** 2
    out = out[:len(x)] / np.maximum(window_sum[:len(x)], 1e-3)

    energy = np.array([np.sqrt((out[i:i + hop] ** 2).mean()) for i in range(0, len(out), hop)])
    loud = np.where(energy > 0.15 * energy.max())[0] if energy.size and energy.max() > 0 else []
    if len(loud):                                        # trim silence, keep 0.2 s either side
        pad = int(0.2 * rate)
        out = out[max(0, loud[0] * hop - pad):min(len(out), (loud[-1] + 1) * hop + pad)]
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0:
        out *= min(TARGET_PEAK * 32767 / peak, MAX_GAIN)  # even out the volume
    return np.clip(out, -32768, 32767).astype(np.int16)


# ---------- 3. detections ----------

TYPES = ["person", "animal", "electronics", "kitchenware", "food & drink", "furniture",
         "clothing & accessories", "personal care", "stationery & books", "toys & games",
         "tools", "plants", "decor", "fixtures", "vehicles", "other"]

# other words for the same thing -> one name, so memory and questions match
SYNONYMS = {
    "cellphone": "phone", "cell phone": "phone", "mobile": "phone", "mobile phone": "phone",
    "smartphone": "phone", "iphone": "phone", "cup": "mug", "coffee mug": "mug", "coffee cup": "mug",
    "spectacles": "glasses", "eyeglasses": "glasses", "specs": "glasses", "sunglass": "sunglasses",
    "key": "keys", "keychain": "keys", "car keys": "keys", "remote control": "remote",
    "tv remote": "remote", "purse": "wallet", "notebook computer": "laptop", "computer mouse": "mouse",
    "earbuds": "earphones", "earphone": "earphones", "headphone": "headphones",
    "water bottle": "bottle", "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "people": "person", "human": "person",
}
MIN_BOX_SHARE = 0.0005              # boxes smaller than this share of the frame are noise
MAX_BOX_SHARE = 0.9                 # bigger than this is "the whole picture", not an object
SAME_THING_IOU = 0.5                # boxes overlapping this much are the same thing


def clean_name(name):
    name = re.sub(r"[^a-z0-9 &'-]", " ", str(name).lower())
    name = re.sub(r"^(a|an|the|some)\s+", "", re.sub(r"\s+", " ", name).strip())
    return SYNONYMS.get(name, name) or "thing"


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def clean_detections(raw, width, height, previous=()):
    """Gemini's raw JSON objects -> checked, tidied, de-duplicated objects in frame pixels."""
    found = []
    for o in raw if isinstance(raw, list) else []:
        if not isinstance(o, dict):
            continue
        box = o.get("box_2d")
        if not (isinstance(box, (list, tuple)) and len(box) == 4
                and all(isinstance(v, (int, float)) for v in box)):
            continue                                     # missing or garbled box
        y1, x1, y2, x2 = (min(max(float(v), 0.0), 1000.0) for v in box)   # Gemini uses 0-1000
        x1, x2 = sorted((x1 * width / 1000, x2 * width / 1000))
        y1, y2 = sorted((y1 * height / 1000, y2 * height / 1000))
        box = (int(x1), int(y1), int(x2), int(y2))
        name = clean_name(o.get("label", ""))
        kind = str(o.get("type", "")).strip().lower()
        if name == "person" or kind == "person":
            name, kind = "person", "person"
        if kind not in TYPES:
            kind = "other"
        share = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
        if share < MIN_BOX_SHARE or (share > MAX_BOX_SHARE and kind != "person"):
            continue                                     # a speck, or the whole picture
        place = re.sub(r"\s+", " ", str(o.get("place", ""))).strip().rstrip(".")
        found.append(dict(box=box, name=name, type=kind, place=place[:1].lower() + place[1:],
                          movable=bool(o.get("movable")) and kind not in ("person", "furniture", "fixtures")))

    kept = []                                            # merge duplicates: same name, same spot
    for o in sorted(found, key=lambda o: -(o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])):
        if not any(k["name"] == o["name"] and iou(k["box"], o["box"]) > SAME_THING_IOU for k in kept):
            kept.append(o)

    for o in kept:                                       # same thing as last look under another name? keep the old name
        match = max(previous, key=lambda p: iou(p["box"], o["box"]), default=None)
        if match and match["name"] != o["name"] and match["type"] == o["type"] \
                and iou(match["box"], o["box"]) > SAME_THING_IOU \
                and not any(k["name"] == match["name"] for k in kept):
            o["name"] = match["name"]
    return kept
