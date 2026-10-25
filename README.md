# Rewind — OxHack 2026

A camera watches the places people put things down, says what it sees, remembers
where each item settled, and answers "where is my phone?" out loud and on a web page.

```
webcam → main.py (YOLO-World + Gemini) → rewind.db → server.py → web page
```

| File | What it does |
|---|---|
| `main.py` | Webcam window with live boxes for 80 everyday things and people. Cleans each frame (less noise, better light) before detection, says each thing's name, listens to the microphone, logs placed/gone with a photo (people blurred), asks Gemini where the item is |
| `db.py` | The shared `events` table (SQLite) |
| `server.py` | Web server: `/items`, `/where?q=…`, photos, the page |
| `static/` | The page: one big button, photo tiles, voice in and out |
| `.env` | Your Gemini API key (never committed) |
| `yolo11s.pt` | Standard YOLO model (80 things), downloads itself on first run |
| `yolov8s-worldv2.pt` | YOLO-World, only used with `--items` for things outside those 80 (keys, glasses) |

## Setup (once)

Windows PowerShell, from this folder:

```powershell
py -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
copy .env.example .env      # then open .env and paste your key after GEMINI_API_KEY=
```

macOS: `python3 -m venv .venv`, then `.venv/bin/python -m pip install -r requirements.txt`.

No key? Leave `.env` empty: Rewind still works and says "on the left of the camera view" instead.

## Run

Terminal 1 — the camera (Q in the window to quit):

```powershell
.venv/Scripts/python main.py
```

Terminal 2 — the web page, then open http://localhost:8000 in Chrome:

```powershell
.venv/Scripts/python -m uvicorn server:app --reload
```

Hold a cup, phone or bottle up to the camera: it gets a box and Rewind says its name.
Put it down and leave it: the terminal prints `cup placed …`, then `Gemini: cup next to…`,
and a tile appears on the page.

Ask out loud: "where is my phone?" or "what can you see?". Rewind answers with its voice.

In the window: **E** switches between the cleaned-up (enhanced) and raw camera view, **Q** quits.
The top bar shows video fps, detection speed and the microphone level.

Options: `--source 1` (another camera), `--source clip.mp4` (a video),
`--items "keys,glasses"` (names outside the 80, uses YOLO-World), `--no-voice`, `--no-mic`,
`--mic-threshold 60` (if it doesn't hear you; compare with the mic number on screen),
`--no-window --seconds 20`.

What it can name: person, cell phone, cup, bottle, laptop, book, remote, keyboard, mouse,
scissors, backpack, handbag, umbrella, chair, tv and the rest of the 80 COCO classes.

## Privacy

Everything runs on the laptop except Gemini: one still photo per placed item (people
blurred), and a short recording each time someone speaks near the microphone. Both only
when a key is in `.env`; `--no-mic` turns listening off.
Photos are deleted after 24 hours. No video is stored.

## Git

```bash
git pull
git status                  # .env, rewind.db, snapshots/, weights/ and yolo11s.pt must not appear
git add -A
git commit -m "what changed"
git push
```
