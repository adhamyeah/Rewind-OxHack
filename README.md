# Rewind — OxHack 2026

A camera watches the places people put things down, remembers where each item
settled, and answers "where are my keys?" with a spoken sentence and a photo.

```
webcam → main.py (YOLO-World + Gemini) → rewind.db → server.py → web page
```

| File | What it does |
|---|---|
| `main.py` | Webcam window. Detects items once the scene goes still, logs placed/gone, saves a photo (people blurred), asks Gemini where the item is |
| `db.py` | The shared `events` table (SQLite) |
| `server.py` | Web server: `/items`, `/where?q=…`, photos, the page |
| `static/` | The page: one big button, photo tiles, voice in and out |
| `.env` | Your Gemini API key (never committed) |
| `yolov8s-worldv2.pt` | YOLO-World model, committed so nobody has to download it |

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

Put an item down and take your hand away. Within a second the terminal prints
`keys placed …`, then `Gemini: keys next to the blue mug`, and the tile appears.

Options: `--source 1` (another camera), `--source clip.mp4` (a video),
`--items "keys,glasses"` (other names), `--no-window --seconds 20`.

The first run downloads a 338 MB text encoder into `weights/`; wait for it.

## Privacy

Everything runs on the laptop except the Gemini description: one still photo per
placed item, with people blurred, and only when a key is in `.env`.
Photos are deleted after 24 hours. No video is stored.

## Git

```bash
git pull
git status                  # .env, rewind.db, snapshots/ and weights/ must not appear
git add -A
git commit -m "what changed"
git push
```
