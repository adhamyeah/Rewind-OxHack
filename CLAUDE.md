# Rewind

A camera watches the places people put things down, remembers where each item
settled, and answers "where are my keys?" with a spoken sentence and a photo.
Built for people living with dementia. Follow `TICKETS.md` for what's left.

## Layout

- `detector.py`: camera loop; YOLO-World runs once the scene changes then goes still
- `memory.py`: placed/gone rules, snapshots (people blurred), writes events via `db.py`
- `zones.py` / `zone_editor.py`: named areas of the view, saved in `zones.json` (per camera, not committed)
- `camera.py`: opens the webcam (DirectShow + MJPG on Windows)
- `db.py`: SQLite helpers; the detector writes, the server only reads
- `server.py`: FastAPI `/items`, `/where`, `/snapshots/*`, serves `static/`
- `static/`: the UI (`index.html`, `app.js`, `style.css`)
- `seed_fake.py`: fake rows + photos so the UI works without a camera
- `tests/`: pytest

## Data contract: `events` table (see `db.py`)

| column | meaning |
|---|---|
| item | detector's name, e.g. `keys`, `remote control` |
| event | `placed` or `gone` |
| zone | phrase that finishes "Your keys are…", e.g. `on the kitchen table` |
| x1, y1, x2, y2 | box in camera pixels |
| at | unix time |
| image | snapshot path (`snapshots/…jpg`), placed events only |
| picked_up | 1 if a person touched it just before it went |

`/items` returns `[{item, status, ago, photo, answer}]`, newest first.

## Run

```bash
python seed_fake.py                 # optional: fake data
uvicorn server:app --reload         # http://localhost:8000 in Chrome
python zone_editor.py               # once per camera position
python detector.py                  # camera, in a second terminal
python -m pytest                    # tests (pip install -r requirements-dev.txt)
```

## Rules

- New FastAPI routes go above the `app.mount(...)` lines in `server.py`.
- Never commit `rewind.db`, `snapshots/`, `.env` or API keys.
- Commit before letting an agent change things.
