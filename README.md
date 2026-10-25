# Rewind — OxHack 2026

A camera watches the places people put things down, says what it sees, remembers
where each item settled, and answers "where is my phone?" out loud and on a web page.

```
webcam → main.py (Gemini sees) → rewind.db → server.py → web page
```

Gemini does all the seeing: there's no local detection model. Each "look" sends one
cleaned-up frame to Gemini, which returns every object and person with a box, a name
and where it is ("next to the blue mug"). It can name almost anything.

| File | What it does |
|---|---|
| `main.py` | Webcam window. Cleans each frame (less noise, better light), asks Gemini to look when the scene changes, draws the boxes, says each new thing's name, listens to the microphone, logs placed/gone with a photo (people blurred) |
| `db.py` | The shared `events` table (SQLite) and the spoken answer wording |
| `server.py` | Web server: `/items`, `/where?q=…`, photos, the page |
| `static/` | The page: one big button, photo tiles, voice in and out |
| `.env` | Your Gemini API key (never committed) |

## Setup (once)

Windows PowerShell, from this folder:

```powershell
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
copy .env.example .env      # then open .env and paste your key after GEMINI_API_KEY=
```

macOS: `python3 -m venv .venv`, then `.venv/bin/python -m pip install -r requirements.txt`.

A Gemini key is required: without it Rewind can't see. Get one free at Google AI Studio.

## Run

Terminal 1 — the camera (Q in the window to quit):

```powershell
.venv/Scripts/python main.py
```

Terminal 2 — the web page, then open http://localhost:8000 in Chrome:

```powershell
.venv/Scripts/python -m uvicorn server:app --reload
```

Put something on the table and take your hand away. Within a few seconds it gets a box,
Rewind says its name, and a tile appears on the page.

Ask out loud: "where is my phone?" or "what can you see?". Rewind answers with its voice.

In the window: **SPACE** looks now, **E** switches between the cleaned-up and raw view,
**Q** quits. The top bar shows the video fps, how many looks so far, what Rewind is doing
("waiting for still", "looking...", "last look 12s ago") or a Gemini problem, and the mic level.

## How often it looks (and the free quota)

Every look is one Gemini request, and free keys have a daily limit per model
(`gemini-3.5-flash` allows only 20 a day; the default `gemini-3.1-flash-lite` allows far more).
So Rewind doesn't look all the time. It looks when:

- the picture changed and then stayed still for a moment (a hand put something down and left)
- an item just disappeared (one re-check 6 s later, so a single miss doesn't count as gone)
- a minute passed with no look, or you press SPACE

Never more than one look every 4 seconds. Between looks the boxes stay where they were.
If the top bar says "daily limit reached", Rewind waits and tries again each minute; to use
another model, put `GEMINI_MODEL=...` in `.env`. Voice questions use the same quota.

The terminal stays quiet: Rewind just says the names of what it sees. Add `--verbose` to also
print every look, placed/gone event and heard question.

Options: `--source 1` (another camera), `--source clip.mp4` (a video), `--no-voice`,
`--no-mic`, `--mic-threshold 60` (if it doesn't hear you; compare with the mic number on
screen), `--no-window --seconds 20`.

## Privacy

Gemini sees a still frame each time Rewind looks (not video), and a short recording each
time someone speaks near the microphone (`--no-mic` turns listening off). Saved photos of
put-down items have people blurred and are deleted after 24 hours. No video is stored.

## Git

```bash
git pull
git status                  # .env, rewind.db and snapshots/ must not appear
git add -A
git commit -m "what changed"
git push
```
