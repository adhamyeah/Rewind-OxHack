# Rewind — OxHack 2026

Ctrl+Z for the physical world. A camera watches the table, remembers where things were put
down, and answers "where are my keys?" out loud — then replays the moment they were put down.

This branch (`merged`) combines both versions: the detection, memory, Gemini, voice and
microphone from `detection`, with the web page and rewind slider from `keshav-changes`.

## Setup

One venv per machine, created from the repo directory. Never commit `.venv/`.

### macOS

```bash
cd rewind-oxhack
python3 -m venv .venv          # if `python3 --version` < 3.10, use /opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd rewind-oxhack
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # optional but recommended: skips the ~2.5 GB CUDA torch build
python -m pip install -r requirements.txt
```

If `Activate.ps1` is blocked by PowerShell's execution policy, run
`Set-ExecutionPolicy -Scope Process Bypass` first, or use cmd and
`.venv\Scripts\activate.bat` instead.

Optional extras (Tiger Data copy, the physical rewind knob): `pip install -r requirements-extras.txt`.

### Weights

The three `.pt` weights are committed, so `git pull` brings them. If one is missing:

```bash
curl -L -o yolov8m-worldv2.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-worldv2.pt
```

(Windows PowerShell: use `curl.exe`, not `curl`.) If the CLIP text model download fails with an
SSL error (some venue Wi-Fi), fetch it with curl:

```bash
mkdir -p weights/clip
curl -L -o weights/clip/ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

### API keys (once)

Copy `.env.example` to `.env` and fill it in (it's git-ignored; never commit it). The ElevenLabs
key needs **Text to Speech** and **Speech to Text** access. Then check everything:

```bash
python doctor.py --network      # packages, keys, camera, microphone, weights, one Gemini + ElevenLabs call
```

## Run

```bash
python rewind.py                # opens http://127.0.0.1:8000 in your browser;  Ctrl+C to quit
python rewind.py --source 1     # second camera (e.g. a phone webcam)
```

On the page:

- **Ask** — just say "where are my glasses?" (the microphone is always listening for questions),
  hold **Talk** (or the space bar), or type it. Rewind answers out loud and **replays the moment**
  it was put down (or taken) as a little rewind gif, with the item outlined in yellow.
- **The slider** steps back through the table's history: one moment each time the scene settles
  or something is put down or taken. Times count from when Rewind started (`0m 00s`): every run
  starts fresh. Arrow keys step, **R** replays, **L** goes back to live.
- **What changed** lists put down / picked up / put back / out of view; **Replay** shows each one.
- **Zones** — click **Draw a zone**, click the corners of an area on the live picture, name it
  like "on the kitchen table", save. Answers then say "Your keys are on the kitchen table".
  (Tape the camera down: zones don't move with it.)

Options: `--port 8001` (if 8000 is busy), `--no-browser`, `--no-gemini` (offline / save quota),
`--no-voice` (print answers), `--quiet` (speak only when asked), `--no-listen` (no always-on mic;
Talk still works), `--wake` (noisy room: only answer spoken questions that start with "Rewind"),
`--knob COM3` (physical rewind knob, see `knob/knob.ino`). Questions typed in the terminal work too.
Only your own computer can open the page (it listens on 127.0.0.1).

`python phase1_camera_gate.py` is a quick camera + detection check without the rest.

## Files

| File | What it does |
| --- | --- |
| `rewind.py` | Starts everything and opens the page |
| `phase1_camera_gate.py` | Camera + detection: YOLO-World medium at 640 for items, yolo11n for people, change gate, "scene settled" |
| `memory.py` | Put down / picked up / in your hand / put back, last-seen times, the saved moments and their gifs (`rewind.db`) |
| `describe.py` | Gemini: checks each put-down, scans the scene when it settles, describes spots, understands odd questions |
| `zones.py` | Named areas of the camera view (`zones.json`); drawn from the page, or with `zone_editor.py` |
| `answer.py` | Question → item → answer + which moment to replay |
| `listen.py` | Spoken questions: always-on microphone and the Talk button → ElevenLabs speech-to-text |
| `speak.py` | Spoken answers: ElevenLabs voice |
| `web.py`, `static/index.html` | The web page: live view, rewind slider, ask box, events, zones |
| `ui.py` | Drawing boxes, the rewind gif, notes between threads |
| `store.py`, `knob.py`, `config.py` | Optional Tiger Data copy, optional knob, keys from `.env` |
| `doctor.py` | Checks the machine is ready |
| `tests/` | `pip install -r requirements-dev.txt` then `python -m pytest` |

## Git workflow (use this loop every time)

```bash
git pull                          # grab the other person's changes first
# ...work...
git status                        # always check what's staged before committing
# the venv + __pycache__ + .DS_Store should be ignored here; the .pt weights should NOT be
# (the weights are committed on purpose so both machines skip the download)
git add -A                        # stage everything, dotfiles included
# never use `git add *` — the shell glob skips dotfiles like .gitignore
git commit -m "short description of what changed"
git push
```

If `git push` is rejected because the other person pushed first:

```bash
git pull --rebase
git push
```
