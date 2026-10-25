# Rewind — OxHack 2026

Ctrl+Z for the physical world. Overhead camera + change-gated detection:
an object move triggers one detection, logged as appeared/left-view events.

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

### Weights (one time, both platforms, run from repo root)

YOLO-World does **not** auto-download its weights. Fetch once:

```bash
curl -L -o yolov8m-worldv2.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-worldv2.pt
```

(Windows PowerShell: use `curl.exe`, not `curl`, or `Invoke-WebRequest -Uri <url> -OutFile yolov8m-worldv2.pt`.)

## Run

**Once: API keys.** Create a file called `.env` in the repo root (it's git-ignored; never commit it):

```
GEMINI_API_KEY=your-gemini-key
ELEVENLABS_API_KEY=your-elevenlabs-key
ELEVENLABS_VOICE_ID=optional-voice-id
```

The ElevenLabs key needs **Text to Speech** and **Speech to Text** access. Then check each part:

```bash
python describe.py              # Gemini: one tiny test call, prints OK or what's wrong
python speak.py                 # ElevenLabs says a test sentence out loud
python listen.py                # say "where are my keys?" - prints what it heard
```

**Once per camera position: zones.** Tape the camera down, then draw the areas
items get left on (name them like "on the kitchen table"):

```bash
python zone_editor.py           # click corners, Enter to name, q to save
```

**Every time:**

```bash
python phase1_camera_gate.py    # q to quit;  --no-gemini / --no-voice / --no-listen / --quiet / --wake
```

Just ask out loud: "where are my glasses?" - or press **1-8** in the video window.
Say "show me where my glasses were" (or press **r** after asking) for the **rewind timeline**:
the item's last places plus a replay of the moment it was put down. `python timeline.py glasses`
opens it on its own. In a noisy room, `--wake` makes Rewind answer only questions that start with "Rewind".

| File | What it does |
| --- | --- |
| `phase1_camera_gate.py` | Camera + detection (YOLO-World items, yolo11n people). Starts everything. |
| `zones.py`, `zone_editor.py` | Named areas of the camera view, and the tool to draw them |
| `memory.py` | Put down / picked up / in your hand, last-seen times, snapshots and clips, in `rewind.db` |
| `describe.py` | Gemini: checks each put-down, scans the scene when it settles, describes spots, understands odd questions |
| `listen.py` | Spoken questions: microphone + ElevenLabs speech-to-text |
| `speak.py` | Spoken answers: ElevenLabs voice |
| `timeline.py` | The rewind timeline: history + replay |

## Git workflow (use this loop every time)

```bash
git pull                          # grab the other person's changes first
# ...work...
git status                        # always check what's staged before committing
# the venv + __pycache__ + .DS_Store should be ignored here; the .pt weights should NOT be
# (the weights are committed on purpose so both machines skip the 26 MB download)
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
