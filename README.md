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
curl -L -o yolov8s-worldv2.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt
```

(Windows PowerShell: use `curl.exe`, not `curl`, or `Invoke-WebRequest -Uri <url> -OutFile yolov8s-worldv2.pt`.)

## Run

```bash
python phase1_camera_gate.py    # Q to quit
```

VSCode: Command Palette -> "Python: Select Interpreter" -> pick the `.venv`
interpreter at the repo path, then run from the VSCode terminal (it activates
the venv automatically).

## Open vocabulary (branch `007`)

Rewind isn't limited to the 9 starting items in `config.ITEMS` any more:

- **Gemini detects every object in the frame** at each keyframe (furniture, devices, papers,
  signs, clothing, anything held up to the camera) and names each by what it really is
  (jacket, phone, tablet, poster...), from any camera angle. Names are kept consistent: "coffee cup"
  -> mug, "red stapler" -> stapler once "stapler" is known. People, body parts and the room
  itself (walls, floor, windows) are never items.
- **YOLO-World learns from Gemini**: it starts with no names at all and looks only for what
  Gemini has actually predicted (apple, stapler, jacket...), so the live view tracks those
  between keyframes. Only the 40 most recent names are kept (`config.MAX_LEARNED`).
- **The window always shows Gemini's latest boxes** (for 20 s, `config.GEMINI_BOX_SECONDS`),
  even while things move; YOLO adds people and anything Gemini's last look missed.
- **Questions match anything seen** ("where's my stapler?"), and a thing never seen gets
  "I haven't seen the stapler yet" instead of "I don't know that item".

A keyframe (one Gemini look) fires when the scene settles, or after 5 seconds of
constant change (`config.MAX_KEYFRAME_WAIT`), so a thing held up to the camera still gets
seen. Press **SPACE** in the window to look right now.

The names live in `vocab.py`. Set `OPEN_VOCAB = False` in `config.py` to go back to
only the starting items. Run as usual: `python rewind.py`.

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
