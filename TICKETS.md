# Rewind — tickets

Gap analysis of `main` (commit `02db1b0`) against *Rewind — Full Build Guide*.
Owners are the guide's roles: **Vision**, **Backend**, **Frontend**, **Pitch**.

## Where `main` is today

| Stage | Status | Notes |
|---|---|---|
| 0 Setup | Partial | README + venv steps done. `requirements.txt` lacks fastapi, uvicorn, CLIP. README says the `.pt` weights are committed, but they are not in the repo. |
| 1 Smooth camera | Partial, different design | `phase1_camera_gate.py` runs YOLO only when the scene changes, instead of the guide's 3-thread loop. Good for CPU laptops, but see REW-04. |
| 2 Detect items | Partial | YOLO-World with 8 item names, one global confidence. No per-item thresholds, no size filter, no ArUco, no `person` class. |
| 3 Zones | Not started | |
| 4 Last-seen memory | Not started | Currently prints "appeared / left view" only. |
| 5 Ask "where is it?" | Not started | |
| 6 Timeline | Not started (stretch) | |
| 7 LLM layer | Not started (stretch) | |
| 8 Polish + privacy | Not started | |
| 9 Demo + pitch | Not started | |

## Tickets

Priority: P0 = needed for the ugly end-to-end demo, P1 = makes it good, P2 = stretch.

### P0

- [done] **REW-01 Data contract + `db.py`** — Backend. `events` table from Stage 4 with WAL, shared by detector and server; seed script with fake rows so UI work isn't blocked on the camera.
  *Done when* `python seed_fake.py` fills `rewind.db` and tests pass.
- [done] **REW-02 `server.py`: `/items`, `/where`, snapshots** — Backend. Stage 5 server with synonyms and spoken-style answers; routes declared above the `/` mount.
  *Done when* `/items` returns JSON for the seeded rows and `/where?q=where are my specs` answers about glasses.
- [done] **REW-03 `static/` UI** — Frontend. One big button, tiles with photo, Chrome speech in/out, typed fallback for Firefox, 24 px+ text.
  *Done when* tapping a tile speaks its answer and the button answers a spoken question.
- [done: kept the change gate, detect once the scene is still (see detector.py)] **REW-04 Decide: change gate vs continuous detection** — Vision + team. The guide's memory rule ("same zone 5 detections in a row") assumes ~10 detections/s; the change gate gives ~1 per move. Pick one: (a) keep the gate and confirm "placed" when the scene has *settled* (diff low for ~0.5 s) then detect, or (b) switch to the guide's continuous threaded detector.
  *Done when* the choice is written in CLAUDE.md.
- [done] **REW-05 `zones.py` + `zone_editor.py`** — Vision. Polygon zones, bottom-centre lookup, left/middle/right fallback.
  *Done when* moving an item between drawn zones changes its printed zone.
- [done] **REW-06 `memory.py` + `detector.py`** — Vision. Placed/gone rules, picked-up via person overlap, snapshot with people blurred, writes via `db.py`. Rename `phase1_camera_gate.py` → `detector.py`.
  *Done when* putting keys down adds a `placed` row and a JPEG in `snapshots/`.
- [done] **REW-07 Fix live view freezing** — Vision. `phase1_camera_gate.py:90` shows the last *annotated* frame forever after the first detection, so the window stops showing live video. Draw the last boxes onto the live frame instead.
  *Done when* video stays live between detections.
- [done; yolov8s-worldv2.pt now committed as the README says] **REW-08 Repo hygiene** — Backend. Add fastapi/uvicorn/CLIP to requirements; ignore `rewind.db*`, `snapshots/`, `.env`; fix the README weights claim (commit the `.pt` or remove the sentence); add `CLAUDE.md` (layout, events table, how to run).

### P1

- **REW-09 Detection quality** — Vision. Per-item confidence dict, drop oversized boxes, try synonym names ("keychain", "eyeglasses"), add `person` to classes; ArUco tags for keys/glasses if still unreliable after 2 h.
- [done] **REW-10 Privacy: snapshot auto-delete after 24 h** — Backend. Run on detector start and hourly.
- **REW-11 Dementia-friendly wording pass** — Frontend + Pitch. Calm, non-blaming answers; times as people say them; check the spoken answer stands alone.
- **REW-12 Test on the demo laptop** — All. Agree which laptop; check fps, camera mount, lighting.

### P2 (only after P0 works end to end)

- **REW-13 `/history` + timeline view** — Backend + Frontend (Stage 6).
- **REW-14 Flipbook of the moment** — Vision + Frontend (Stage 6).
- **REW-15 `describe.py` LLM spot descriptions, opt-in** — Pitch (Stage 7). API key via env/.env only.
- **REW-16 Odd-question matching via LLM** — Pitch (Stage 7).

### Pitch

- **REW-17 Demo script, slides, cited dementia stat, backup video, README screenshots** — Pitch (Stage 9).
