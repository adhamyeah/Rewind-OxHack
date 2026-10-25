"""
timeline.py - the Rewind timeline (Stage 6).

For one item it shows the last few places it was put down or picked up, with
photos and times, next to a replay of the seconds before the latest put-down
(people blurred). That replay is what the name "Rewind" promises.

Opens from the live camera window: say "show me where my glasses were" (or
"rewind my glasses"), or ask with a number key and then press r.
Or on its own:   python timeline.py glasses

Only the current run counts: each time Rewind starts, the timeline starts
empty (older runs stay in rewind.db but are never shown).
"""

import glob
import os
import sqlite3
import sys
import time
from datetime import datetime

import cv2
import numpy as np

ROWS = 5
WINDOW = "Rewind timeline"
BOARD_W, CLIP_W, CLIP_H = 430, 480, 360
BG, INK, QUIET, ACCENT = (32, 32, 32), (240, 240, 240), (170, 170, 170), (0, 165, 255)


def ascii_text(text):
    """OpenCV can only draw plain ASCII."""
    text = (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return text.encode("ascii", "ignore").decode()


def wrap(text, width):
    words, lines, line = ascii_text(text).split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def history(item, db_path="rewind.db", n=ROWS, since=0):
    """The item's latest events, newest first - only from `since` on (the start of this run)."""
    if not os.path.exists(db_path):
        return []
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT event, zone, description, at, image, clip, picked_up FROM events "
        "WHERE item = ? AND event IN ('placed', 'gone') AND at >= ? ORDER BY id DESC LIMIT ?",
        (item, since, n)).fetchall()
    db.close()
    return rows


def describe_row(row):
    from memory import place_phrase

    event, zone, description, at, _, _, picked_up = row
    when = datetime.fromtimestamp(at).strftime("%H:%M")
    if event == "placed":
        return when, "put down", description or place_phrase(zone)
    return when, "picked up" if picked_up else "out of view", f"was {description or place_phrase(zone)}"


class TimelineView:
    def __init__(self, item, db_path="rewind.db", seconds=25, since=0):
        self.item = item
        self.since = since
        rows = history(item, db_path, since=since)
        self.board = self._board(rows)
        # Replay the LATEST put-down only - never fall back to an older one.
        latest = next((r for r in rows if r[0] == "placed"), None)
        clip = latest[5] if latest and latest[5] and os.path.isdir(latest[5]) else None
        paths = sorted(glob.glob(os.path.join(clip, "*.jpg"))) if clip else []
        self.frames = [cv2.resize(cv2.imread(p), (CLIP_W, CLIP_H)) for p in paths]
        self.started = time.time()
        self.until = self.started + seconds

    def _board(self, rows):
        height = max(CLIP_H + 60, 70 + ROWS * 92)
        board = np.full((height, BOARD_W, 3), BG, np.uint8)
        cv2.putText(board, ascii_text(f"{self.item}: last places"), (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INK, 2)
        if not rows:
            started = f" (since {datetime.fromtimestamp(self.since):%H:%M})" if self.since else ""
            cv2.putText(board, ascii_text(f"Not seen yet this time{started}."), (16, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, QUIET, 1)
        for i, row in enumerate(rows):
            y = 60 + i * 92
            when, what, where = describe_row(row)
            image = row[4]
            thumb = cv2.imread(image) if image and os.path.exists(image) else None
            if thumb is not None:
                board[y:y + 72, 16:112] = cv2.resize(thumb, (96, 72))
            else:
                cv2.rectangle(board, (16, y), (111, y + 71), (70, 70, 70), -1)
                cv2.putText(board, "gone", (36, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, QUIET, 1)
            colour = ACCENT if i == 0 else INK
            cv2.putText(board, f"{when}  {what}", (124, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1 + (i == 0))
            for j, line in enumerate(wrap(where, 36)[:3]):
                cv2.putText(board, line, (124, y + 40 + 17 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, QUIET, 1)
        return board

    def frame(self):
        """The image to show now, or None once the view has timed out."""
        now = time.time()
        if now > self.until:
            return None
        h = self.board.shape[0]
        canvas = np.full((h, BOARD_W + CLIP_W + 32, 3), BG, np.uint8)
        canvas[:, :BOARD_W] = self.board
        x0, y0 = BOARD_W + 16, 52
        if self.frames:
            # play the clip at ~6 fps, hold the last frame for 1.5 s, then loop
            loop = len(self.frames) / 6 + 1.5
            i = min(int(((now - self.started) % loop) * 6), len(self.frames) - 1)
            canvas[y0:y0 + CLIP_H, x0:x0 + CLIP_W] = self.frames[i]
            label = "<< REWIND" if i < len(self.frames) - 1 else "put down here"
            cv2.putText(canvas, label, (x0, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT, 2)
        else:
            cv2.putText(canvas, "no replay saved for this one yet", (x0, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, QUIET, 1)
        return canvas


if __name__ == "__main__":
    from memory import latest_session_start

    name = " ".join(sys.argv[1:]) or "glasses"
    view = TimelineView(name, seconds=3600, since=latest_session_start())  # the most recent run only
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while True:
        img = view.frame()
        if img is None:
            break
        cv2.imshow(WINDOW, img)
        if cv2.waitKey(50) & 0xFF in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
