"""
answer.py - one place that answers "where is my ...?", however it was asked:
typed on the web page, the page's Talk button, said out loud (listen.py, always on),
or typed in the terminal.

  question -> which item (word match first, Gemini if the phrasing is odd)
           -> memory.explain(): the answer, and the moment it was put down (or taken)
           -> the page jumps to that moment and plays its rewind gif   (screen first)
           -> Rewind says the answer out loud on its own thread          (never delays the screen)
"""

import re

from listen import find_item, parse_question


def strip_wake_word(text):
    """'Rewind, where are...' / 'Hey Rewind ...' is just calling its name; 'rewind my wallet' is a request."""
    return re.sub(r"^\W*((hey|ok|okay)\s+rewind\b\W*|rewind\s*[,.!?:]+\s*)", "", text, flags=re.I)


class Answerer:
    def __init__(self, memory, voice, ui_state, items, wake_word=False):
        self.memory, self.voice, self.ui, self.items = memory, voice, ui_state, list(items)
        self.wake_word = wake_word  # noisy room: the always-on mic only answers "Rewind, ..."

    def _item(self, text):
        item = find_item(text, self.items)
        describer = self.memory.describer
        if not item and describer and describer.enabled:
            item = describer.match_item(text)  # odd phrasing: let Gemini work out the item
        return item

    def ask(self, question, heard=None, source="typed", item=None):
        """Answer one question (typed, Talk button, or a transcript). -> {"text", "jump", "heard"}
        item: already worked out (skip the lookup), or False = we already know it's none of ours."""
        body = strip_wake_word(question).strip()
        item = item or (self._item(body) if item is None else None)
        if not item:
            text = (f"Sorry, I can only look for your {', '.join(self.items[:-1])}, or {self.items[-1]}."
                    if body else "Sorry, I didn't catch a question.")
            jump = None
        else:
            text, jump = self.memory.explain(item)
            if jump:
                self.ui.request_jump(jump, caption=text)  # screen first
        print(f"[answer] {text}")
        self.ui.set_answer(text, heard, source)
        self.voice.say(text)
        return {"text": text, "jump": jump, "heard": heard}

    def heard(self, text):
        """A transcript from the always-on microphone: only answer things that sound like questions,
        so chatter in the room is ignored."""
        if self.wake_word and "rewind" not in text.lower():
            return
        body = strip_wake_word(text)
        item, question, rewind = parse_question(body, self.items)
        short = len(body.split()) <= 3  # just "glasses?" counts as a question
        if not item and (question or rewind):
            item = self._item(body)
        if item and (question or rewind or short):
            self.ask(body, heard=text, source="voice", item=item)
        elif question and any(w in body.lower() for w in ("where", "find", "lost")):
            self.ask(body, heard=text, source="voice", item=False)  # "Sorry, I can only look for ..."
