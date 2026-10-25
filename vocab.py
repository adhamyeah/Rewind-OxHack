"""
Vocabulary: the names Rewind tracks, all of them predicted by Gemini.

It starts EMPTY: nothing is assumed to be on the table. Each Gemini keyframe names what
it actually sees, and canonical() keeps those names consistent from look to look:
  "coffee cup" / "blue mug" -> "mug"      (config.SYNONYMS, and "<colour> <name>" of known names)
  "A Phone Charger."        -> "phone charger"   (a new thing, tidied)
  "person" / "white wall"   -> None              (people and the room are never items)
config.ITEMS are only used for that consistency, never as a list of things to look for.
Every predicted name is learned, and YOLO-World looks for exactly those (see live_detector),
so the live view tracks what Gemini found between keyframes. Only the MAX_LEARNED most
recent names are kept: YOLO-World gets less reliable with a very long list.
"""

import re
import threading

import config

_ARTICLES = re.compile(r"^(my|your|a|an|the|some|this|that)\s+")


def tidy(text):
    """'  A Phone-Charger.' -> 'a phone-charger' -> 'phone-charger'."""
    text = re.sub(r"[^a-z0-9 &'-]", " ", str(text or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    return _ARTICLES.sub("", text)


class Vocabulary:
    def __init__(self, base=None, synonyms=None, max_learned=None):
        self.base = list(config.ITEMS if base is None else base)
        self.synonyms = config.SYNONYMS if synonyms is None else synonyms
        self.max_learned = config.MAX_LEARNED if max_learned is None else max_learned
        self.learned = []        # every name Gemini predicted, newest last
        self.version = 0         # goes up whenever names() changes, so YOLO-World can refresh
        self.lock = threading.Lock()

    def names(self):
        """The names Gemini has actually predicted (most recent MAX_LEARNED)."""
        with self.lock:
            return list(self.learned)

    def _phrases(self):
        """(phrase, name) for every way to say a tracked name, longest phrase first."""
        with self.lock:
            pairs = [(s, item) for item, syns in self.synonyms.items() for s in syns]
            pairs += [(n, n) for n in self.base + self.learned]
        return sorted(pairs, key=lambda p: -len(p[0]))

    def canonical(self, label):
        """Gemini's name for a thing -> the name we track, or None if it isn't an item."""
        text = tidy(label)
        if not text:
            return None
        for phrase, name in self._phrases():
            # whole name, plural, or the end of a longer one ("blue mug", "car keys")
            if text in (phrase, phrase + "s") or text.endswith(" " + phrase) or text.endswith(" " + phrase + "s"):
                return name
        if text in config.NOT_ITEMS or text.split()[-1] in config.NOT_ITEMS:
            return None
        return text

    def find_in_question(self, question):
        """'where's my red stapler?' -> 'stapler', if it's a name (or synonym) we track."""
        text = (question or "").lower()
        for phrase, name in self._phrases():
            # \b keeps "pen" from matching "open"; the optional s allows plurals
            if re.search(rf"\b{re.escape(phrase)}s?\b", text):
                return name
        return None

    def learn(self, labels):
        """Remember names Gemini predicted. Returns the ones that were new."""
        new = []
        with self.lock:
            before = list(self.learned)
            for label in labels:
                if not label:
                    continue
                if label in self.learned:
                    self.learned.remove(label)   # seen again: now the most recent
                else:
                    new.append(label)
                self.learned.append(label)
            self.learned = self.learned[-self.max_learned:]
            if sorted(self.learned) != sorted(before):
                self.version += 1
        return new


VOCAB = Vocabulary()     # the one shared by the running app
