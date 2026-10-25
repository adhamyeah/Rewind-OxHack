"""
listen.py - hands-free questions (Stage 5): just say "where are my glasses?"

The microphone is watched on this computer. Only when someone actually speaks
is that short clip sent to ElevenLabs speech-to-text; silence never leaves the
laptop. While Rewind is talking (or someone is holding the page's Talk button),
the mic is ignored so it doesn't answer itself or answer twice.

The web page's Talk button records in the browser instead; that audio comes
here too (Listener.transcribe_upload), so both use the same speech-to-text.

Setup (once):
  pip install -U sounddevice elevenlabs
  ELEVENLABS_API_KEY in .env, with "Speech to Text" access switched on for the key
Check it works:   python listen.py      then say "where are my keys?"
"""

import re
import threading
import time
from collections import deque

import numpy as np

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.03
START_FACTOR = 3.0          # speech = this many times louder than the room's background noise...
MIN_LEVEL = 300             # ...and at least this loud (16-bit audio), so faint hiss never triggers it
END_SILENCE = 0.8           # this long quiet = the sentence is over
MIN_SPEECH, MAX_SPEECH = 0.4, 8.0
PRE_ROLL = 0.3              # keep a little audio from just before the voice started
MAX_CLIPS_PER_MINUTE = 10   # cap on clips sent to ElevenLabs (saves credits in a noisy room)
STT_MODEL = "scribe_v2"
FALLBACK_STT_MODEL = "scribe_v1"   # used automatically if ElevenLabs won't accept scribe_v2 for this key

# Other words people use for each item. Add your own.
SYNONYMS = {
    "keys": ["keys", "key", "keyring", "key ring"],
    "glasses": ["glasses", "specs", "spectacles", "eyeglasses", "sunglasses"],
    "wallet": ["wallet", "purse", "card holder"],
    "phone": ["phone", "mobile", "cellphone", "cell phone", "iphone", "smartphone"],
    "mug": ["mug", "cup", "coffee", "tea"],
    "headphones": ["headphones", "earphones", "headset", "earbuds", "airpods"],
    "pen": ["pen", "pencil", "biro"],
    "water bottle": ["water bottle", "bottle", "flask"],
}
QUESTION_WORDS = ("where", "find", "lost", "seen", "left", "put", "missing", "rewind", "show", "looking for")
# Browsers record webm/opus (Chrome, Firefox) or mp4/aac (Safari); ElevenLabs works the format out itself.
UPLOAD_EXTENSIONS = {"webm": "webm", "mp4": "mp4", "ogg": "ogg", "wav": "wav", "mpeg": "mp3"}
REWIND_WORDS = ("show", "rewind", "timeline", "history", "replay")


def find_item(text, items):
    """The first item mentioned in the text, by name or by another word for it."""
    low = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
    hits = []
    for item in items:
        for word in sorted(set(SYNONYMS.get(item, []) + [item]), key=len, reverse=True):
            at = low.find(f" {word} ")
            if at >= 0:
                hits.append((at, item))
                break
    return min(hits)[1] if hits else None


def parse_question(text, items):
    """-> (item or None, is it a question?, do they want the rewind timeline?)"""
    low = text.lower()
    return (find_item(text, items),
            any(word in low for word in QUESTION_WORDS),
            any(word in low for word in REWIND_WORDS))


class Listener:
    def __init__(self, on_text, is_busy=lambda: False):
        self.on_text = on_text      # called with each transcript, on a background thread
        self.is_busy = is_busy      # True while Rewind is talking
        self.client = None
        self.sent = deque()
        self.errors = 0
        self.level = 0.0
        self.model = STT_MODEL
        self.language = "eng"
        self.sd = None
        self.listening = False      # True while the always-on microphone is running

        from speak import load_key

        key = load_key()
        if not key:
            print("[listen] no ELEVENLABS_API_KEY in .env - voice questions are off (typing still works)")
            return
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as e:
            print(f"[listen] elevenlabs isn't installed ({e}): pip install -U elevenlabs - voice questions are off")
            return
        self.client = ElevenLabs(api_key=key)
        try:
            import sounddevice

            self.sd = sounddevice
        except (ImportError, OSError) as e:
            print(f"[listen] can't use the microphone here ({e}): pip install -U sounddevice "
                  "- hands-free questions are off (the page's Talk button still works)")

    @property
    def enabled(self):
        return self.client is not None

    def start(self):
        if self.enabled and self.sd is not None:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        block = int(SAMPLE_RATE * BLOCK_SECONDS)
        try:
            with self.sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=block) as stream:
                self.listening = True
                self.listen(lambda: stream.read(block)[0][:, 0])
        except Exception as e:
            print(f"[listen] microphone problem ({type(e).__name__}): {e} - hands-free questions are off")
        self.listening = False

    def listen(self, read_block):
        """read_block() -> one block of int16 samples. Runs forever."""
        rms = lambda x: float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
        noise = float(np.median([rms(read_block()) for _ in range(int(1.0 / BLOCK_SECONDS))]))  # 1 s of room sound
        print("[listen] listening - just ask, e.g. \"where are my glasses?\"")
        before = deque(maxlen=int(PRE_ROLL / BLOCK_SECONDS))
        speech, talking, loud_run, quiet = [], False, 0, 0.0
        while True:
            samples = read_block()
            level = self.level = rms(samples)
            if self.is_busy():  # Rewind is talking: don't listen to ourselves
                speech, talking, loud_run = [], False, 0
                before.clear()
                continue
            threshold = max(noise * START_FACTOR, MIN_LEVEL)
            if not talking:
                before.append(samples)
                if level > threshold:
                    loud_run += 1
                else:
                    loud_run = 0
                    noise = 0.98 * noise + 0.02 * level  # follow slow changes in room noise
                if loud_run >= 3:  # ~0.1 s of voice: someone started talking
                    talking, speech, quiet = True, list(before), 0.0
                continue
            speech.append(samples)
            quiet = quiet + BLOCK_SECONDS if level < threshold * 0.6 else 0.0
            length = len(speech) * BLOCK_SECONDS
            if quiet >= END_SILENCE or length >= MAX_SPEECH:
                talking, loud_run = False, 0
                before.clear()
                if length - quiet >= MIN_SPEECH and self._allowed():
                    clip = np.concatenate(speech)
                    threading.Thread(target=self.transcribe, args=(clip,), daemon=True).start()

    def _allowed(self):
        now = time.time()
        while self.sent and now - self.sent[0] > 60:
            self.sent.popleft()
        if len(self.sent) >= MAX_CLIPS_PER_MINUTE:
            return False
        self.sent.append(now)
        return True

    def _speech_to_text(self, file):
        """One ElevenLabs speech-to-text call. Switches model or language if ElevenLabs refuses them; raises otherwise."""
        from speak import api_error_text

        for _ in range(3):
            try:
                options = {"language_code": self.language} if self.language else {}
                result = self.client.speech_to_text.convert(file=file, model_id=self.model,
                                                            tag_audio_events=False, **options)
                return (result.text or "").strip()
            except Exception as e:
                reason = api_error_text(e)
                if "model" in reason.lower() and self.model != FALLBACK_STT_MODEL:
                    print(f"[listen] {self.model} not accepted ({reason}) - switching to {FALLBACK_STT_MODEL}")
                    self.model = FALLBACK_STT_MODEL
                    continue
                if "language" in reason.lower() and self.language:
                    self.language = None  # let ElevenLabs detect the language itself
                    continue
                raise
        return ""

    def _report(self, e):
        from speak import api_error_text

        reason, status = api_error_text(e), getattr(e, "status_code", None)
        self.errors += 1
        if self.errors <= 3:
            hint = ""
            if status in (401, 403):
                hint = ("\n  -> ElevenLabs refused this key for speech-to-text. Make sure the key in .env is "
                        "the one you switched Speech to Text on for (ElevenLabs only shows a key once, "
                        "so an older key may still be in .env).")
            elif status == 429:
                hint = "\n  -> Too many requests, or out of ElevenLabs credits."
            print(f"[listen] speech-to-text failed - {reason}{hint}")

    def transcribe(self, samples):
        """A clip from the always-on microphone -> text -> on_text (background thread)."""
        from speak import pcm_to_wav

        wav = pcm_to_wav(samples.astype(np.int16).tobytes(), SAMPLE_RATE)
        try:
            text = self._speech_to_text(("question.wav", wav, "audio/wav"))
        except Exception as e:
            self._report(e)
            return
        if text and not re.fullmatch(r"[\W_]*(\(.*\)|\[.*\])?[\W_]*", text):  # skip "(silence)" and the like
            print(f'[heard] "{text}"')
            self.on_text(text)

    def transcribe_upload(self, audio, content_type="audio/webm"):
        """Audio recorded by the web page's Talk button -> text. Raises if it can't (the page shows a message)."""
        if not self.enabled:
            raise RuntimeError("speech-to-text is off (no ELEVENLABS_API_KEY or elevenlabs not installed)")
        kind = content_type.split(";")[0].split("/")[-1].lower()
        try:
            text = self._speech_to_text((f"speech.{UPLOAD_EXTENSIONS.get(kind, 'webm')}", audio, content_type))
        except Exception as e:
            self._report(e)
            raise
        return "" if re.fullmatch(r"[\W_]*(\(.*\)|\[.*\])?[\W_]*", text) else text

if __name__ == "__main__":
    items = list(SYNONYMS)

    def show(text):
        item, question, rewind = parse_question(text, items)
        print(f"  -> item: {item}, question: {question}, rewind: {rewind}")

    listener = Listener(show)
    if not listener.enabled or listener.sd is None:
        raise SystemExit(1)
    try:
        print(f"[listen] microphone: {listener.sd.query_devices(kind='input')['name']}")
    except Exception:
        pass
    print("[listen] test runs for 60 seconds - press Ctrl+C to stop sooner. "
          "The level should jump into the hundreds or thousands when you talk.")
    listener.start()
    try:
        for _ in range(120):
            time.sleep(0.5)
            print(f"\r  mic level {listener.level:7.0f}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
