"""
speak.py - Rewind's voice, using ElevenLabs text-to-speech.

Speaks in the background so the camera never waits. Plays audio with what
your computer already has (Windows: built-in winsound; Mac: afplay; Linux:
aplay), so no extra audio library is needed. Phrases it has said before are
remembered, so repeats are instant and don't use up ElevenLabs credits.
With no key it just prints what it would have said.

Setup (once):
  pip install -U elevenlabs
  add a line to your .env file:   ELEVENLABS_API_KEY=your-key
  optional, to pick a voice:      ELEVENLABS_VOICE_ID=the-voice-id
Check it works:   python speak.py "Your glasses are on the desk."
"""

import io
import os
import platform
import queue
import subprocess
import tempfile
import threading
import time
import wave

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"   # "George", a calm British voice. ELEVENLABS_VOICE_ID in .env overrides it
MODEL_ID = "eleven_flash_v2_5"      # ElevenLabs' fastest, cheapest model
SAMPLE_RATE = 22050                 # raw PCM at this rate, wrapped into a WAV here (works on every plan)


def load_key():
    """ELEVENLABS_API_KEY (and optional ELEVENLABS_VOICE_ID) from the environment or a .env file."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, ".env"), ".env"):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    name, sep, value = line.strip().partition("=")
                    name, value = name.strip(), value.strip().strip('"').strip("'")
                    if sep and name in ("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID") and value:
                        os.environ.setdefault(name, value)
    return os.environ.get("ELEVENLABS_API_KEY")


def api_error_text(e):
    """ElevenLabs errors start with a wall of headers; pull out the status code and the actual reason."""
    status, body = getattr(e, "status_code", None), getattr(e, "body", None)
    if status is None:
        return f"{type(e).__name__}: {str(e)[:300]}"
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        reason = detail.get("message") or detail.get("status") or detail
    elif isinstance(detail, list) and detail and isinstance(detail[0], dict):  # validation errors
        reason = "; ".join(str(d.get("msg", d)) for d in detail[:3])
    else:
        reason = detail
    return f"HTTP {status}: {str(reason)[:300]}"


def pcm_to_wav(pcm, rate=SAMPLE_RATE):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def play_wav(wav):
    """Play WAV bytes and wait until they finish."""
    system = platform.system()
    if system == "Windows":
        import winsound

        winsound.PlaySound(wav, winsound.SND_MEMORY)
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav)
        path = f.name
    try:
        players = [["afplay", path]] if system == "Darwin" else [["aplay", "-q", path], ["paplay", path]]
        for cmd in players:
            try:
                subprocess.run(cmd, check=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        print("[voice] no audio player found (aplay/paplay)")
    finally:
        os.remove(path)


class Voice:
    def __init__(self):
        self.client = None
        self.cache = {}        # text -> WAV bytes
        self.errors = 0
        self.jobs = queue.Queue(maxsize=3)  # if speech piles up, the oldest lines are dropped
        self.speaking = False
        self.last_spoke = 0.0

        key = load_key()
        if not key:
            print("[voice] no ELEVENLABS_API_KEY found (.env file) - answers will be printed, not spoken")
            return
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError:
            print("[voice] elevenlabs isn't installed: pip install -U elevenlabs - answers will be printed only")
            return
        self.client = ElevenLabs(api_key=key)
        self.voice_id = os.environ.get("ELEVENLABS_VOICE_ID") or VOICE_ID
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[voice] on (ElevenLabs {MODEL_ID}, voice {self.voice_id})")

    @property
    def enabled(self):
        return self.client is not None

    def busy(self):
        """True while Rewind is talking (and just after), so the microphone doesn't hear itself."""
        return self.speaking or not self.jobs.empty() or time.time() - self.last_spoke < 0.8

    def say(self, text):
        print(f"[voice] {text}")
        if not self.enabled:
            return
        while True:
            try:
                self.jobs.put_nowait(text)
                return
            except queue.Full:
                try:
                    self.jobs.get_nowait()  # drop the oldest line rather than fall behind
                except queue.Empty:
                    pass

    def synthesize(self, text):
        if text not in self.cache:
            chunks = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                model_id=MODEL_ID,
                output_format=f"pcm_{SAMPLE_RATE}",
                text=text,
            )
            self.cache[text] = pcm_to_wav(b"".join(chunks))
        return self.cache[text]

    def _loop(self):
        while True:
            text = self.jobs.get()
            self.speaking = True
            try:
                play_wav(self.synthesize(text))
            except Exception as e:
                self.errors += 1
                if self.errors <= 3:  # don't flood the terminal
                    print(f"[voice] failed - {api_error_text(e)}")
            finally:
                self.speaking = False
                self.last_spoke = time.time()


if __name__ == "__main__":
    import sys
    import time

    line = " ".join(sys.argv[1:]) or "Hello, I'm Rewind. Your glasses are on the desk, next to the laptop."
    voice = Voice()
    if not voice.enabled:
        raise SystemExit(1)
    started = time.perf_counter()
    try:
        wav = voice.synthesize(line)
    except Exception as e:
        msg = api_error_text(e)
        print(f"FAILED: {msg}")
        if "401" in msg or "invalid_api_key" in msg or "Unauthorized" in msg:
            print("-> Key problem: check ELEVENLABS_API_KEY in .env was copied fully.")
        elif "quota" in msg.lower() or "credits" in msg.lower():
            print("-> Out of ElevenLabs credits for this month.")
        elif "voice" in msg.lower() and "not found" in msg.lower():
            print("-> That voice isn't available to your key: add it to 'My Voices' on the ElevenLabs site, "
                  "or remove ELEVENLABS_VOICE_ID from .env to use the default voice.")
        raise SystemExit(1)
    print(f"OK: got {len(wav) // 1024} KB of audio in {time.perf_counter() - started:.1f}s - playing it now")
    play_wav(wav)
