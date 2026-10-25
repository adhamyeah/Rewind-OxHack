"""
Answerer: question -> item label -> Memory.where() -> spoken answer + window jump.

Order matters: the window jump is requested FIRST, voice is generated on its own thread,
so a slow (or broken) ElevenLabs never delays the screen.
"""

import os
import subprocess
import sys
import threading
import time
import wave

import config
from gemini_detector import gemini_pick_label, normalise_label
from vocab import VOCAB


def play_wav(path):
    """Play a wav with whatever the OS has (afplay on Mac, winsound on Windows, aplay on Linux)."""
    if sys.platform == "darwin":
        subprocess.run(["afplay", path], check=False)
    elif sys.platform == "win32":
        import winsound

        winsound.PlaySound(path, winsound.SND_FILENAME)
    else:
        subprocess.run(["aplay", "-q", path], check=False)


def speak_elevenlabs(text):
    """Text -> ElevenLabs PCM -> WAV file -> play. Raises on failure (caller prints and carries on)."""
    if not (config.ELEVENLABS_API_KEY and config.ELEVENLABS_VOICE_ID):
        raise RuntimeError("ElevenLabs key or voice id missing")
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
    audio = client.text_to_speech.convert(voice_id=config.ELEVENLABS_VOICE_ID, text=text,
                                          output_format="pcm_16000")
    pcm = b"".join(audio)
    with wave.open(config.ANSWER_WAV, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    play_wav(config.ANSWER_WAV)


class Answerer:
    def __init__(self, memory, ui_state, speak=speak_elevenlabs, pick_label=gemini_pick_label,
                 ready=lambda: True, vocab=None):
        self.memory = memory
        self.ui = ui_state
        self.speak = speak
        self.pick_label = pick_label
        self.ready = ready  # True when memory is up to date with the table
        self.vocab = vocab if vocab is not None else (VOCAB if config.OPEN_VOCAB else None)

    def ask(self, question):
        """Answer one question. Returns the answer text."""
        # any name we track (including ones learned from Gemini), else let Gemini work it out
        find = self.vocab.find_in_question if self.vocab is not None else normalise_label
        label = find(question) or self.pick_label(question)
        if label is None:
            text = "Sorry, I didn't catch which thing you mean. Try 'where are my keys?'"
            print(f"[answer] {text}")
            self._say(text)
            return text
        self._wait_until_ready()
        answer = self.memory.where(label)
        if answer.keyframe is not None:
            self.ui.request_jump(answer.keyframe, answer.box, label, answer.text)  # screen first
        print(f"[answer] {answer.text}")
        self._say(answer.text)
        return answer.text

    def _wait_until_ready(self):
        """If the table just changed, give memory a few seconds to catch up before answering."""
        end = time.time() + config.ASK_WAIT_SECONDS
        if not self.ready():
            print("[answer] memory is catching up with the table...")
        while not self.ready() and time.time() < end:
            time.sleep(0.1)

    def _say(self, text):
        def run():
            try:
                self.speak(text)
            except Exception as e:
                print(f"[voice] skipped ({e})")

        threading.Thread(target=run, daemon=True).start()
