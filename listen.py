"""Speech-to-text: audio recorded in the browser -> ElevenLabs -> text. Raises on failure (the caller handles it)."""

import io

import config

# Browsers record webm/opus (Chrome, Firefox) or mp4/aac (Safari); ElevenLabs detects the format itself.
EXTENSIONS = {"webm": "webm", "mp4": "mp4", "ogg": "ogg", "wav": "wav", "mpeg": "mp3"}


def transcribe(audio_bytes, content_type="audio/webm"):
    if not config.ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is missing")
    from elevenlabs.client import ElevenLabs

    kind = content_type.split(";")[0].split("/")[-1].lower()
    name = f"speech.{EXTENSIONS.get(kind, 'webm')}"
    client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
    result = client.speech_to_text.convert(model_id=config.ELEVENLABS_STT_MODEL,
                                           file=(name, io.BytesIO(audio_bytes), content_type),
                                           language_code="en")
    return result.text or ""
