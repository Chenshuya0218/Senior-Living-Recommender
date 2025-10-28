from faster_whisper import WhisperModel
from pathlib import Path

def transcribe_audio(audio_path: str, model_size: str = "base"):
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    segments, info = model.transcribe(audio_path, vad_filter=True, language="en")
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip(), list(segments)
