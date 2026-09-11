"""
Local, free, open-weight speech-to-text via Whisper (Apache-2.0/MIT, no API
key). Output is adapted to the same {"segments": [{"start","end","text"}]}
shape that 09_POST/subtitle/subtitle.py's cues_from_whisper_result() expects,
so ASR output can be burned into subtitles without going through a paid API.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseASRProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Returns {"text": str, "segments": [{"start": float, "end": float, "text": str}, ...]}."""


class WhisperLocalASRProvider(BaseASRProvider):
    """Runs openai/whisper-large-v3 (or any Whisper checkpoint) locally via
    transformers -- no API key, needs a GPU for reasonable speed on longer
    clips (CPU works but is slow). See 06_MODELS/asr/registry.json."""

    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        from transformers import pipeline

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            return_timestamps=True,
        )
        return self._pipe

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        pipe = self._load()
        generate_kwargs = {"language": language} if language else {}
        result = pipe(audio_path, generate_kwargs=generate_kwargs)
        segments = [
            {"start": chunk["timestamp"][0], "end": chunk["timestamp"][1], "text": chunk["text"]}
            for chunk in result.get("chunks", [])
            if chunk.get("timestamp", (None, None))[1] is not None
        ]
        return {"text": result.get("text", ""), "segments": segments}
