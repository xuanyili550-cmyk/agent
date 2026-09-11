from __future__ import annotations

import importlib
from typing import Any, Dict

from ..workers.celery_app import celery_app

__all__ = ["tts_task"]


@celery_app.task(name="tts_task", bind=True)
def tts_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Text-to-speech GPU queue task (dialogue line -> voice audio).

    Expected payload: {"text": str, "voice_id": str, "character_id": str | None,
      "episode_id": str | None, "shot_id": str | None, "seed": int | None,
      "output_dir": str | None, "asset_log_path": str | None, "license": str | None}.
    Expected return: {"file_path": str, "model": str}.

    Calls the real 07_GENERATION.audio.tts_generator classes: TTSGenerator wraps a
    BaseTTSProvider (ElevenLabsTTSProvider), which raises NotConfiguredError at
    construction time if ELEVENLABS_API_KEY is unset. voice_id must be a provider-side
    voice id (see 07_GENERATION.voice.voice_cloner.VoiceCloner for enrollment, a
    separate one-time step from per-line synthesis).
    """
    tts_generator_mod = importlib.import_module("07_GENERATION.audio.tts_generator")

    provider = tts_generator_mod.ElevenLabsTTSProvider()
    generator = tts_generator_mod.TTSGenerator(
        provider=provider,
        asset_log_path=payload.get("asset_log_path") or "./outputs/asset_records.jsonl",
    )
    file_path = generator.synthesize(
        text=payload["text"],
        voice_id=payload["voice_id"],
        output_dir=payload.get("output_dir") or "./outputs/audio",
        seed=payload.get("seed"),
        character_id=payload.get("character_id"),
        episode_id=payload.get("episode_id"),
        shot_id=payload.get("shot_id"),
        license=payload.get("license"),
    )

    return {"file_path": file_path, "model": type(provider).__name__}
