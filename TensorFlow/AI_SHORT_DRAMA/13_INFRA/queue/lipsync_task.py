from __future__ import annotations

import importlib
from typing import Any, Dict

from ..workers.celery_app import celery_app

__all__ = ["lipsync_task"]


@celery_app.task(name="lipsync_task", bind=True)
def lipsync_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Lipsync GPU queue task (aligns a character video with a voice track).

    Expected payload: {"video_path": str, "audio_path": str, "character_id": str | None,
      "episode_id": str | None, "shot_id": str | None, "seed": int | None,
      "output_dir": str | None, "asset_log_path": str | None, "license": str | None}.
    Expected return: {"file_path": str, "model": str}.

    Calls the real 07_GENERATION.lipsync.lipsync_generator classes: LipSyncGenerator
    wraps a BaseLipSyncProvider (SyncSoLipSyncProvider), which raises
    NotConfiguredError at construction time if SYNC_API_KEY is unset.
    """
    lipsync_generator_mod = importlib.import_module("07_GENERATION.lipsync.lipsync_generator")

    provider = lipsync_generator_mod.SyncSoLipSyncProvider()
    generator = lipsync_generator_mod.LipSyncGenerator(
        provider=provider,
        asset_log_path=payload.get("asset_log_path") or "./outputs/asset_records.jsonl",
    )
    file_path = generator.sync(
        video_path=payload["video_path"],
        audio_path=payload["audio_path"],
        output_dir=payload.get("output_dir") or "./outputs/lipsync",
        seed=payload.get("seed"),
        character_id=payload.get("character_id"),
        episode_id=payload.get("episode_id"),
        shot_id=payload.get("shot_id"),
        license=payload.get("license"),
    )

    return {"file_path": file_path, "model": type(provider).__name__}
