from __future__ import annotations

import importlib
from typing import Any, Dict

from ..workers.celery_app import celery_app

__all__ = ["video_task"]


@celery_app.task(name="video_task", bind=True)
def video_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Video-generation GPU queue task (image-to-video / text-to-video for a Shot).

    Expected payload: {"prompt": str, "source_image": str | None, "shot_id": str | None,
      "character_id": str | None, "episode_id": str | None, "duration_sec": float,
      "seed": int | None, "provider": str | None ("runway" 默认 | "pika" | "mochi" 本地免费),
      "output_dir": str | None, "asset_log_path": str | None, "license": str | None}.
    Expected return: {"file_path": str, "model": str, "seed": int | None}.

    Calls the real 07_GENERATION.video.video_generator classes: VideoGenerator wraps a
    BaseVideoProvider (RunwayVideoProvider or PikaVideoProvider), each of which raises
    NotConfiguredError at construction time if its API key env var is unset -- that
    error propagates so a misconfigured worker fails loudly instead of silently.
    """
    video_generator_mod = importlib.import_module("07_GENERATION.video.video_generator")

    provider_name = (payload.get("provider") or "runway").lower()
    if provider_name == "pika":
        provider = video_generator_mod.PikaVideoProvider()
    elif provider_name == "mochi":
        provider = video_generator_mod.MochiLocalVideoProvider()
    else:
        provider = video_generator_mod.RunwayVideoProvider()

    generator = video_generator_mod.VideoGenerator(
        provider=provider,
        asset_log_path=payload.get("asset_log_path") or "./outputs/asset_records.jsonl",
    )
    seed = payload.get("seed")
    file_path = generator.generate(
        prompt=payload["prompt"],
        image_path=payload.get("source_image"),
        duration_sec=payload.get("duration_sec", 4.0),
        seed=seed,
        output_dir=payload.get("output_dir") or "./outputs/videos",
        character_id=payload.get("character_id"),
        episode_id=payload.get("episode_id"),
        shot_id=payload.get("shot_id"),
        license=payload.get("license"),
    )

    return {"file_path": file_path, "model": type(provider).__name__, "seed": seed}
