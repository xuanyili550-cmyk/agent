from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict

from ..workers.celery_app import celery_app

__all__ = ["image_task"]


@celery_app.task(name="image_task", bind=True)
def image_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Image-generation GPU queue task.

    Expected payload: {"prompt": str, "negative_prompt": str | None,
      "character_id": str | None, "episode_id": str | None, "shot_id": str | None,
      "seed": int | None, "model": str | None (a diffusers model id/path; omit or
      pass "dummy" to use the offline DummyImageGenerator), "width": int,
      "height": int, "output_dir": str | None, "license": str | None}.
    Expected return: {"file_path": str, "model": str, "seed": int | None, "asset_id": str}.

    Calls the real 07_GENERATION.image.image_generator classes -- DiffusersImageGenerator
    when payload["model"] names a real model id/path, DummyImageGenerator otherwise -- and
    appends an AssetRecord via 07_GENERATION.asset_registry, mirroring 07_GENERATION/demo.py.
    """
    image_generator_mod = importlib.import_module("07_GENERATION.image.image_generator")
    asset_registry = importlib.import_module("07_GENERATION.asset_registry")

    model = payload.get("model")
    if model and model != "dummy":
        generator = image_generator_mod.DiffusersImageGenerator(model_id_or_path=model)
    else:
        generator = image_generator_mod.DummyImageGenerator()

    seed = payload.get("seed")
    image = generator.generate(
        prompt=payload["prompt"],
        negative_prompt=payload.get("negative_prompt"),
        seed=seed,
        width=payload.get("width", 1024),
        height=payload.get("height", 1024),
    )

    output_dir = Path(payload.get("output_dir") or "./outputs/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_id = asset_registry.new_asset_id("img")
    file_name = f"{payload.get('shot_id') or asset_id}.png"
    file_path = output_dir / file_name
    image.save(file_path)

    record = asset_registry.AssetRecord(
        asset_id=asset_id,
        file_path=str(file_path),
        character_id=payload.get("character_id"),
        episode_id=payload.get("episode_id"),
        shot_id=payload.get("shot_id"),
        model=model or "dummy-placeholder-generator",
        prompt=payload["prompt"],
        seed=seed,
        license=payload.get("license"),
        created_at=asset_registry.now_iso(),
    )
    asset_log_path = payload.get("asset_log_path") or str(output_dir.parent / "asset_records.jsonl")
    asset_registry.write_asset_record(record, asset_log_path)

    return {"file_path": str(file_path), "model": record.model, "seed": seed, "asset_id": asset_id}
