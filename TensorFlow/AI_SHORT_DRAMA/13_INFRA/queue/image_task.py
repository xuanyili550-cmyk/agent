"""单张图像生成任务：调 07_GENERATION 的图像生成器出一张图并登记素材。

和 shot_task 的区别：这里是"给一个 prompt 出一张图"的原子任务，不做 QC、不走重试阶梯，
给 API 手工生图 / 角色定妆图 / 调试用；正式的镜头生产（图 -> QC -> 三级重试 -> 落库）走 shot_task。
两者共用 ``resolve_character_assets`` 解析角色参考图和 LoRA，保证角色一致性逻辑只有一份。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..config import get_settings
from ..database import repository as repo
from ..database.session import session_scope
from ..workers.celery_app import celery_app
from ._common import mod
from .shot_task import resolve_character_assets

__all__ = ["image_task"]


@celery_app.task(name="image_task", bind=True)
def image_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """单张图像生成任务（不带 QC 与重试阶梯；带这些的镜头生产走 shot_task）。

    payload: {"prompt": str, "negative_prompt": str | None, "character_id": str | None,
      "reference_character_ids": list[str] | None, "reference_images": list[str] | None,
      "lora_path": str | None, "episode_id": str | None, "shot_id": str | None, "seed": int | None,
      "model": str | None (diffusers 模型 id/路径；不传或 "dummy" 用离线 DummyImageGenerator),
      "width": int, "height": int, "output_dir": str | None, "license": str | None}
    返回: {"file_path": str, "model": str, "seed": int | None, "asset_id": str}

    角色一致性：reference_character_ids 会去 characters 表 / 06_MODELS/character_loras.json 查
    参考图和 LoRA，传给 DiffusersImageGenerator 的 IP-Adapter / LoRA；payload 里显式给的优先。
    AssetRecord 同时写 jsonl（07 的本地账本）和 assets 表。
    """
    settings = get_settings()
    image_generator_mod = mod("07_GENERATION.image.image_generator")
    asset_registry = mod("07_GENERATION.asset_registry")

    # 后端选择：payload 指定 > settings.image_backend；"dummy" 走离线占位生成器（测试/无 GPU 环境）
    model = payload.get("model") or settings.image_backend
    if model and model != "dummy":
        generator = image_generator_mod.DiffusersImageGenerator(model_id_or_path=model)
    else:
        generator = image_generator_mod.DummyImageGenerator()

    # 没给 reference_character_ids 时退回用 character_id 单个角色
    character_ids = list(payload.get("reference_character_ids") or ([payload["character_id"]] if payload.get("character_id") else []))
    with session_scope() as db:
        assets = resolve_character_assets(character_ids, db, project_id=payload.get("project_id"))
    reference_images = payload.get("reference_images") or assets["reference_images"] or None
    lora_path = payload.get("lora_path") or assets["lora_path"]

    seed = payload.get("seed")
    image = generator.generate(
        prompt=payload["prompt"],
        negative_prompt=payload.get("negative_prompt"),
        seed=seed,
        width=payload.get("width", 1024),
        height=payload.get("height", 1024),
        lora_path=lora_path,
        reference_images=reference_images,
    )

    output_dir = Path(payload.get("output_dir") or settings.artifacts_root / "images")
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_id = asset_registry.new_asset_id("img")
    # 文件名优先用 shot_id，方便人在产物目录里按镜头找图；没有镜头时用素材 id
    file_name = f"{payload.get('shot_id') or asset_id}.png"
    file_path = output_dir / file_name
    image.save(file_path)

    record = asset_registry.AssetRecord(
        asset_id=asset_id,
        file_path=str(file_path),
        character_id=payload.get("character_id") or (character_ids[0] if character_ids else None),
        episode_id=payload.get("episode_id"),
        shot_id=payload.get("shot_id"),
        model=model or "dummy-placeholder-generator",
        prompt=payload["prompt"],
        seed=seed,
        license=payload.get("license"),
        created_at=asset_registry.now_iso(),
    )
    # 双写：jsonl 账本是 07_GENERATION 离线也能用的记录，assets 表供 API / QC 关联查询
    asset_log_path = payload.get("asset_log_path") or str(output_dir.parent / "asset_records.jsonl")
    asset_registry.write_asset_record(record, asset_log_path)
    with session_scope() as db:
        repo.persist_asset(db, record, kind="image", width=payload.get("width", 1024), height=payload.get("height", 1024))

    return {
        "file_path": str(file_path),
        "model": record.model,
        "seed": seed,
        "asset_id": asset_id,
        "reference_images": len(reference_images or []),
        "lora_path": lora_path,
    }
