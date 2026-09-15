"""口型同步任务：把角色视频和配音音轨对齐（07_GENERATION.lipsync 的队列封装）。

在生产链里处于 video_task / tts_task 之后：有了镜头视频和对白语音，再把口型对上。
本任务只做参数透传，不做 QC / 落库（素材登记由 07 的 LipSyncGenerator 写 jsonl 账本），
provider 的 API key 缺失会在构造时就抛 NotConfiguredError，让配置错误尽早暴露。
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

from ..workers.celery_app import celery_app

__all__ = ["lipsync_task"]


@celery_app.task(name="lipsync_task", bind=True)
def lipsync_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """口型同步 GPU 队列任务（把角色视频与语音音轨对齐）。

    payload: {"video_path": str, "audio_path": str, "character_id": str | None,
      "episode_id": str | None, "shot_id": str | None, "seed": int | None,
      "output_dir": str | None, "asset_log_path": str | None, "license": str | None}。
    返回: {"file_path": str, "model": str}。

    直接调用 07_GENERATION.lipsync.lipsync_generator 里的真实类：LipSyncGenerator
    包装一个 BaseLipSyncProvider（SyncSoLipSyncProvider），后者在 SYNC_API_KEY 未设置时
    构造即抛 NotConfiguredError。
    """
    # 数字开头的顶层包只能用 importlib 按名字 import
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

    # model 返回 provider 类名：结果里能看出这条素材是哪家服务出的
    return {"file_path": file_path, "model": type(provider).__name__}
