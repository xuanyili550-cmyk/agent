"""视频生成任务：给一个镜头做图生视频 / 文生视频（07_GENERATION.video 的队列封装）。

在生产链里接在 shot_task 之后：shot_task 产出 QC 通过的关键帧，这里以它为 source_image 生成动态镜头。
支持 Runway（默认）、Pika（云端，都需 API key）和 Mochi（本地免费）三种 provider；
本任务只做 provider 选择和参数透传，素材登记由 07 的 VideoGenerator 写 jsonl 账本。
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

from ..workers.celery_app import celery_app

__all__ = ["video_task"]


@celery_app.task(name="video_task", bind=True)
def video_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """视频生成 GPU 队列任务（对一个 Shot 做图生视频 / 文生视频）。

    payload: {"prompt": str, "source_image": str | None, "shot_id": str | None,
      "character_id": str | None, "episode_id": str | None, "duration_sec": float,
      "seed": int | None, "provider": str | None ("runway" 默认 | "pika" | "mochi" 本地免费),
      "output_dir": str | None, "asset_log_path": str | None, "license": str | None}。
    返回: {"file_path": str, "model": str, "seed": int | None}。

    直接调用 07_GENERATION.video.video_generator 里的真实类：VideoGenerator 包装一个
    BaseVideoProvider（RunwayVideoProvider 或 PikaVideoProvider），它们在各自的 API key 环境变量
    未设置时构造即抛 NotConfiguredError——这个错误会原样抛出去，让配置错误的 worker 大声失败而不是静默跳过。
    """
    # 数字开头的顶层包只能用 importlib 按名字 import
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

    # model 返回 provider 类名；seed 原样带回，方便复现同一段视频
    return {"file_path": file_path, "model": type(provider).__name__, "seed": seed}
