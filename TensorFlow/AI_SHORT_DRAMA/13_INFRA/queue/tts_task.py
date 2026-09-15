"""语音合成任务：一句对白 -> 一段配音音频（07_GENERATION.audio 的队列封装）。

在生产链里和 video_task 并行：镜头画面和对白语音各自生成，之后由 lipsync_task 对口型、09_POST 混音。
支持三种 provider：ElevenLabs（云端，需 API key）、Bark（本地免费）、XTTS（本地，需单独 venv）；
本任务只做 provider 选择和参数透传，素材登记由 07 的 TTSGenerator 写 jsonl 账本。
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

from ..workers.celery_app import celery_app

__all__ = ["tts_task"]


@celery_app.task(name="tts_task", bind=True)
def tts_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """文本转语音 GPU 队列任务（对白台词 -> 语音音频）。

    payload: {"text": str, "voice_id": str, "provider": "elevenlabs"|"bark"|"xtts", "character_id": str | None,
      "episode_id": str | None, "shot_id": str | None, "seed": int | None,
      "output_dir": str | None, "asset_log_path": str | None, "license": str | None}。
    返回: {"file_path": str, "model": str}。

    直接调用 07_GENERATION.audio.tts_generator 里的真实类：TTSGenerator 包装一个
    BaseTTSProvider（ElevenLabsTTSProvider），后者在 ELEVENLABS_API_KEY 未设置时构造即抛
    NotConfiguredError。voice_id 必须是 provider 侧的音色 id（音色注册见
    07_GENERATION.voice.voice_cloner.VoiceCloner，那是一次性的前置步骤，和逐句合成分开）。
    """
    # 数字开头的顶层包只能用 importlib 按名字 import
    tts_generator_mod = importlib.import_module("07_GENERATION.audio.tts_generator")

    # payload["provider"]: "elevenlabs"（默认，需 ELEVENLABS_API_KEY）| "bark"（本地免费）| "xtts"（本地，需 .venv_xtts）
    provider_name = (payload.get("provider") or "elevenlabs").lower()
    if provider_name == "bark":
        provider = tts_generator_mod.BarkLocalTTSProvider(model_id=payload.get("model") or "suno/bark-small", device=payload.get("device") or "cpu")
    elif provider_name == "xtts":
        provider = tts_generator_mod.CoquiXTTSLocalTTSProvider()
    else:
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

    # model 返回 provider 类名：结果里能看出这条素材是哪家服务出的
    return {"file_path": file_path, "model": type(provider).__name__}
