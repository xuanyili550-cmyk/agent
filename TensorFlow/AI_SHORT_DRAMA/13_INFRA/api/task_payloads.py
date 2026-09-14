"""/tasks 入队 payload 的按队列校验。之前 payload 是裸 dict，拼错字段名要到 worker 里才炸；
现在在 API 入口就用 Pydantic 拦下来（extra="forbid"：多一个不认识的字段也报 422）。"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMTaskPayload(_Strict):
    prompt: str = Field(min_length=1)
    system: Optional[str] = None
    model: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    context_id: Optional[str] = None


class ImageTaskPayload(_Strict):
    prompt: str = Field(min_length=1)
    negative_prompt: Optional[str] = None
    character_id: Optional[str] = None
    reference_character_ids: Optional[list[str]] = None
    reference_images: Optional[list[str]] = None
    lora_path: Optional[str] = None
    episode_id: Optional[str] = None
    shot_id: Optional[str] = None
    seed: Optional[int] = None
    model: Optional[str] = None
    width: int = Field(1024, ge=64, le=4096)
    height: int = Field(1024, ge=64, le=4096)
    output_dir: Optional[str] = None
    asset_log_path: Optional[str] = None
    license: Optional[str] = None


class ShotTaskPayload(_Strict):
    shot_id: str
    prompt: str = Field(min_length=1)
    negative_prompt: Optional[str] = None
    reference_character_ids: Optional[list[str]] = None
    character_id: Optional[str] = None
    episode_id: Optional[str] = None
    run_id: Optional[str] = None
    seed: Optional[int] = None
    output_dir: Optional[str] = None
    image_prompt: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    license: Optional[str] = None


class VideoTaskPayload(_Strict):
    prompt: str = Field(min_length=1)
    source_image: Optional[str] = None
    shot_id: Optional[str] = None
    character_id: Optional[str] = None
    episode_id: Optional[str] = None
    duration_sec: float = Field(4.0, gt=0, le=60)
    seed: Optional[int] = None
    provider: Literal["runway", "pika", "mochi"] = "runway"
    output_dir: Optional[str] = None
    asset_log_path: Optional[str] = None
    license: Optional[str] = None


class TTSTaskPayload(_Strict):
    text: str = Field(min_length=1)
    voice_id: str
    provider: Literal["elevenlabs", "bark", "xtts"] = "elevenlabs"
    model: Optional[str] = None
    device: Optional[str] = None
    character_id: Optional[str] = None
    episode_id: Optional[str] = None
    shot_id: Optional[str] = None
    seed: Optional[int] = None
    output_dir: Optional[str] = None
    asset_log_path: Optional[str] = None
    license: Optional[str] = None


class LipsyncTaskPayload(_Strict):
    video_path: str
    audio_path: str
    shot_id: Optional[str] = None
    character_id: Optional[str] = None
    episode_id: Optional[str] = None
    seed: Optional[int] = None
    output_dir: Optional[str] = None
    asset_log_path: Optional[str] = None
    license: Optional[str] = None


class QCTaskPayload(_Strict):
    asset_id: Optional[str] = None
    shot_id: Optional[str] = None
    episode_id: Optional[str] = None
    reference_character_image: Optional[str] = None
    generated_image: Optional[str] = None
    character_id: Optional[str] = None
    scene_description: Optional[str] = None
    scene_media_image: Optional[str] = None
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    attempt: int = 1


class AnalyticsTaskPayload(_Strict):
    sources: list[dict[str, Any]] = Field(default_factory=list)
    experiment: Optional[str] = None
    since_days: Optional[int] = Field(None, ge=1, le=365)


PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "llm": LLMTaskPayload,
    "image": ImageTaskPayload,
    "shot": ShotTaskPayload,
    "video": VideoTaskPayload,
    "tts": TTSTaskPayload,
    "lipsync": LipsyncTaskPayload,
    "qc": QCTaskPayload,
    "analytics": AnalyticsTaskPayload,
}


def validate_payload(queue: str, payload: dict[str, Any]) -> dict[str, Any]:
    model = PAYLOAD_MODELS[queue]
    return model.model_validate(payload).model_dump(exclude_none=True)
