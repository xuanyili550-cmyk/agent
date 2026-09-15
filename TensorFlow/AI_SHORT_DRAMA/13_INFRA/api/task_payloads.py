"""/tasks 入队 payload 的按队列校验。之前 payload 是裸 dict，拼错字段名要到 worker 里才炸；
现在在 API 入口就用 Pydantic 拦下来（extra="forbid"：多一个不认识的字段也报 422）。

每个队列（llm / image / shot / video / tts / lipsync / qc / analytics）对应一个 payload 模型，
字段和 ``13_INFRA/queue/*_task.py`` 里任务函数实际读取的键一一对应；``PAYLOAD_MODELS`` 是队列名到模型的映射，
``GET /tasks/schemas`` 也靠它把 JSON Schema 暴露给前端。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """所有 payload 模型的基类：禁止未知字段，让拼错的键名在 API 层就报 422 而不是静默丢弃。"""

    model_config = ConfigDict(extra="forbid")


class LLMTaskPayload(_Strict):
    """llm 队列：单次 LLM 调用（prompt + 可选 system/model/参数；context_id 用于接续长期会话记忆）。"""

    prompt: str = Field(min_length=1)
    system: Optional[str] = None
    model: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    context_id: Optional[str] = None


class ImageTaskPayload(_Strict):
    """image 队列：文生图/参考图生图。尺寸限制在 64~4096，避免误传把 GPU 打爆。"""

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
    """shot 队列：流水线里的"镜头生成 + QC + 重试"复合任务；shot_id 必填，image_prompt 是剧本阶段落库的结构化提示词。"""

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
    """video 队列：图生视频/文生视频；provider 限定为已接入的三家，时长上限 60 秒。"""

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
    """tts 队列：文本转语音；voice_id 必填（角色音色），provider 默认 elevenlabs。"""

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
    """lipsync 队列：把音频对口型合到视频上；video_path / audio_path 必填。"""

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
    """qc 队列：质检（角色一致性 / 场景匹配 / 音画）。字段全部可选，按传入的素材决定跑哪几项；attempt 是重试轮次。"""

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
    """analytics 队列：拉取投放数据做分析；sources 是各平台数据源配置，since_days 限制回看窗口。"""

    sources: list[dict[str, Any]] = Field(default_factory=list)
    experiment: Optional[str] = None
    since_days: Optional[int] = Field(None, ge=1, le=365)


class AssistantTaskPayload(_Strict):
    """assistant 队列：制片助理（ToolAgent）对一次运行的产出提问；context_id 不传则按 run 续接会话。"""

    run_id: str
    question: str = Field(min_length=2)
    context_id: Optional[str] = None


# 队列名 -> payload 模型；routers/tasks.py 的 _TASKS 必须和这里的键保持一致
PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "llm": LLMTaskPayload,
    "assistant": AssistantTaskPayload,
    "image": ImageTaskPayload,
    "shot": ShotTaskPayload,
    "video": VideoTaskPayload,
    "tts": TTSTaskPayload,
    "lipsync": LipsyncTaskPayload,
    "qc": QCTaskPayload,
    "analytics": AnalyticsTaskPayload,
}


def validate_payload(queue: str, payload: dict[str, Any]) -> dict[str, Any]:
    """按队列校验 payload 并返回干净的 dict。

    ``exclude_none=True``：没传的可选字段不塞 None 进去，worker 端用 ``payload.get()`` 取默认值的逻辑才不会被 None 覆盖。
    校验失败抛 pydantic ``ValidationError``，由路由层转成 422。
    """
    model = PAYLOAD_MODELS[queue]
    return model.model_validate(payload).model_dump(exclude_none=True)
