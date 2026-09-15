"""
清单行结构定义：04_DATASET/metadata/*.jsonl 各类清单文件中"每一行"的 Pydantic 模型。

在流水线中的位置：数据层的契约。``generate_sample_manifests.py`` 用这些模型序列化写出 JSONL，
``dataset_loaders.py`` 读出来的列名也必须与这里的字段一致。

为什么用 Pydantic 而不是裸 dict：写入时字段名 / 类型自动校验，缺 ``source`` / ``license`` 之类的合规字段会直接报错，
避免把来源不明的素材混进训练集。每类清单都强制携带 ``source`` 和 ``license``，就是为了版权可追溯。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# 数据切分只允许 train / val 两种取值，防止手写清单时拼错
Split = Literal["train", "val"]


class CharacterManifestEntry(BaseModel):
    """角色参考图清单的一行：一个角色 id 对应一张参考图及其文字描述，用于角色 LoRA 训练。"""

    character_id: str
    image_path: str
    caption: str
    pose: Optional[str] = None
    expression: Optional[str] = None
    source: str
    license: str
    split: Split = "train"


class SceneManifestEntry(BaseModel):
    """场景 / 环境参考图清单的一行，附带时段与天气标签，便于按条件筛选。"""

    environment_id: str
    image_path: str
    caption: str
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    source: str
    license: str
    split: Split = "train"


class StyleManifestEntry(BaseModel):
    """视觉风格参考图清单的一行；``style_tags`` 是风格关键词列表，用于风格 LoRA 训练与检索。"""

    style_id: str
    image_path: str
    caption: str
    style_tags: list[str] = Field(default_factory=list)
    source: str
    license: str
    split: Split = "train"


class VoiceManifestEntry(BaseModel):
    """语音清单的一行：一段音频及其转写文本，可关联到某个角色，用于 TTS 音色训练 / 评测。"""

    voice_id: str
    character_id: Optional[str] = None
    audio_path: str
    transcript: str
    speaker: Optional[str] = None
    duration_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    source: str
    license: str
    split: Split = "train"


class VideoManifestEntry(BaseModel):
    """视频片段清单的一行：一个片段及其描述、时长、帧率，可关联到出镜角色。"""

    clip_id: str
    video_path: str
    caption: str
    character_id: Optional[str] = None
    duration_sec: Optional[float] = None
    fps: Optional[int] = None
    source: str
    license: str
    split: Split = "train"
