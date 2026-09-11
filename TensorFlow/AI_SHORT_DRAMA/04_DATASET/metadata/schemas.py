"""Pydantic row schemas for the 04_DATASET/metadata/*.jsonl manifests."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Split = Literal["train", "val"]


class CharacterManifestEntry(BaseModel):
    character_id: str
    image_path: str
    caption: str
    pose: Optional[str] = None
    expression: Optional[str] = None
    source: str
    license: str
    split: Split = "train"


class SceneManifestEntry(BaseModel):
    environment_id: str
    image_path: str
    caption: str
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    source: str
    license: str
    split: Split = "train"


class StyleManifestEntry(BaseModel):
    style_id: str
    image_path: str
    caption: str
    style_tags: list[str] = Field(default_factory=list)
    source: str
    license: str
    split: Split = "train"


class VoiceManifestEntry(BaseModel):
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
    clip_id: str
    video_path: str
    caption: str
    character_id: Optional[str] = None
    duration_sec: Optional[float] = None
    fps: Optional[int] = None
    source: str
    license: str
    split: Split = "train"
