from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    project_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CharacterCreate(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    reference_image_path: Optional[str] = None


class CharacterRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    character_id: str
    project_id: str
    name: str
    description: Optional[str] = None
    reference_image_path: Optional[str] = None
    created_at: datetime


class EpisodeCreate(BaseModel):
    project_id: str
    episode_number: int
    title: Optional[str] = None
    status: str = "draft"


class EpisodeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    episode_id: str
    project_id: str
    episode_number: int
    title: Optional[str] = None
    status: str
    created_at: datetime


class SceneCreate(BaseModel):
    episode_id: str
    scene_number: int
    description: Optional[str] = None
    location: Optional[str] = None


class SceneRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    scene_id: str
    episode_id: str
    scene_number: int
    description: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime


class ShotCreate(BaseModel):
    scene_id: str
    shot_number: int
    description: Optional[str] = None
    duration_sec: Optional[float] = None
    video_path: Optional[str] = None


class ShotRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    shot_id: str
    scene_id: str
    shot_number: int
    description: Optional[str] = None
    duration_sec: Optional[float] = None
    video_path: Optional[str] = None
    created_at: datetime


class AssetCreate(BaseModel):
    file_path: str
    character_id: Optional[str] = None
    episode_id: Optional[str] = None
    shot_id: Optional[str] = None
    model: Optional[str] = None
    prompt: Optional[str] = None
    seed: Optional[int] = None
    version: Optional[str] = None
    license: Optional[str] = None


class AssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    asset_id: str
    file_path: str
    character_id: Optional[str] = None
    episode_id: Optional[str] = None
    shot_id: Optional[str] = None
    model: Optional[str] = None
    prompt: Optional[str] = None
    seed: Optional[int] = None
    version: Optional[str] = None
    license: Optional[str] = None
    created_at: datetime


class TaskEnqueueRequest(BaseModel):
    queue: str
    payload: dict


class TaskEnqueueResponse(BaseModel):
    task_id: str
    queue: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
