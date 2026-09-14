from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    review_status: str = "pending_review"
    review_notes: Optional[str] = None
    video_path: Optional[str] = None
    manifest_path: Optional[str] = None
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


# ---- 流水线 ---------------------------------------------------------------------------


class PipelineCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    idea: str = Field(min_length=4, description="一句话故事创意")
    num_characters: int = Field(3, ge=1, le=6)
    num_scenes: int = Field(2, ge=1, le=8)
    episode_numbers: list[int] = Field(default_factory=lambda: [1])
    num_episodes_planned: int = Field(12, ge=1, le=100)
    target_duration_seconds: int = Field(300, ge=60, le=600)
    prompt_mode: Literal["llm", "template"] = "llm"
    planner: Literal["llm", "rule"] = "llm"
    editorial: bool = True
    max_revision_rounds: int = Field(2, ge=0, le=5)
    image_backend: Optional[str] = None


class PipelineApproveRequest(BaseModel):
    episode_ids: Optional[list[str]] = None
    notes: Optional[str] = None


class PipelineRejectRequest(BaseModel):
    notes: str = Field(min_length=1)


class PipelineRunRead(BaseModel):
    run_id: str
    project_id: str
    status: str
    stage: Optional[str] = None
    params: dict
    result: dict
    error: Optional[str] = None
    episodes: list[dict] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class PipelineUsageRead(BaseModel):
    run_id: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    by_agent: dict[str, dict]


class EpisodeReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    notes: Optional[str] = None
