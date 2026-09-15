"""API 请求/响应的 Pydantic schema。

命名约定：``XxxCreate`` 是 POST 请求体，``XxxRead`` 是响应体（``from_attributes=True`` 让 SQLAlchemy ORM 对象能直接序列化）。
ORM 模型（database/models.py）和对外 schema 分开定义，是为了不把数据库内部字段（如关系、内部状态）
原样暴露出去，也方便 API 契约独立于表结构演进。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    """创建项目的请求体。"""

    name: str
    description: Optional[str] = None


class ProjectRead(BaseModel):
    """项目响应体。"""

    model_config = ConfigDict(from_attributes=True)

    project_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CharacterCreate(BaseModel):
    """创建角色的请求体；reference_image_path 是角色参考图，供生图阶段保持人物一致性。"""

    project_id: str
    name: str
    description: Optional[str] = None
    reference_image_path: Optional[str] = None


class CharacterRead(BaseModel):
    """角色响应体。"""

    model_config = ConfigDict(from_attributes=True)

    character_id: str
    project_id: str
    name: str
    description: Optional[str] = None
    reference_image_path: Optional[str] = None
    created_at: datetime


class EpisodeCreate(BaseModel):
    """创建剧集的请求体。"""

    project_id: str
    episode_number: int
    title: Optional[str] = None
    status: str = "draft"


class EpisodeRead(BaseModel):
    """剧集响应体；review_status 是人工审核状态，video_path / manifest_path 在渲染/打包后才有值。"""

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
    """创建场景的请求体（当前没有挂对外路由，保留给内部/未来使用）。"""

    episode_id: str
    scene_number: int
    description: Optional[str] = None
    location: Optional[str] = None


class SceneRead(BaseModel):
    """场景响应体。"""

    model_config = ConfigDict(from_attributes=True)

    scene_id: str
    episode_id: str
    scene_number: int
    description: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime


class ShotCreate(BaseModel):
    """创建镜头的请求体。"""

    scene_id: str
    shot_number: int
    description: Optional[str] = None
    duration_sec: Optional[float] = None
    video_path: Optional[str] = None


class ShotRead(BaseModel):
    """镜头响应体。"""

    model_config = ConfigDict(from_attributes=True)

    shot_id: str
    scene_id: str
    shot_number: int
    description: Optional[str] = None
    duration_sec: Optional[float] = None
    video_path: Optional[str] = None
    created_at: datetime


class AssetCreate(BaseModel):
    """登记生成素材的请求体：文件路径 + 溯源信息（模型、prompt、seed、版本、许可）。"""

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
    """素材响应体。"""

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
    """POST /tasks 请求体：队列名 + 裸 payload（payload 的细校验在 task_payloads.py 里按队列做）。"""

    queue: str
    payload: dict


class TaskEnqueueResponse(BaseModel):
    """入队成功的响应：Celery task_id 和当前状态（通常 PENDING；eager 模式下可能直接是 SUCCESS）。"""

    task_id: str
    queue: str
    status: str


class TaskStatusResponse(BaseModel):
    """GET /tasks/{task_id} 响应：status 是 Celery 状态名，成功时带 result，失败时带 error 文本。"""

    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ---- 流水线 ---------------------------------------------------------------------------


class PipelineCreateRequest(BaseModel):
    """POST /pipelines/episodes 请求体：一句创意 + 生成参数。

    ``extra="forbid"``：参数名拼错直接 422，而不是被静默忽略后用默认值跑一遍昂贵的 LLM 流程。
    各数值都有上下限，防止误传把成本或时长打爆。
    """

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
    """人工批准请求：episode_ids 为空表示批准该 run 下全部集。"""

    episode_ids: Optional[list[str]] = None
    notes: Optional[str] = None


class PipelineRejectRequest(BaseModel):
    """人工打回请求：notes 必填，打回总得说明原因。"""

    notes: str = Field(min_length=1)


class PipelineRunRead(BaseModel):
    """流水线运行的响应体：状态/阶段/参数/结果，外加各集的摘要列表（由 routers/pipelines._read 组装）。"""

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
    """一次运行的 LLM 用量汇总：总调用次数、输入/输出 token、美元成本，以及按 agent 拆分的明细。"""

    run_id: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    by_agent: dict[str, dict]


class PipelineAssistantRequest(BaseModel):
    """向制片助理 Agent 提问：question 是自然语言；context_id 不传则同一 run 的多次提问共享会话记忆。"""

    question: str = Field(min_length=2, description="例如：帮我检查第 1 集有没有引用错误")
    context_id: Optional[str] = None


class PipelineAssistantResponse(BaseModel):
    """制片助理的回复。eager 模式（本地/测试）result 立即可用；生产模式先拿 task_id 去 GET /tasks/{task_id} 轮询。"""

    run_id: str
    task_id: str
    status: str
    result: Optional[dict] = None  # assistant_task 的返回：answer / steps / tool_rounds / stopped_reason / flagged / usage
    error: Optional[str] = None


class EpisodeReviewRequest(BaseModel):
    """单集人工审校请求：decision 只能是 approved / rejected。"""

    decision: Literal["approved", "rejected"]
    notes: Optional[str] = None
