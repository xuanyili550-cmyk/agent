"""SQLAlchemy 表定义。

设计原则（对应 ARCHITECTURE.md 2.2 "03_STRUCTURED_DATA 是唯一真源"）：
业务对象（StoryBible / Character / Episode / Scene / Shot / Prompt）的**字段定义只在
03_STRUCTURED_DATA/schemas.py**，数据库这边不重复定义一份字段不同的 Shot——每张业务表只保留
"主键 + 外键 + 需要建索引/筛选的少数列"，完整对象原样存在 ``data`` JSON 列里，读出来用
``schemas.Shot.model_validate(row.data)`` 还原成强类型对象。这样 schema 加字段时不用改表。

主键 = ``<project_id>__<业务 id>``（见 repository.scoped_id）：03 里的业务 id
（``char_su_wanwan`` / ``shot_001_01_02``）在每个项目里都从头编号，直接当主键会跨项目互相覆盖；
加了项目前缀之后 AssetRecord.shot_id 和 Shot 才能在数据库里真正关联上，``data`` 里仍保留原始
业务 id。通过 API 手工创建的记录没有业务 id 时退回 uuid 默认值。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, List, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

ID_LEN = 128


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Project(Base):
    __tablename__ = "projects"

    project_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    characters: Mapped[List[Character]] = relationship(back_populates="project")
    episodes: Mapped[List[Episode]] = relationship(back_populates="project")
    story_bibles: Mapped[List[StoryBible]] = relationship(back_populates="project")


class StoryBible(Base):
    __tablename__ = "story_bibles"

    story_bible_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # schemas.StoryBible
    season_arcs: Mapped[list[Any]] = mapped_column(JSON, default=list)  # list[schemas.SeasonArc]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped[Project] = relationship(back_populates="story_bibles")


class Character(Base):
    __tablename__ = "characters"

    character_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    role: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    reference_image_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    lora_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)  # 角色 LoRA 权重，保证跨镜头长相一致
    data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)  # schemas.Character
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped[Project] = relationship(back_populates="characters")
    assets: Mapped[List[Asset]] = relationship(back_populates="character")


class Episode(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), nullable=False, index=True)
    episode_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="draft")  # draft/scripted/producing/rendered/published
    # 人工/总编审审校状态：pending_review -> approved / rejected；approved 才允许进入生产
    review_status: Mapped[str] = mapped_column(String(50), default="pending_review")
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    season_id: Mapped[Optional[str]] = mapped_column(String(ID_LEN), nullable=True)
    data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)  # schemas.Episode
    script: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)  # schemas.Script
    editorial: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)  # 质检官 + 总编审的判定
    video_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    manifest_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    project: Mapped[Project] = relationship(back_populates="episodes")
    scenes: Mapped[List[Scene]] = relationship(back_populates="episode")
    assets: Mapped[List[Asset]] = relationship(back_populates="episode")

    __table_args__ = (Index("ix_episodes_project_number", "project_id", "episode_number"),)


class Scene(Base):
    __tablename__ = "scenes"

    scene_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.episode_id"), nullable=False, index=True)
    scene_number: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)  # schemas.Scene
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    episode: Mapped[Episode] = relationship(back_populates="scenes")
    shots: Mapped[List[Shot]] = relationship(back_populates="scene")


class Shot(Base):
    __tablename__ = "shots"

    shot_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.scene_id"), nullable=False, index=True)
    episode_id: Mapped[Optional[str]] = mapped_column(ForeignKey("episodes.episode_id"), nullable=True, index=True)
    shot_number: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)  # QC 通过的关键帧
    status: Mapped[str] = mapped_column(String(50), default="planned")  # planned/generating/approved/failed
    data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)  # schemas.Shot
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    scene: Mapped[Scene] = relationship(back_populates="shots")
    assets: Mapped[List[Asset]] = relationship(back_populates="shot")
    prompts: Mapped[List[Prompt]] = relationship(back_populates="shot")


class Prompt(Base):
    """ImagePrompt / VideoPrompt 各一行，kind 区分；完整对象在 data 里。"""

    __tablename__ = "prompts"

    prompt_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    shot_id: Mapped[str] = mapped_column(ForeignKey("shots.shot_id"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)  # image / video
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    shot: Mapped[Shot] = relationship(back_populates="prompts")


class Asset(Base):
    __tablename__ = "assets"

    asset_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), default="image")  # image/video/audio
    character_id: Mapped[Optional[str]] = mapped_column(ForeignKey("characters.character_id"), nullable=True)
    episode_id: Mapped[Optional[str]] = mapped_column(ForeignKey("episodes.episode_id"), nullable=True)
    shot_id: Mapped[Optional[str]] = mapped_column(ForeignKey("shots.shot_id"), nullable=True, index=True)
    model: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    license: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    qc_status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/approved/rejected
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    retry_level: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)  # same_params/rewrite_prompt/downscale
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    storage_key: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    character: Mapped[Optional[Character]] = relationship(back_populates="assets")
    episode: Mapped[Optional[Episode]] = relationship(back_populates="assets")
    shot: Mapped[Optional[Shot]] = relationship(back_populates="assets")
    qc_reports: Mapped[List[QCReportRow]] = relationship(back_populates="asset")


class QCReportRow(Base):
    __tablename__ = "qc_reports"

    qc_report_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    asset_id: Mapped[Optional[str]] = mapped_column(ForeignKey("assets.asset_id"), nullable=True, index=True)
    shot_id: Mapped[Optional[str]] = mapped_column(String(ID_LEN), nullable=True, index=True)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)  # approved / retry
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # 08_QC QCReport
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    asset: Mapped[Optional[Asset]] = relationship(back_populates="qc_reports")


class LLMUsage(Base):
    """每次 LLM 调用一行：谁调的、用了多少 token、估算花了多少钱。成本核算和限额都靠它。"""

    __tablename__ = "llm_usage"

    usage_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    context_id: Mapped[Optional[str]] = mapped_column(String(ID_LEN), nullable=True, index=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(ID_LEN), nullable=True, index=True)
    agent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PipelineRun(Base):
    """一次"从创意到成片"的流水线运行。API 用它查进度、人工审核用它放行。"""

    __tablename__ = "pipeline_runs"

    run_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), nullable=False, index=True)
    episode_id: Mapped[Optional[str]] = mapped_column(String(ID_LEN), nullable=True)
    # queued -> scripting -> awaiting_review -> producing -> rendering -> publishing -> done / failed / rejected
    status: Mapped[str] = mapped_column(String(50), default="queued", index=True)
    stage: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class PublishRecord(Base):
    __tablename__ = "publish_records"

    publish_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    episode_id: Mapped[str] = mapped_column(String(ID_LEN), nullable=False, index=True)
    platform: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)  # 10_EPISODES.PublishStatus
    remote_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    remote_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    dry_run: Mapped[bool] = mapped_column(default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    attempted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AnalyticsEvent(Base):
    """用户级事件（自家 App/SDK 上报或文件导入）。平台聚合指标另存 AnalyticsMetric。"""

    __tablename__ = "analytics_events"

    event_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    source: Mapped[str] = mapped_column(String(50), nullable=False)  # app / file / youtube / tiktok
    user_id: Mapped[Optional[str]] = mapped_column(String(ID_LEN), nullable=True, index=True)
    episode_id: Mapped[Optional[str]] = mapped_column(String(ID_LEN), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    surface: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    amount_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    experiment: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    variant: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    raw: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)


class AnalyticsMetric(Base):
    """平台侧按天聚合指标（YouTube Analytics / TikTok 只给聚合数，不给用户级事件）。"""

    __tablename__ = "analytics_metrics"

    metric_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    episode_id: Mapped[Optional[str]] = mapped_column(String(ID_LEN), nullable=True, index=True)
    remote_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    day: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    views: Mapped[int] = mapped_column(Integer, default=0)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    comments: Mapped[int] = mapped_column(Integer, default=0)
    shares: Mapped[int] = mapped_column(Integer, default=0)
    watch_time_minutes: Mapped[float] = mapped_column(Float, default=0.0)
    estimated_revenue_usd: Mapped[float] = mapped_column(Float, default=0.0)
    raw: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    __table_args__ = (Index("ix_metrics_source_remote_day", "source", "remote_id", "day", unique=True),)


class AnalyticsSnapshot(Base):
    __tablename__ = "analytics_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    metric: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # retention/ctr/revenue/experiment
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
