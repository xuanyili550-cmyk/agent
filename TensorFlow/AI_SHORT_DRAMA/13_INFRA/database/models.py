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

表分三组：
- 剧本产物：projects / story_bibles / characters / episodes / scenes / shots / prompts；
- 生产过程：assets（每次生成一行）/ qc_reports（每次 QC 一行）/ llm_usage（每次 LLM 调用一行）/ pipeline_runs；
- 发布与分析：publish_records / analytics_events / analytics_metrics / analytics_snapshots。
所有时间列都是带时区的 UTC；主键统一 String(ID_LEN)，因为业务 id 是可读字符串而不是整数自增。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, List, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

# 主键/外键列宽：36 位 uuid 放不下 "<project_id>__<业务 id>" 这种带前缀的 id，放宽到 128
ID_LEN = 128


def _uuid() -> str:
    """主键默认值：没有业务 id 的记录（API 手工创建、素材、用量、运行记录）用随机 uuid 字符串。"""
    return str(uuid.uuid4())


def _now() -> datetime:
    """时间列默认值：统一用带时区的 UTC，避免 SQLite/Postgres 之间 naive/aware 混用出错。"""
    return datetime.now(UTC)


class Project(Base):
    """项目（一部剧）：所有业务对象的顶层归属；project_id 也是其它表主键的前缀。"""

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
    """故事圣经（schemas.StoryBible）：一个项目的世界观/人设/主线；季度弧线列表一起存在 season_arcs 里。"""

    __tablename__ = "story_bibles"

    story_bible_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # schemas.StoryBible
    season_arcs: Mapped[list[Any]] = mapped_column(JSON, default=list)  # list[schemas.SeasonArc]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped[Project] = relationship(back_populates="story_bibles")


class Character(Base):
    """角色（schemas.Character）。除 JSON 全量外单独拉出 reference_image_path / lora_path 两列，
    因为镜头生成时要按角色 id 直接查参考图和 LoRA（见 queue.shot_task.resolve_character_assets）。"""

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
    """一集（schemas.Episode）。同时挂着剧本 ``script`` 和编审结论 ``editorial`` 两个 JSON，
    以及生产状态 ``status`` 和人工审核状态 ``review_status``——两条状态线分开：
    status 记"做到哪一步"，review_status 记"有没有被放行"，只有 approved 的集才能进生产。"""

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

    # API 按"项目 + 集号"列出剧集是最常见的查询，建联合索引
    __table_args__ = (Index("ix_episodes_project_number", "project_id", "episode_number"),)


class Scene(Base):
    """场次（schemas.Scene）：一集下的一个场景；对白行也在 data 里，渲染时按 shot_id 取出来做字幕。"""

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
    """镜头（schemas.Shot）：生产的最小单元。episode_id 在 scene 之外冗余一份，
    是为了 render_task 能直接按集取全部镜头而不用 join scenes；
    status / image_path / video_path 由镜头生产任务回写，是生产结果而非剧本内容。"""

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
    """生成素材登记（对应 07_GENERATION.asset_registry.AssetRecord）：每次生成一行，重试不覆盖，
    attempt / retry_level / qc_status 记录这一次是阶梯第几级、结果如何，事后可以追溯每个镜头的完整尝试史。
    model / prompt / seed 一起存是为了可复现（同样参数能再生成一张一样的图）。"""

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
    """08_QC 的 QCReport 落库（每次 QC 一行）。类名加 Row 后缀是为了和 08_QC 里的 pydantic ``QCReport`` 区分。
    shot_id 不做外键：素材可能先于 Shot 存在（手工调 qc_task），也可能在重跑时被解除关联。"""

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
    """每次 LLM 调用一行：谁调的、用了多少 token、估算花了多少钱。成本核算和限额都靠它。

    context_id（会话）和 run_id（流水线运行）都做索引，API 的 /pipelines/{run_id}/usage 按 run_id 聚合。"""

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
    """一次"从创意到成片"的流水线运行。API 用它查进度、人工审核用它放行。

    params 是启动时的全部输入（story_task 只从这里取参数，不靠 Celery 参数传递，重跑时可原样复用）；
    result 是各阶段累积写入的产出（episode_ids / failed_shots / renders / manifests / publish...），
    用 repository.update_run 合并更新而不是覆盖，所以后一阶段不会抹掉前一阶段记录的东西。"""

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
    """一次向某个平台的发布尝试（每个平台一行，重试再加一行）。
    episode_id 存的是 manifest 里的业务 id（ep_001）而不是数据库主键，因为发布环节只跟 10_EPISODES 的 manifest 打交道；
    status 之后由 publish_status_task 轮询回写。"""

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
    """用户级事件（自家 App/SDK 上报或文件导入）。平台聚合指标另存 AnalyticsMetric。

    event_id 由数据源给出去重键（见 12_ANALYTICS.ingest），重复导入同一批事件会撞主键而不是产生重复行。"""

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
    """平台侧按天聚合指标（YouTube Analytics / TikTok 只给聚合数，不给用户级事件）。

    (source, remote_id, day) 唯一：同一视频同一天重复拉取是更新不是新增（见 queue.analytics_task.ingest_events）。"""

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
    """一次分析计算的结果快照（留存 / CTR / 收入 / 实验各一行）。
    存快照而不是每次现算，是为了 API 和看板读起来快，也能回看历史某次计算的结果。"""

    __tablename__ = "analytics_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(ID_LEN), primary_key=True, default=_uuid)
    metric: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # retention/ctr/revenue/experiment
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
