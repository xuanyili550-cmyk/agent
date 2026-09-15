"""把 03_STRUCTURED_DATA 对象接进数据库：JSON data 列、业务 id 主键、prompts/qc/usage/pipeline/publish/analytics 表

Revision ID: a3c9e1d2f4b5
Revises: f7d0958c1f12
Create Date: 2026-09-14

v0.2 生产化改造对应的表结构变更，和 models.py 当前定义对齐。做了三类事：
1. 主键/外键列从 String(36) 放宽到 String(128)：主键改成 ``<project_id>__<业务 id>``（见 repository.scoped_id），
   36 位放不下；
2. 既有六张表补列：各业务表加 JSON ``data`` 列整体存 03 的对象；episodes 加审核状态/剧本/编审结论；
   shots 加 episode_id 冗余外键和生产状态；assets 加 QC 状态与重试阶梯字段；
3. 新建九张表：story_bibles、prompts、qc_reports、llm_usage、pipeline_runs、publish_records、
   analytics_events / analytics_metrics / analytics_snapshots。
全部用 ``batch_alter_table``：SQLite 不支持 ALTER COLUMN / ADD CONSTRAINT，batch 模式会用"建新表-拷数据-换名"
的方式模拟，Postgres 上则退化成普通 ALTER，同一份脚本两种库都能跑。
新增的 NOT NULL 列都带 server_default，已有行升级时才不会因为缺值而失败。
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# alembic 用的版本标识：本迁移基于 f7d0958c1f12（初始建表）
revision: str = "a3c9e1d2f4b5"
down_revision: Union[str, Sequence[str], None] = "f7d0958c1f12"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# 与 models.ID_LEN 保持一致；迁移脚本不能 import models（迁移必须冻结在写它时的定义），所以这里复制一份
ID_LEN = 128


def upgrade() -> None:
    """升级：放宽 id 列宽 -> 给既有表补列/索引/外键 -> 建九张新表。
    顺序不能反：新表的外键引用的是放宽后的 String(128) 列。"""
    # 主键从 36 位 uuid 放宽到 128 位业务 id（char_su_wanwan / shot_001_01_02）
    for table, column in [
        ("projects", "project_id"),
        ("characters", "character_id"),
        ("characters", "project_id"),
        ("episodes", "episode_id"),
        ("episodes", "project_id"),
        ("scenes", "scene_id"),
        ("scenes", "episode_id"),
        ("shots", "shot_id"),
        ("shots", "scene_id"),
        ("assets", "asset_id"),
        ("assets", "character_id"),
        ("assets", "episode_id"),
        ("assets", "shot_id"),
    ]:
        with op.batch_alter_table(table) as batch:
            batch.alter_column(column, type_=sa.String(ID_LEN), existing_type=sa.String(36))

    with op.batch_alter_table("characters") as batch:
        batch.add_column(sa.Column("role", sa.String(50), nullable=True))
        batch.add_column(sa.Column("lora_path", sa.String(1024), nullable=True))
        batch.add_column(sa.Column("data", sa.JSON(), nullable=True))

    with op.batch_alter_table("episodes") as batch:
        # 已有行默认 pending_review：旧数据一律当作未审核，宁可多审一次也不能漏
        batch.add_column(sa.Column("review_status", sa.String(50), nullable=False, server_default="pending_review"))
        batch.add_column(sa.Column("review_notes", sa.Text(), nullable=True))
        batch.add_column(sa.Column("season_id", sa.String(ID_LEN), nullable=True))
        batch.add_column(sa.Column("data", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("script", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("editorial", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("video_path", sa.String(1024), nullable=True))
        batch.add_column(sa.Column("manifest_path", sa.String(1024), nullable=True))
        batch.add_column(sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_index("ix_episodes_project_number", ["project_id", "episode_number"])

    with op.batch_alter_table("scenes") as batch:
        batch.add_column(sa.Column("data", sa.JSON(), nullable=True))

    with op.batch_alter_table("shots") as batch:
        # episode_id 冗余一份到 shots，render_task 按集取镜头时不用 join scenes
        batch.add_column(sa.Column("episode_id", sa.String(ID_LEN), nullable=True))
        batch.add_column(sa.Column("image_path", sa.String(1024), nullable=True))
        batch.add_column(sa.Column("status", sa.String(50), nullable=False, server_default="planned"))
        batch.add_column(sa.Column("data", sa.JSON(), nullable=True))
        # 外键必须显式命名：SQLite batch 模式和 downgrade 的 drop_constraint 都要靠名字找到它
        batch.create_foreign_key("fk_shots_episode", "episodes", ["episode_id"], ["episode_id"])
        batch.create_index("ix_shots_episode_id", ["episode_id"])

    with op.batch_alter_table("assets") as batch:
        batch.add_column(sa.Column("kind", sa.String(20), nullable=False, server_default="image"))
        batch.add_column(sa.Column("qc_status", sa.String(20), nullable=False, server_default="pending"))
        batch.add_column(sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"))
        batch.add_column(sa.Column("retry_level", sa.String(30), nullable=True))
        batch.add_column(sa.Column("width", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("height", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("storage_key", sa.String(1024), nullable=True))
        batch.create_index("ix_assets_shot_id", ["shot_id"])

    op.create_table(
        "story_bibles",
        sa.Column("story_bible_id", sa.String(ID_LEN), primary_key=True),
        sa.Column("project_id", sa.String(ID_LEN), sa.ForeignKey("projects.project_id"), nullable=False, index=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("season_arcs", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "prompts",
        sa.Column("prompt_id", sa.String(ID_LEN), primary_key=True),
        sa.Column("shot_id", sa.String(ID_LEN), sa.ForeignKey("shots.shot_id"), nullable=False, index=True),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "qc_reports",
        sa.Column("qc_report_id", sa.String(ID_LEN), primary_key=True),
        sa.Column("asset_id", sa.String(ID_LEN), sa.ForeignKey("assets.asset_id"), nullable=True, index=True),
        sa.Column("shot_id", sa.String(ID_LEN), nullable=True, index=True),
        sa.Column("decision", sa.String(20), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "llm_usage",
        sa.Column("usage_id", sa.String(ID_LEN), primary_key=True),
        sa.Column("context_id", sa.String(ID_LEN), nullable=True, index=True),
        sa.Column("run_id", sa.String(ID_LEN), nullable=True, index=True),
        sa.Column("agent", sa.String(100), nullable=True),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "pipeline_runs",
        sa.Column("run_id", sa.String(ID_LEN), primary_key=True),
        sa.Column("project_id", sa.String(ID_LEN), sa.ForeignKey("projects.project_id"), nullable=False, index=True),
        sa.Column("episode_id", sa.String(ID_LEN), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="queued", index=True),
        sa.Column("stage", sa.String(50), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "publish_records",
        sa.Column("publish_id", sa.String(ID_LEN), primary_key=True),
        sa.Column("episode_id", sa.String(ID_LEN), nullable=False, index=True),
        sa.Column("platform", sa.String(50), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("remote_id", sa.String(255), nullable=True),
        sa.Column("remote_url", sa.String(1024), nullable=True),
        sa.Column("dry_run", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "analytics_events",
        sa.Column("event_id", sa.String(ID_LEN), primary_key=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("user_id", sa.String(ID_LEN), nullable=True, index=True),
        sa.Column("episode_id", sa.String(ID_LEN), nullable=True, index=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("surface", sa.String(50), nullable=True),
        sa.Column("amount_usd", sa.Float(), nullable=True),
        sa.Column("experiment", sa.String(100), nullable=True),
        sa.Column("variant", sa.String(50), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("raw", sa.JSON(), nullable=True),
    )
    op.create_table(
        "analytics_metrics",
        sa.Column("metric_id", sa.String(ID_LEN), primary_key=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("episode_id", sa.String(ID_LEN), nullable=True, index=True),
        sa.Column("remote_id", sa.String(255), nullable=True),
        sa.Column("day", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("views", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("likes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("comments", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("shares", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("watch_time_minutes", sa.Float(), nullable=False, server_default="0"),
        sa.Column("estimated_revenue_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("raw", sa.JSON(), nullable=True),
    )
    # 同一来源同一视频同一天只能有一行，拉取重复数据时是更新而不是新增
    op.create_index("ix_metrics_source_remote_day", "analytics_metrics", ["source", "remote_id", "day"], unique=True)
    op.create_table(
        "analytics_snapshots",
        sa.Column("snapshot_id", sa.String(ID_LEN), primary_key=True),
        sa.Column("metric", sa.String(50), nullable=False, index=True),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    """回滚到 f7d0958c1f12：先删九张新表，再逐表删掉补的索引/外键/列。
    id 列宽不缩回 36：缩窄可能截断已有的带前缀主键，回滚也不该丢数据。"""
    for table in ("analytics_snapshots", "analytics_metrics", "analytics_events", "publish_records", "pipeline_runs", "llm_usage", "qc_reports", "prompts", "story_bibles"):
        op.drop_table(table)
    # SQLite 的 batch 模式重建表时不会自动带走旧索引，先显式删掉再删列
    with op.batch_alter_table("assets") as batch:
        batch.drop_index("ix_assets_shot_id")
        for col in ("kind", "qc_status", "attempt", "retry_level", "width", "height", "storage_key"):
            batch.drop_column(col)
    with op.batch_alter_table("shots") as batch:
        batch.drop_index("ix_shots_episode_id")
        batch.drop_constraint("fk_shots_episode", type_="foreignkey")
        for col in ("episode_id", "image_path", "status", "data"):
            batch.drop_column(col)
    with op.batch_alter_table("scenes") as batch:
        batch.drop_column("data")
    with op.batch_alter_table("episodes") as batch:
        batch.drop_index("ix_episodes_project_number")
        for col in ("review_status", "review_notes", "season_id", "data", "script", "editorial", "video_path", "manifest_path", "updated_at"):
            batch.drop_column(col)
    with op.batch_alter_table("characters") as batch:
        for col in ("role", "lora_path", "data"):
            batch.drop_column(col)
