"""把 03_STRUCTURED_DATA 对象接进数据库：JSON data 列、业务 id 主键、prompts/qc/usage/pipeline/publish/analytics 表

Revision ID: a3c9e1d2f4b5
Revises: f7d0958c1f12
Create Date: 2026-09-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a3c9e1d2f4b5"
down_revision: Union[str, Sequence[str], None] = "f7d0958c1f12"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ID_LEN = 128


def upgrade() -> None:
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
        batch.add_column(sa.Column("episode_id", sa.String(ID_LEN), nullable=True))
        batch.add_column(sa.Column("image_path", sa.String(1024), nullable=True))
        batch.add_column(sa.Column("status", sa.String(50), nullable=False, server_default="planned"))
        batch.add_column(sa.Column("data", sa.JSON(), nullable=True))
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
    op.create_index("ix_metrics_source_remote_day", "analytics_metrics", ["source", "remote_id", "day"], unique=True)
    op.create_table(
        "analytics_snapshots",
        sa.Column("snapshot_id", sa.String(ID_LEN), primary_key=True),
        sa.Column("metric", sa.String(50), nullable=False, index=True),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
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
