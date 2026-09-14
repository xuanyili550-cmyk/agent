"""分析任务：从各数据源拉取 -> 入库 -> 计算留存/CTR/收入/实验 -> 存快照。

数据源见 12_ANALYTICS/ingest：自家 App 事件（用户级）走 AnalyticsEvent 表，
YouTube/TikTok 只给聚合数走 AnalyticsMetric 表。计算逻辑复用 12_ANALYTICS 原有的
compute_* 函数（它们吃 DataFrame，这里从表里读出来喂给它们）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from ..database.models import AnalyticsEvent, AnalyticsMetric, AnalyticsSnapshot
from ..database.session import session_scope
from ..observability import get_logger
from ..workers.celery_app import celery_app
from ._common import mod

__all__ = ["analytics_task", "ingest_events", "compute_snapshots"]

log = get_logger("queue.analytics")


def ingest_events(source_name: str, source, since: Optional[datetime] = None) -> Dict[str, int]:
    """source 是 12_ANALYTICS.ingest.base.AnalyticsSource 的实例。"""
    events = 0
    metrics = 0
    with session_scope() as db:
        for item in source.fetch(since=since):
            if item.kind == "event":
                db.add(
                    AnalyticsEvent(
                        event_id=item.dedupe_key,
                        source=source_name,
                        user_id=item.user_id,
                        episode_id=item.episode_id,
                        event_type=item.event_type,
                        surface=item.surface,
                        amount_usd=item.amount_usd,
                        experiment=item.experiment,
                        variant=item.variant,
                        timestamp=item.timestamp,
                        raw=item.raw,
                    )
                )
                db.flush()
                events += 1
            else:
                existing = db.query(AnalyticsMetric).filter_by(source=source_name, remote_id=item.remote_id, day=item.timestamp).first()
                row = existing or AnalyticsMetric(source=source_name, remote_id=item.remote_id, day=item.timestamp)
                row.episode_id = item.episode_id
                row.views, row.likes, row.comments, row.shares = item.views, item.likes, item.comments, item.shares
                row.watch_time_minutes, row.estimated_revenue_usd, row.raw = item.watch_time_minutes, item.estimated_revenue_usd, item.raw
                if existing is None:
                    db.add(row)
                metrics += 1
    return {"events": events, "metrics": metrics}


def _events_frame(db) -> pd.DataFrame:
    rows = db.query(AnalyticsEvent).all()
    if not rows:
        return pd.DataFrame(columns=["user_id", "episode_id", "event_type", "surface", "amount_usd", "experiment", "variant", "timestamp"])
    df = pd.DataFrame(
        [
            {
                "user_id": r.user_id,
                "episode_id": r.episode_id,
                "event_type": r.event_type,
                "surface": r.surface,
                "amount_usd": r.amount_usd,
                "experiment": r.experiment,
                "variant": r.variant,
                "timestamp": r.timestamp,
            }
            for r in rows
        ]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["event_date"] = df["timestamp"].dt.floor("D")
    return df


def compute_snapshots(experiment: Optional[str] = None) -> Dict[str, Any]:
    retention = mod("12_ANALYTICS.retention.compute_retention")
    ctr = mod("12_ANALYTICS.ctr.compute_ctr")
    revenue = mod("12_ANALYTICS.revenue.compute_revenue")
    ab = mod("12_ANALYTICS.experiments.ab_framework")
    out: Dict[str, Any] = {}
    with session_scope() as db:
        df = _events_frame(db)
        if df.empty:
            return {"empty": True}
        out["retention"] = retention.compute_retention(df).to_dict(orient="records")
        ctr_df = df[df["surface"].notna()]
        out["ctr"] = ctr.compute_ctr(ctr_df).to_dict(orient="records") if not ctr_df.empty else []
        rev = revenue.compute_revenue(df.assign(amount_usd=df["amount_usd"].fillna(0.0)))
        rev["revenue_by_episode"] = rev["revenue_by_episode"].to_dict(orient="records")
        out["revenue"] = rev
        if experiment:
            exp_df = df[df["experiment"] == experiment]
            if not exp_df.empty:
                out["experiment"] = {"name": experiment, "summary": ab.summarize_experiment(exp_df, experiment).to_dict(orient="records")}
        metrics = db.query(AnalyticsMetric).all()
        out["platform_metrics"] = {
            "days": len(metrics),
            "views": int(sum(m.views for m in metrics)),
            "watch_time_minutes": float(sum(m.watch_time_minutes for m in metrics)),
            "estimated_revenue_usd": float(sum(m.estimated_revenue_usd for m in metrics)),
        }
        for name, data in out.items():
            db.add(AnalyticsSnapshot(metric=name, data=data if isinstance(data, dict) else {"rows": data}))
    return out


@celery_app.task(name="analytics_task", bind=True)
def analytics_task(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """payload: {"sources": [{"name": "file", "path": "..."}], "experiment": str | None, "since_days": int | None}"""
    payload = payload or {}
    ingest = mod("12_ANALYTICS.ingest")
    since = datetime.now(UTC) - timedelta(days=int(payload["since_days"])) if payload.get("since_days") else None
    ingested = {}
    for spec in payload.get("sources", []):
        source = ingest.build_source(spec)
        ingested[spec["name"]] = ingest_events(spec["name"], source, since=since)
    snapshots = compute_snapshots(payload.get("experiment"))
    return {"ingested": ingested, "snapshots": {k: (v if not isinstance(v, list) else len(v)) for k, v in snapshots.items()}}
