"""剧集 manifest 任务：渲染完成后为每集写 10_EPISODES 格式的 episode_manifest.json 并回写数据库。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

from ..config import get_settings
from ..database import repository as repo
from ..database.models import Episode, QCReportRow
from ..database.session import session_scope
from ..workers.celery_app import celery_app
from ._common import mod

__all__ = ["manifest_task", "write_episode_manifest"]


def write_episode_manifest(episode_id: str, render: Dict[str, Any], *, series_id: str, platforms: list[str]) -> Dict[str, Any]:
    settings = get_settings()
    em = mod("10_EPISODES.episode_manifest")
    out_dir = Path(settings.artifacts_root) / "episodes" / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with session_scope() as db:
        row = db.get(Episode, episode_id)
        if row is None:
            raise LookupError(f"Episode {episode_id} 不存在")
        episode = repo.load_episode(db, episode_id)
        title = (episode.title if episode else row.title) or episode_id
        number = row.episode_number
        qc_rows = (
            db.query(QCReportRow)
            .filter(QCReportRow.shot_id.in_(render.get("rendered_shot_keys") or render["rendered_shots"]))
            .order_by(QCReportRow.created_at.desc())
            .all()
        )
        qc_ref = None
        if qc_rows:
            qc_path = out_dir / "qc_report.json"
            import json

            qc_path.write_text(json.dumps([r.data for r in qc_rows], ensure_ascii=False, indent=2), encoding="utf-8")
            qc_ref = str(qc_path)

        manifest = em.EpisodeManifest(
            episode_id=repo.unscoped_id(episode_id),
            series_id=series_id,
            episode_number=number,
            title=title,
            video_path=render["video_path"],
            duration_sec=float(render["duration_sec"]),
            subtitle_path=render.get("subtitle_path"),
            shot_ids=list(render["rendered_shots"]),
            qc_report_ref=qc_ref,
            publish_status_per_platform={p: em.PlatformPublishStatus(platform=p) for p in platforms},
            created_at=datetime.now(UTC),
            language="zh",
        )
        path = out_dir / "episode_manifest.json"
        manifest.save(path)
        row.manifest_path = str(path)
    return {"episode_id": episode_id, "manifest_path": str(path)}


@celery_app.task(name="manifest_task", bind=True)
def manifest_task(self, render_result: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    settings = get_settings()
    from ..database.models import PipelineRun

    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        series_id = run.project_id
        repo.update_run(db, run_id, stage="manifest")
    manifests = [
        write_episode_manifest(r["episode_id"], r, series_id=series_id, platforms=settings.publish_platforms) for r in render_result.get("renders", [])
    ]
    with session_scope() as db:
        repo.update_run(db, run_id, stage="manifest_done", manifests=manifests)
    return {"run_id": run_id, "manifests": manifests}
