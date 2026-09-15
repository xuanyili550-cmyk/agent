"""剧集 manifest 任务：渲染完成后为每集写 10_EPISODES 格式的 episode_manifest.json 并回写数据库。

在流水线里的位置：render_task -> **manifest_task** -> publish_task。manifest 是渲染和发布之间的交接件：
发布环节（11_PUBLISH）只认 manifest 文件，不直接读数据库，所以这里把成片路径、时长、镜头列表、
QC 报告引用等从库里整理成文件；每个平台的发布状态也从这份 manifest 开始跟踪。
"""

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
    """为一集生成 episode_manifest.json（以及可选的 qc_report.json），并把路径写回 Episode.manifest_path。

    ``episode_id`` 是数据库主键（带项目前缀），``render`` 是 render_task 对这一集的结果。
    manifest 里的 episode_id 用 ``unscoped_id`` 还原成业务 id（ep_001）：manifest 是给发布端和人看的
    交付物，不该暴露数据库内部的前缀格式。
    """
    settings = get_settings()
    em = mod("10_EPISODES.episode_manifest")
    out_dir = Path(settings.artifacts_root) / "episodes" / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with session_scope() as db:
        row = db.get(Episode, episode_id)
        if row is None:
            raise LookupError(f"Episode {episode_id} 不存在")
        episode = repo.load_episode(db, episode_id)
        # 标题优先用 03 schema 对象里的，手工建的空集没有 data 就退回行上的 title，再不行用 id 兜底
        title = (episode.title if episode else row.title) or episode_id
        number = row.episode_number
        # QC 报告按数据库镜头主键查（rendered_shot_keys）；旧结果没有这个字段时退回业务 id
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
            # 每个目标平台先占一个初始状态位，publish_task 之后逐个更新
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
    """链式任务：接 render_task 的结果，为其中每一集写 manifest，并把路径列表记进 PipelineRun.result。
    series_id 取 run 所属的 project_id——一个项目就是一部剧。"""
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
