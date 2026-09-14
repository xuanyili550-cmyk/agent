"""渲染任务（chord 回调）：收集本集所有镜头任务的结果，用 09_POST 出成片。

只用 QC 通过的镜头；没有任何镜头通过时把 run 标为 failed。失败镜头 id 写进 run.result，
人工可以只补这几个镜头再重渲染（POST /pipelines/{run_id}/rerender）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..config import get_settings
from ..database import repository as repo
from ..database.models import Episode, PipelineRun, Shot
from ..database.session import session_scope
from ..observability import PIPELINE_RUNS, get_logger
from ..workers.celery_app import celery_app
from ._common import mod

__all__ = ["render_task", "render_episode_from_db"]

log = get_logger("queue.render")


def render_episode_from_db(run_id: str, episode_id: str) -> Dict[str, Any]:
    settings = get_settings()
    assemble = mod("09_POST.assemble")
    with session_scope() as db:
        shots = repo.load_shots_for_episode(db, episode_id)
        scenes = repo.load_scenes_for_episode(db, episode_id)
        dialogue_by_shot: dict[str, list[str]] = {}
        for scene in scenes:
            for line in scene.dialogue:
                dialogue_by_shot.setdefault(line.shot_id, []).append(line.line_zh)
        media: List[Any] = []
        skipped: list[str] = []
        rendered_keys: list[str] = []
        for shot_db_id, shot in shots:
            row = db.get(Shot, shot_db_id)
            if row is None or row.status != "approved" or not (row.video_path or row.image_path):
                skipped.append(shot.id)
                continue
            rendered_keys.append(shot_db_id)
            media.append(
                assemble.ShotMedia(
                    shot_id=shot.id, duration_sec=shot.duration, video_path=row.video_path, image_path=row.image_path, dialogue=dialogue_by_shot.get(shot.id)
                )
            )
    if not media:
        raise RuntimeError(f"{episode_id} 没有任何 QC 通过的镜头，无法渲染")

    business_episode_id = repo.unscoped_id(episode_id)
    out_dir = Path(settings.artifacts_root) / "episodes" / episode_id
    final = assemble.render_from_media(business_episode_id, media, out_dir, width=settings.image_width, height=settings.image_height)
    duration = sum(m.duration_sec for m in media)
    with session_scope() as db:
        row = db.get(Episode, episode_id)
        if row:
            row.video_path = str(final)
            row.status = "rendered"
    srt = final.with_suffix(".srt")
    return {
        "episode_id": episode_id,
        "video_path": str(final),
        "subtitle_path": str(srt) if srt.exists() else None,
        "duration_sec": duration,
        "rendered_shots": [m.shot_id for m in media],
        "rendered_shot_keys": rendered_keys,
        "skipped_shots": skipped,
    }


@celery_app.task(name="render_task", bind=True)
def render_task(self, shot_results: List[Dict[str, Any]] | Dict[str, Any] | None, run_id: str) -> Dict[str, Any]:
    if isinstance(shot_results, dict):
        shot_results = [shot_results]
    shot_results = shot_results or []
    failed = [r.get("business_shot_id", r["shot_id"]) for r in shot_results if r.get("status") != "approved"]
    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        episode_ids = list((run.result or {}).get("episode_ids") or [])
        repo.update_run(
            db,
            run_id,
            status="rendering",
            stage="render",
            failed_shots=failed,
            approved_shots=[r.get("business_shot_id", r["shot_id"]) for r in shot_results if r.get("status") == "approved"],
        )
    PIPELINE_RUNS.labels("rendering").inc()
    try:
        renders = [render_episode_from_db(run_id, ep_id) for ep_id in episode_ids]
    except Exception as exc:
        with session_scope() as db:
            repo.update_run(db, run_id, status="failed", stage="render", error=f"{type(exc).__name__}: {str(exc)[:500]}")
        PIPELINE_RUNS.labels("failed").inc()
        raise
    with session_scope() as db:
        repo.update_run(db, run_id, status="rendered", stage="render_done", renders=renders)
    return {"run_id": run_id, "renders": renders, "failed_shots": failed}
