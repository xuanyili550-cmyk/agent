"""渲染任务（chord 回调）：收集本集所有镜头任务的结果，用 09_POST 出成片。

只用 QC 通过的镜头；没有任何镜头通过时把 run 标为 failed。失败镜头 id 写进 run.result，
人工可以只补这几个镜头再重渲染（POST /pipelines/{run_id}/rerender）。

在流水线里的位置：chord(shot_task × N) -> **render_task** -> manifest_task -> publish_task。
镜头素材、时长、对白都从数据库读而不是从 shot_results 里取，所以"补几个镜头后重渲染"和首次渲染走同一条路径。
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
    """按数据库里的镜头状态渲染一集：取 approved 且有画面的镜头，拼上场次对白做字幕，交给 09_POST.assemble 出片。

    返回渲染结果 dict，其中 ``rendered_shots`` 是业务镜头 id（给 manifest / 人看），
    ``rendered_shot_keys`` 是数据库主键（manifest_task 用它查 QC 报告），``skipped_shots`` 是没通过的镜头。
    一个通过的镜头都没有就抛 RuntimeError，由 render_task 把 run 标成 failed。
    """
    settings = get_settings()
    assemble = mod("09_POST.assemble")
    with session_scope() as db:
        shots = repo.load_shots_for_episode(db, episode_id)
        scenes = repo.load_scenes_for_episode(db, episode_id)
        # 对白挂在场次上而不是镜头上，先按 shot_id 归拢成字幕行列表
        dialogue_by_shot: dict[str, list[str]] = {}
        for scene in scenes:
            for line in scene.dialogue:
                dialogue_by_shot.setdefault(line.shot_id, []).append(line.line_zh)
        media: List[Any] = []
        skipped: list[str] = []
        rendered_keys: list[str] = []
        for shot_db_id, shot in shots:
            row = db.get(Shot, shot_db_id)
            # 只用 QC 通过且至少有关键帧或视频的镜头；其余记进 skipped 供人工补镜头
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

    # 成片文件名用业务 id（ep_001），产物目录仍按数据库主键分目录避免跨项目重名
    business_episode_id = repo.unscoped_id(episode_id)
    out_dir = Path(settings.artifacts_root) / "episodes" / episode_id
    final = assemble.render_from_media(business_episode_id, media, out_dir, width=settings.image_width, height=settings.image_height)
    duration = sum(m.duration_sec for m in media)
    with session_scope() as db:
        row = db.get(Episode, episode_id)
        if row:
            row.video_path = str(final)
            row.status = "rendered"
    # 09_POST 有对白时会在成片旁边写同名 .srt；没有就不带字幕路径
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
    """chord 回调：汇总各 shot_task 的结果（approved / failed 镜头列表记进 run.result），
    然后对 run.result.episode_ids 里的每一集调 render_episode_from_db。

    ``shot_results`` 可能是列表（正常 chord）、单个 dict（只有一个镜头时 Celery 不包列表）或 None（重渲染入口），
    先统一成列表。渲染异常时把 run 标 failed 并原样抛出，让 Celery 也记录失败。
    """
    if isinstance(shot_results, dict):
        shot_results = [shot_results]
    shot_results = shot_results or []
    # 对外展示用业务镜头 id；旧结果没有 business_shot_id 时退回 shot_id
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
