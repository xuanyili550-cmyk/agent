"""端到端编排：把队列任务串成"创意 -> 剧本 -> 编审 -> 镜头生成/QC -> 渲染 -> manifest -> 发布"。

    start_pipeline(run_id)
        chain( story_task(run_id) -> pipeline_gate_task(run_id) )
    pipeline_gate_task
        require_human_review 且没有 approved 的集  -> run.status = awaiting_review，停
        否则                                        -> start_production(run_id)
    start_production(run_id)
        chord( [shot_task(镜头 payload) ...]  ->  render_task(run_id) )
          | manifest_task(run_id) | publish_task(run_id)
    approve_run(run_id)   人工审核通过后调用 -> start_production

镜头 payload 在 start_production 时才从数据库读（剧本阶段落库的 Shot/ImagePrompt），
所以人工可以在审核阶段直接改数据库里的 prompt 再放行。
"""

from __future__ import annotations

from typing import Any, Dict

from celery import chain, chord

from .config import get_settings
from .database import repository as repo
from .database.models import Episode, PipelineRun, Project
from .database.session import session_scope
from .observability import PIPELINE_RUNS, get_logger
from .queue.manifest_task import manifest_task
from .queue.publish_task import publish_task
from .queue.render_task import render_task
from .queue.shot_task import shot_task
from .queue.story_task import story_task
from .workers.celery_app import celery_app

log = get_logger("orchestration")


def create_run(project_id: str, params: Dict[str, Any]) -> str:
    with session_scope() as db:
        if db.get(Project, project_id) is None:
            raise LookupError(f"Project {project_id} 不存在")
        run = PipelineRun(project_id=project_id, params=params, status="queued", stage="queued")
        db.add(run)
        db.flush()
        run_id = run.run_id
    PIPELINE_RUNS.labels("queued").inc()
    return run_id


def start_pipeline(run_id: str):
    """入队故事阶段 + 门控；返回 Celery AsyncResult。"""
    return chain(story_task.s(run_id), pipeline_gate_task.s(run_id)).apply_async()


def _shot_payloads(db, run: PipelineRun, episode_ids: list[str]) -> list[Dict[str, Any]]:
    settings = get_settings()
    payloads: list[Dict[str, Any]] = []
    for ep_id in episode_ids:
        for shot_db_id, shot in repo.load_shots_for_episode(db, ep_id):
            prompt = repo.load_image_prompt(db, shot_db_id)
            payloads.append(
                {
                    "shot_id": shot_db_id,  # 数据库 id（带项目前缀）；业务 id 在 shot.id
                    "business_shot_id": shot.id,
                    "project_id": run.project_id,
                    "episode_id": ep_id,
                    "run_id": run.run_id,
                    "prompt": prompt.prompt_text if prompt else f"{shot.location}，{shot.action}",
                    "negative_prompt": prompt.negative_prompt if prompt else None,
                    "reference_character_ids": list((prompt.reference_character_ids if prompt else None) or shot.character),
                    "image_prompt": prompt.model_dump(mode="json") if prompt else None,
                    "output_dir": str(settings.artifacts_root / "shots" / run.run_id / shot.id),
                    "seed": prompt.seed if prompt else None,
                    "model": (run.params or {}).get("image_backend") or settings.image_backend,
                }
            )
    return payloads


def approved_episode_ids(db, run: PipelineRun) -> list[str]:
    ids = list((run.result or {}).get("episode_ids") or [])
    approved = []
    for ep_id in ids:
        row = db.get(Episode, ep_id)
        if row and row.review_status == "approved":
            approved.append(ep_id)
    return approved


def start_production(run_id: str):
    """镜头生成（并行）-> 渲染 -> manifest -> 发布。只处理 review_status=approved 的集。"""
    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        if run is None:
            raise LookupError(f"PipelineRun {run_id} 不存在")
        episode_ids = approved_episode_ids(db, run)
        if not episode_ids:
            raise RuntimeError("没有已批准（review_status=approved）的集，无法进入生产")
        payloads = _shot_payloads(db, run, episode_ids)
        if not payloads:
            raise RuntimeError("已批准的集没有任何镜头")
        run.result = {**(run.result or {}), "episode_ids": episode_ids}
        repo.update_run(db, run_id, status="producing", stage="shots", shots_total=len(payloads))
        for ep_id in episode_ids:
            ep = db.get(Episode, ep_id)
            ep.status = "producing"
    PIPELINE_RUNS.labels("producing").inc()
    workflow = chain(
        chord([shot_task.s(p) for p in payloads], render_task.s(run_id)),
        manifest_task.s(run_id),
        publish_task.s(run_id),
    )
    return workflow.apply_async()


@celery_app.task(name="pipeline_gate_task", bind=True)
def pipeline_gate_task(self, story_result: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    settings = get_settings()
    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        approved = approved_episode_ids(db, run)
        needing = list((run.result or {}).get("episodes_needing_review") or [])
        episode_ids = list((run.result or {}).get("episode_ids") or [])
        if not settings.require_human_review:
            # 不要求人工审核：把 AI 编审没打回的集直接标 approved
            for ep_id in episode_ids:
                if ep_id in needing:
                    continue
                row = db.get(Episode, ep_id)
                if row:
                    row.review_status = "approved"
            db.flush()
            approved = approved_episode_ids(db, run)
        if not approved:
            repo.update_run(db, run_id, status="awaiting_review", stage="review")
            PIPELINE_RUNS.labels("awaiting_review").inc()
            log.info("流水线等待审核", extra={"run_id": run_id, "needing_review": needing})
            return {"run_id": run_id, "status": "awaiting_review"}
    start_production(run_id)
    return {"run_id": run_id, "status": "producing", "episodes": approved}


def approve_run(run_id: str, episode_ids: list[str] | None = None, notes: str | None = None) -> Dict[str, Any]:
    """人工批准：把指定集（默认全部）标 approved 后进入生产。"""
    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        if run is None:
            raise LookupError(f"PipelineRun {run_id} 不存在")
        if run.status not in ("awaiting_review", "scripted", "rejected"):
            raise RuntimeError(f"run 状态为 {run.status}，不能批准")
        targets = episode_ids or list((run.result or {}).get("episode_ids") or [])
        for ep_id in targets:
            row = db.get(Episode, ep_id)
            if row:
                row.review_status = "approved"
                if notes:
                    row.review_notes = notes
    start_production(run_id)
    return {"run_id": run_id, "approved": targets}


def reject_run(run_id: str, notes: str) -> Dict[str, Any]:
    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        if run is None:
            raise LookupError(f"PipelineRun {run_id} 不存在")
        for ep_id in (run.result or {}).get("episode_ids") or []:
            row = db.get(Episode, ep_id)
            if row:
                row.review_status = "rejected"
                row.review_notes = notes
        repo.update_run(db, run_id, status="rejected", stage="review", error=notes)
    PIPELINE_RUNS.labels("rejected").inc()
    return {"run_id": run_id, "status": "rejected"}


def rerender_run(run_id: str):
    """镜头补生成后重新渲染（跳过故事阶段）。"""
    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        if run is None:
            raise LookupError(f"PipelineRun {run_id} 不存在")
        repo.update_run(db, run_id, status="rendering", stage="render")
    return chain(render_task.s([], run_id), manifest_task.s(run_id), publish_task.s(run_id)).apply_async()
