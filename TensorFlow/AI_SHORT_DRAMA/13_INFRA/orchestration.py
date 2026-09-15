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

本模块是 API 路由（routers/pipelines.py）和 Celery 任务之间的"胶水层"：路由只调这里的函数，
不直接碰 Celery 原语；各阶段的状态流转（queued -> scripted/awaiting_review -> producing -> rendering -> ...）
也集中在这里维护，PipelineRun 表是唯一的进度真相源。
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
    """新建一条 PipelineRun 记录（status=queued）并返回 run_id。

    先落库再入队（入队在 start_pipeline 里），这样即使 broker 挂了也有 run 记录可查；
    project 不存在抛 LookupError，由路由转成 404。
    """
    with session_scope() as db:
        if db.get(Project, project_id) is None:
            raise LookupError(f"Project {project_id} 不存在")
        run = PipelineRun(project_id=project_id, params=params, status="queued", stage="queued")
        db.add(run)
        # flush 拿到数据库生成的 run_id，但事务由 session_scope 在退出时统一提交
        db.flush()
        run_id = run.run_id
    PIPELINE_RUNS.labels("queued").inc()
    return run_id


def start_pipeline(run_id: str):
    """入队故事阶段 + 门控；返回 Celery AsyncResult。"""
    # chain：story_task 的返回值会作为第一个参数传给 pipeline_gate_task（它的签名是 (story_result, run_id)）
    return chain(story_task.s(run_id), pipeline_gate_task.s(run_id)).apply_async()


def _shot_payloads(db, run: PipelineRun, episode_ids: list[str]) -> list[Dict[str, Any]]:
    """把已批准各集的 Shot + ImagePrompt 从数据库读出来，组装成 shot_task 的 payload 列表。

    在生产阶段才读而不是剧本阶段就生成，是为了让人工审核时改过的 prompt / 参考角色能生效。
    没有 ImagePrompt 的镜头退化用 "地点，动作" 拼一个 prompt，保证每个镜头都能出图。
    """
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
                    # 参考角色优先取 ImagePrompt 里显式指定的，没有就用镜头本身出场的角色
                    "reference_character_ids": list((prompt.reference_character_ids if prompt else None) or shot.character),
                    "image_prompt": prompt.model_dump(mode="json") if prompt else None,
                    "output_dir": str(settings.artifacts_root / "shots" / run.run_id / shot.id),
                    "seed": prompt.seed if prompt else None,
                    # 生图后端：本次 run 的参数可覆盖全局配置
                    "model": (run.params or {}).get("image_backend") or settings.image_backend,
                }
            )
    return payloads


def approved_episode_ids(db, run: PipelineRun) -> list[str]:
    """返回这次 run 里 review_status == "approved" 的集 id，顺序和 run.result["episode_ids"] 一致。"""
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
        # 把 episode_ids 收窄为已批准的集：后面 render/manifest 只处理这些
        run.result = {**(run.result or {}), "episode_ids": episode_ids}
        repo.update_run(db, run_id, status="producing", stage="shots", shots_total=len(payloads))
        for ep_id in episode_ids:
            ep = db.get(Episode, ep_id)
            ep.status = "producing"
    PIPELINE_RUNS.labels("producing").inc()
    # 先提交事务再入队（with 块已退出）：否则 worker 可能在事务提交前就读到旧状态。
    # chord = 所有 shot_task 并行跑完后再执行 render_task（渲染需要全部镜头就位）；
    # 之后 chain 串上 manifest 和 publish，各自拿到上一步的返回值 + run_id。
    workflow = chain(
        chord([shot_task.s(p) for p in payloads], render_task.s(run_id)),
        manifest_task.s(run_id),
        publish_task.s(run_id),
    )
    return workflow.apply_async()


@celery_app.task(name="pipeline_gate_task", bind=True)
def pipeline_gate_task(self, story_result: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """剧本阶段之后的"门控"任务：决定是停下等人工审核，还是直接进入生产。

    做成 Celery 任务而不是普通函数，是因为它要接在 story_task 后面由 worker 异步触发；
    ``story_result`` 是 chain 传来的上一步返回值，这里不用它，只按数据库里的 review_status 判断。
    """
    settings = get_settings()
    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        approved = approved_episode_ids(db, run)
        # AI 编审（总编审 agent）打回的集：即使不要求人工审核也不能自动放行
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
            # 一集都没批准：停在 awaiting_review，等 approve_run / 单集 review 接口放行
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
        # 只有停在审核相关状态的 run 才能批准：生产中/已完成的 run 再批准会重复入队镜头任务
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
    """人工打回：该 run 下所有集标 rejected 并记录原因，run 状态置为 rejected（之后仍可 approve_run 重新放行）。"""
    with session_scope() as db:
        run = db.get(PipelineRun, run_id)
        if run is None:
            raise LookupError(f"PipelineRun {run_id} 不存在")
        for ep_id in (run.result or {}).get("episode_ids") or []:
            row = db.get(Episode, ep_id)
            if row:
                row.review_status = "rejected"
                row.review_notes = notes
        # 打回原因同时写进 run.error，列表页不用逐集展开就能看到
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
    # render_task 的第一个参数在正常流程里是 chord 汇总的镜头结果列表；这里没有前置镜头任务，传空列表，
    # render_task 会直接按数据库里已有的镜头素材渲染
    return chain(render_task.s([], run_id), manifest_task.s(run_id), publish_task.s(run_id)).apply_async()
