"""端到端流水线接口。

POST /pipelines/episodes            一句创意 -> 建 PipelineRun -> 入队故事阶段
GET  /pipelines/{run_id}            进度 / 阶段 / 结果（含编审记录、失败镜头、成片路径）
GET  /pipelines                     列表（按状态筛）
POST /pipelines/{run_id}/approve    人工审核通过 -> 进入生产（镜头生成 -> 渲染 -> manifest -> 发布）
POST /pipelines/{run_id}/reject     人工打回
POST /pipelines/{run_id}/rerender   镜头补生成后重新渲染
GET  /pipelines/{run_id}/usage      这次运行的 LLM token / 成本
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from ... import orchestration
from ...database.models import Episode, LLMUsage, PipelineRun
from ...database.repository import unscoped_id
from ..deps import get_db
from ..pagination import Page, page_params, paginate
from ..schemas import (
    PipelineApproveRequest,
    PipelineCreateRequest,
    PipelineRejectRequest,
    PipelineRunRead,
    PipelineUsageRead,
)
from ..security import require_api_key

router = APIRouter(prefix="/pipelines", tags=["pipelines"], dependencies=[Depends(require_api_key)])


def _run_or_404(db: Session, run_id: str) -> PipelineRun:
    run = db.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="PipelineRun not found")
    return run


def _read(run: PipelineRun, db: Session) -> PipelineRunRead:
    episodes = []
    for ep_id in (run.result or {}).get("episode_ids") or []:
        row = db.get(Episode, ep_id)
        if row:
            episodes.append(
                {
                    "episode_id": row.episode_id,
                    "business_id": unscoped_id(row.episode_id),
                    "episode_number": row.episode_number,
                    "title": row.title,
                    "status": row.status,
                    "review_status": row.review_status,
                    "video_path": row.video_path,
                    "manifest_path": row.manifest_path,
                }
            )
    return PipelineRunRead(
        run_id=run.run_id,
        project_id=run.project_id,
        status=run.status,
        stage=run.stage,
        params=run.params or {},
        result=run.result or {},
        error=run.error,
        episodes=episodes,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@router.post("/episodes", response_model=PipelineRunRead, status_code=202)
def create_episode_pipeline(payload: PipelineCreateRequest, db: Session = Depends(get_db)) -> PipelineRunRead:
    try:
        run_id = orchestration.create_run(payload.project_id, payload.model_dump(exclude={"project_id"}, exclude_none=True))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        orchestration.start_pipeline(run_id)
    except Exception as exc:  # eager 模式下故事阶段的异常会同步冒上来；run.error 已经记录
        db.expire_all()
        if _run_or_404(db, run_id).status != "failed":
            raise HTTPException(status_code=500, detail=str(exc)[:300])
    db.expire_all()
    return _read(_run_or_404(db, run_id), db)


@router.get("", response_model=List[PipelineRunRead])
def list_pipelines(response: Response, status: Optional[str] = None, db: Session = Depends(get_db), page: Page = Depends(page_params)) -> List[PipelineRunRead]:
    query = db.query(PipelineRun).order_by(PipelineRun.created_at.desc())
    if status:
        query = query.filter(PipelineRun.status == status)
    return [_read(run, db) for run in paginate(query, page, response)]


@router.get("/{run_id}", response_model=PipelineRunRead)
def get_pipeline(run_id: str, db: Session = Depends(get_db)) -> PipelineRunRead:
    return _read(_run_or_404(db, run_id), db)


@router.post("/{run_id}/approve", response_model=PipelineRunRead)
def approve_pipeline(run_id: str, payload: PipelineApproveRequest, db: Session = Depends(get_db)) -> PipelineRunRead:
    _run_or_404(db, run_id)
    try:
        orchestration.approve_run(run_id, payload.episode_ids, payload.notes)
    except RuntimeError as exc:
        db.expire_all()
        if _run_or_404(db, run_id).status == "failed":
            return _read(_run_or_404(db, run_id), db)  # eager 模式：下游任务已把 run 标为 failed，错误在 run.error 里
        raise HTTPException(status_code=409, detail=str(exc))
    db.expire_all()
    return _read(_run_or_404(db, run_id), db)


@router.post("/{run_id}/reject", response_model=PipelineRunRead)
def reject_pipeline(run_id: str, payload: PipelineRejectRequest, db: Session = Depends(get_db)) -> PipelineRunRead:
    _run_or_404(db, run_id)
    orchestration.reject_run(run_id, payload.notes)
    db.expire_all()
    return _read(_run_or_404(db, run_id), db)


@router.post("/{run_id}/rerender", response_model=PipelineRunRead)
def rerender_pipeline(run_id: str, db: Session = Depends(get_db)) -> PipelineRunRead:
    _run_or_404(db, run_id)
    orchestration.rerender_run(run_id)
    db.expire_all()
    return _read(_run_or_404(db, run_id), db)


@router.get("/{run_id}/usage", response_model=PipelineUsageRead)
def pipeline_usage(run_id: str, db: Session = Depends(get_db)) -> PipelineUsageRead:
    _run_or_404(db, run_id)
    row = (
        db.query(
            func.count(LLMUsage.usage_id),
            func.coalesce(func.sum(LLMUsage.input_tokens), 0),
            func.coalesce(func.sum(LLMUsage.output_tokens), 0),
            func.coalesce(func.sum(LLMUsage.cost_usd), 0.0),
        )
        .filter(LLMUsage.run_id == run_id)
        .one()
    )
    by_agent = {
        agent or "unknown": {"calls": calls, "input_tokens": int(i or 0), "output_tokens": int(o or 0), "cost_usd": float(c or 0.0)}
        for agent, calls, i, o, c in db.query(
            LLMUsage.agent, func.count(LLMUsage.usage_id), func.sum(LLMUsage.input_tokens), func.sum(LLMUsage.output_tokens), func.sum(LLMUsage.cost_usd)
        )
        .filter(LLMUsage.run_id == run_id)
        .group_by(LLMUsage.agent)
        .all()
    }
    return PipelineUsageRead(run_id=run_id, calls=int(row[0]), input_tokens=int(row[1]), output_tokens=int(row[2]), cost_usd=float(row[3]), by_agent=by_agent)
