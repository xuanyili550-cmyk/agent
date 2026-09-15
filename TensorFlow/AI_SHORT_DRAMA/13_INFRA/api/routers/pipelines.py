"""端到端流水线接口。

POST /pipelines/episodes            一句创意 -> 建 PipelineRun -> 入队故事阶段
GET  /pipelines/{run_id}            进度 / 阶段 / 结果（含编审记录、失败镜头、成片路径）
GET  /pipelines                     列表（按状态筛）
POST /pipelines/{run_id}/approve    人工审核通过 -> 进入生产（镜头生成 -> 渲染 -> manifest -> 发布）
POST /pipelines/{run_id}/reject     人工打回
POST /pipelines/{run_id}/rerender   镜头补生成后重新渲染
GET  /pipelines/{run_id}/usage      这次运行的 LLM token / 成本
POST /pipelines/{run_id}/assistant  向制片助理 Agent（带工具循环）提问：查角色/剧本、跑硬校验、标记人工复核

路由层只做参数校验、404/409 转换和响应组装，状态流转全部委托给 ``orchestration`` 模块。
各写接口末尾都 ``db.expire_all()`` 再重读：orchestration 用的是自己的 session_scope 提交的，
当前请求的 session 里缓存的 run 对象已经过期，不刷新会把旧状态返回给调用方。
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from ... import orchestration
from ...database.models import Episode, LLMUsage, PipelineRun
from ...database.repository import unscoped_id
from ...queue.assistant_task import assistant_task
from ..deps import get_db
from ..pagination import Page, page_params, paginate
from ..schemas import (
    PipelineApproveRequest,
    PipelineAssistantRequest,
    PipelineAssistantResponse,
    PipelineCreateRequest,
    PipelineRejectRequest,
    PipelineRunRead,
    PipelineUsageRead,
)
from ..security import require_api_key

router = APIRouter(prefix="/pipelines", tags=["pipelines"], dependencies=[Depends(require_api_key)])


def _run_or_404(db: Session, run_id: str) -> PipelineRun:
    """取 PipelineRun，不存在直接抛 404。"""
    run = db.get(PipelineRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="PipelineRun not found")
    return run


def _read(run: PipelineRun, db: Session) -> PipelineRunRead:
    """把 ORM 的 run 组装成响应体，并附带各集摘要（状态、审核状态、成片/manifest 路径）。

    集列表从 run.result["episode_ids"] 逐个查，而不是走 ORM 关系：episode_ids 是故事阶段写进 result 的，
    只包含这次 run 产出的集，项目下其他 run 的集不该混进来。business_id 是去掉项目前缀的业务 id，前端展示用。
    """
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
    """一句创意启动流水线：建 run -> 入队故事阶段 -> 立刻返回 202 和当前 run 状态。

    project_id 单独传给 create_run，其余字段作为 run.params 存下来（exclude_none：没传的不落库，worker 端用默认值）。
    """
    try:
        run_id = orchestration.create_run(payload.project_id, payload.model_dump(exclude={"project_id"}, exclude_none=True))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    try:
        orchestration.start_pipeline(run_id)
    except Exception as exc:  # eager 模式下故事阶段的异常会同步冒上来；run.error 已经记录
        db.expire_all()
        # run 已被任务标为 failed 的话不再抛 500：调用方从响应体的 status/error 里就能看到失败原因
        if _run_or_404(db, run_id).status != "failed":
            raise HTTPException(status_code=500, detail=str(exc)[:300])
    db.expire_all()
    return _read(_run_or_404(db, run_id), db)


@router.get("", response_model=List[PipelineRunRead])
def list_pipelines(response: Response, status: Optional[str] = None, db: Session = Depends(get_db), page: Page = Depends(page_params)) -> List[PipelineRunRead]:
    """分页列出 run（按创建时间倒序），可用 ``?status=`` 只看某个状态（如 awaiting_review 待审列表）。"""
    query = db.query(PipelineRun).order_by(PipelineRun.created_at.desc())
    if status:
        query = query.filter(PipelineRun.status == status)
    return [_read(run, db) for run in paginate(query, page, response)]


@router.get("/{run_id}", response_model=PipelineRunRead)
def get_pipeline(run_id: str, db: Session = Depends(get_db)) -> PipelineRunRead:
    """查单个 run 的进度、阶段、结果和各集摘要；前端轮询这个接口刷新进度。"""
    return _read(_run_or_404(db, run_id), db)


@router.post("/{run_id}/approve", response_model=PipelineRunRead)
def approve_pipeline(run_id: str, payload: PipelineApproveRequest, db: Session = Depends(get_db)) -> PipelineRunRead:
    """人工批准并进入生产。run 状态不允许批准（比如已在生产中）时返回 409。"""
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
    """人工打回：所有集标 rejected，run 状态置 rejected；之后仍可修改剧本后再 approve。"""
    _run_or_404(db, run_id)
    orchestration.reject_run(run_id, payload.notes)
    db.expire_all()
    return _read(_run_or_404(db, run_id), db)


@router.post("/{run_id}/rerender", response_model=PipelineRunRead)
def rerender_pipeline(run_id: str, db: Session = Depends(get_db)) -> PipelineRunRead:
    """镜头人工补生成/替换之后，跳过故事和镜头阶段，直接重跑渲染 -> manifest -> 发布。"""
    _run_or_404(db, run_id)
    orchestration.rerender_run(run_id)
    db.expire_all()
    return _read(_run_or_404(db, run_id), db)


@router.get("/{run_id}/usage", response_model=PipelineUsageRead)
def pipeline_usage(run_id: str, db: Session = Depends(get_db)) -> PipelineUsageRead:
    """汇总这次 run 的 LLM 用量：总量一条聚合查询，按 agent 拆分再一条 group by 查询。

    聚合在数据库里做而不是把 LLMUsage 行拉出来在 Python 里加：一次 run 可能有几十上百次调用（含编审修订轮），
    ``coalesce(..., 0)`` 保证没有任何记录时也返回 0 而不是 None。
    """
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
    # 按 agent 分组时 sum 可能是 None（理论上不会，因为有 filter 才有行），int(... or 0) 兜底
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


@router.post("/{run_id}/assistant", response_model=PipelineAssistantResponse, status_code=202)
def ask_assistant(run_id: str, payload: PipelineAssistantRequest, db: Session = Depends(get_db)) -> PipelineAssistantResponse:
    """把问题交给制片助理 Agent（assistant_task，llm 队列）。

    LLM 调用不在 API 进程里做：和故事阶段一样入队，API 只负责校验和返回 task_id。
    eager 模式（本地/测试）任务同步跑完，result 直接带回；生产模式用 GET /tasks/{task_id} 轮询。
    """
    run = _run_or_404(db, run_id)
    if not (run.result or {}).get("episode_ids"):
        raise HTTPException(status_code=409, detail="这次运行还没有产出剧集（故事阶段未完成），制片助理没有东西可查")
    async_result = assistant_task.delay({"run_id": run_id, "question": payload.question, "context_id": payload.context_id})
    result = async_result.result if async_result.successful() and isinstance(async_result.result, dict) else None
    error = str(async_result.result)[:500] if async_result.failed() else None
    return PipelineAssistantResponse(run_id=run_id, task_id=async_result.id, status=async_result.status, result=result, error=error)
