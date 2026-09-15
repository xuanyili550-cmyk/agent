"""端到端流水线接口。

POST /pipelines/episodes            一句创意 -> 建 PipelineRun -> 入队故事阶段
GET  /pipelines/{run_id}            进度 / 阶段 / 结果（含编审记录、失败镜头、成片路径）
GET  /pipelines                     列表（按状态筛）
POST /pipelines/{run_id}/approve    人工审核通过 -> 进入生产（镜头生成 -> 渲染 -> manifest -> 发布）
POST /pipelines/{run_id}/reject     人工打回
POST /pipelines/{run_id}/rerender   镜头补生成后重新渲染
GET  /pipelines/{run_id}/usage      这次运行的 LLM token / 成本
GET  /pipelines/{run_id}/stream     SSE 实时推送进度（故事阶段按 LangGraph 节点逐个推）
GET  /pipelines/{run_id}/checkpoint LangGraph 检查点状态：已完成 / 待执行节点，看得出重试会从哪续跑
POST /pipelines/{run_id}/assistant  向制片助理 Agent（带工具循环）提问：查角色/剧本、跑硬校验、标记人工复核
POST /pipelines/{run_id}/assistant/stream  同上，但用 SSE 流式返回（模型 token 级增量 + 每步工具调用）

路由层只做参数校验、404/409 转换和响应组装，状态流转全部委托给 ``orchestration`` 模块。
各写接口末尾都 ``db.expire_all()`` 再重读：orchestration 用的是自己的 session_scope 提交的，
当前请求的 session 里缓存的 run 对象已经过期，不刷新会把旧状态返回给调用方。
"""

from __future__ import annotations

import importlib
import json
import time
from typing import Iterator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from ... import orchestration
from ...config import get_settings
from ...database.models import Episode, LLMUsage, PipelineRun
from ...database.repository import unscoped_id
from ...database.session import session_scope
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


TERMINAL_STATUSES = {"done", "failed", "rejected", "awaiting_review"}


def _sse(event: str, payload: dict) -> str:
    """拼一条 SSE 报文。``ensure_ascii=False`` 让中文在浏览器里直接可读，便于抓包排查。"""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@router.get("/{run_id}/stream")
def stream_pipeline_progress(run_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
    """SSE 实时推送这次运行的进度，直到进入终态或超时。

    数据来自 ``PipelineRun``：故事阶段的 worker 每跑完一个 LangGraph 节点就把进度写进
    ``result["progress"]``（见 queue/story_task.py 的 ``_progress_writer``），这里只负责把变化推出去。

    为什么用"轮询数据库 + SSE"而不是让 worker 直连浏览器：worker 和 API 是不同进程（生产上还在不同
    机器），而进度必须让任意一个 API 副本都能读到、刷新页面后还能看到历史。数据库是这两点的最小公约数；
    轮询间隔 1 秒对一条 SQL 来说毫不费力，却省掉了一整套 pub/sub 依赖。

    每次只在 (status, stage, 进度条数) 发生变化时推送，避免把同样的内容刷给前端。
    浏览器的 EventSource 不能带自定义头，所以前端用 fetch + ReadableStream 读这个流（照样带 X-API-Key）。
    """
    _run_or_404(db, run_id)
    settings = get_settings()

    def events() -> Iterator[str]:
        """逐次比对 run 的状态，有变化就推一条；终态或超时后推 end 并结束。"""
        deadline = time.monotonic() + settings.stream_max_seconds
        last: tuple | None = None
        while True:
            # 每轮开一个短事务：SSE 可能挂很久，不能一直占着连接
            with session_scope() as inner:
                run = inner.get(PipelineRun, run_id)
                if run is None:
                    yield _sse("error", {"detail": "PipelineRun 已被删除"})
                    return
                result = run.result or {}
                progress = list(result.get("progress") or [])
                snapshot = {
                    "run_id": run_id,
                    "status": run.status,
                    "stage": run.stage,
                    "error": run.error,
                    "progress": progress,
                    "progress_nodes": list(result.get("progress_nodes") or []),
                    "progress_total": result.get("progress_total"),
                    "shots_total": result.get("shots_total"),
                    "episode_ids": list(result.get("episode_ids") or []),
                }
                fingerprint = (run.status, run.stage, len(progress))
                terminal = run.status in TERMINAL_STATUSES
            if fingerprint != last:
                last = fingerprint
                yield _sse("progress", snapshot)
            if terminal:
                yield _sse("end", {"run_id": run_id, "status": snapshot["status"]})
                return
            if time.monotonic() >= deadline:
                # 不无限挂着：告诉前端"还没结束，自己重连"，顺手释放服务端连接
                yield _sse("timeout", {"run_id": run_id, "status": snapshot["status"]})
                return
            time.sleep(settings.stream_poll_seconds)

    # X-Accel-Buffering=no：经 nginx 反代时禁用缓冲，否则 SSE 会被攒成一坨再发出来
    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@router.get("/{run_id}/checkpoint")
def pipeline_checkpoint(run_id: str, db: Session = Depends(get_db)) -> dict:
    """这次运行的 LangGraph 检查点状态：已完成节点、待执行节点、是否跑完。

    看这个就知道"任务再被重试会从哪继续"。关掉 ``STORY_CHECKPOINT_ENABLED`` 时 enabled=false。
    """
    _run_or_404(db, run_id)
    settings = get_settings()
    if not settings.story_checkpoint_enabled:
        return {"run_id": run_id, "enabled": False, "completed": [], "pending": [], "finished": False}
    story_task_mod = importlib.import_module("13_INFRA.queue.story_task")
    nodes = story_task_mod.STORY_NODES
    thread_id = f"run:{run_id}"
    checkpointer = story_task_mod.pipeline_checkpointer(settings, thread_id)
    # 只读检查点，不重建整张图：构造真图要先把六个 Agent 和 provider 建起来（本地模型会加载权重），
    # 而这里只想知道"跑到哪了"。checkpointer.list() 已经够用。
    steps = [int((item.metadata or {}).get("step", -1)) for item in checkpointer.list({"configurable": {"thread_id": thread_id}})]
    if not steps:
        return {"run_id": run_id, "enabled": True, "nodes": nodes, "completed": [], "pending": [], "finished": False, "steps": 0}
    # LangGraph 的 step=N 表示前 N 个节点已经执行完（step=-1 是入口前的初始快照）
    step = max(steps)
    return {
        "run_id": run_id,
        "enabled": True,
        "nodes": nodes,
        "completed": nodes[: max(step, 0)],
        "pending": nodes[step : step + 1] if 0 <= step < len(nodes) else [],
        "finished": step >= len(nodes),
        "steps": max(step, 0),
    }


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


@router.post("/{run_id}/assistant/stream")
def stream_assistant(run_id: str, payload: PipelineAssistantRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    """流式版制片助理：SSE 推送模型 token 增量、每一步工具调用和最终答案。

    **这条路径在 API 进程里直接跑 Agent，不入队**，和上面的 ``POST .../assistant`` 是两种用法：
    - 队列版（202 + task_id）：非交互场景、批量任务，LLM 调用归 worker，API 不被长任务占住；
    - 流式版（本接口）：人坐在页面前等着看，必须有一条活着的连接才能边生成边推，
      经过队列就没法把 token 增量送回这条 HTTP 连接了。

    代价是这个请求会占住一个 API 工作线程若干秒——所以只给交互式使用，
    并发高的自动化流程应该走队列版。用量记账、工具指标、确认门和队列版完全一致。
    """
    run = _run_or_404(db, run_id)
    if not (run.result or {}).get("episode_ids"):
        raise HTTPException(status_code=409, detail="这次运行还没有产出剧集（故事阶段未完成），制片助理没有东西可查")
    settings = get_settings()
    context_id = payload.context_id or f"assistant:{run_id}"
    assistant_mod = importlib.import_module("13_INFRA.queue.assistant_task")
    common = importlib.import_module("13_INFRA.queue._common")
    agents = importlib.import_module("02_STORY_ENGINE.agents")

    def events() -> Iterator[str]:
        """构造 Agent 并把 run_stream 的事件逐条转成 SSE。"""
        try:
            provider = common.build_llm_provider(settings)
            ctx = assistant_mod.build_context_from_db(run_id)
            memory = agents.build_memory(provider, context_id, common.build_conversation_store(settings))
            memory.lock_timeout = settings.llm_context_lock_timeout_seconds
            agent = agents.DramaAssistantAgent(provider, ctx, memory=memory)
            common.instrument_tool_agent(agent, settings, context_id=context_id, run_id=run_id)
            yield _sse("start", {"run_id": run_id, "context_id": context_id, "question": payload.question})
            for event in agent.run_stream(payload.question):
                yield _sse(event["event"], event)
            yield _sse("end", {"run_id": run_id, "flagged": list(ctx.flagged)})
        except Exception as exc:
            # 流已经开始了就不能再改 HTTP 状态码，只能把错误作为一条事件推下去让前端显示
            yield _sse("error", {"detail": f"{type(exc).__name__}: {str(exc)[:300]}"})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )
