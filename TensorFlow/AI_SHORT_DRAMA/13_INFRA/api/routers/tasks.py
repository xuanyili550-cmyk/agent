"""/tasks 路由：单个 Celery 任务的入队与状态查询（不经过流水线编排的"裸"任务接口）。

用途：调试某一类生成任务、外部系统按需调用单步能力（比如只生一张图、只做一次 TTS）。
完整的剧集生产走 /pipelines。payload 按队列用 task_payloads 里的模型校验，拼错字段在入口就 422。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from ...queue.analytics_task import analytics_task
from ...queue.assistant_task import assistant_task
from ...queue.image_task import image_task
from ...queue.lipsync_task import lipsync_task
from ...queue.llm_task import llm_task
from ...queue.qc_task import qc_task
from ...queue.shot_task import shot_task
from ...queue.tts_task import tts_task
from ...queue.video_task import video_task
from ...workers.celery_app import celery_app
from ..schemas import TaskEnqueueRequest, TaskEnqueueResponse, TaskStatusResponse
from ..security import require_api_key
from ..task_payloads import PAYLOAD_MODELS, validate_payload

router = APIRouter(prefix="/tasks", tags=["tasks"], dependencies=[Depends(require_api_key)])

# 队列名 -> Celery 任务对象；键必须和 task_payloads.PAYLOAD_MODELS 一致
_TASKS = {
    "llm": llm_task,
    "assistant": assistant_task,
    "image": image_task,
    "shot": shot_task,
    "video": video_task,
    "tts": tts_task,
    "lipsync": lipsync_task,
    "qc": qc_task,
    "analytics": analytics_task,
}


@router.post("", response_model=TaskEnqueueResponse, status_code=202)
def enqueue_task(payload: TaskEnqueueRequest) -> TaskEnqueueResponse:
    """校验 payload 后入队，返回 202 + task_id（202 表示"已接受、异步处理"，不是"已完成"）。

    未知队列名 -> 400；payload 字段不合法 -> 422，并把 pydantic 的错误明细原样带回给调用方定位。
    """
    task = _TASKS.get(payload.queue)
    if task is None:
        raise HTTPException(status_code=400, detail=f"Unknown queue '{payload.queue}', expected one of {sorted(_TASKS)}")
    try:
        clean = validate_payload(payload.queue, payload.payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"queue": payload.queue, "errors": exc.errors(include_url=False)})
    async_result = task.delay(clean)
    return TaskEnqueueResponse(task_id=async_result.id, queue=payload.queue, status=async_result.status)


@router.get("/schemas")
def task_schemas() -> Dict[str, Any]:
    """每个队列接受的 payload JSON Schema，前端/调用方照着拼。"""
    # 注意路由顺序：必须定义在 /{task_id} 之前，否则 "schemas" 会被当成 task_id 匹配
    return {queue: model.model_json_schema() for queue, model in PAYLOAD_MODELS.items()}


@router.get("/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """查任务状态：从 result backend 读 AsyncResult。

    失败时把异常转成字符串放 error；成功时结果不是 dict 的（比如返回了字符串）包成 {"value": ...}，保证响应结构稳定。
    """
    async_result = celery_app.AsyncResult(task_id)
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    if async_result.failed():
        error = str(async_result.result)
    elif async_result.successful():
        raw = async_result.result
        result = raw if isinstance(raw, dict) else {"value": raw}
    return TaskStatusResponse(task_id=task_id, status=async_result.status, result=result, error=error)
