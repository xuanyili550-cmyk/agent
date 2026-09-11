from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from ...queue.image_task import image_task
from ...queue.lipsync_task import lipsync_task
from ...queue.llm_task import llm_task
from ...queue.qc_task import qc_task
from ...queue.tts_task import tts_task
from ...queue.video_task import video_task
from ...workers.celery_app import celery_app
from ..schemas import TaskEnqueueRequest, TaskEnqueueResponse, TaskStatusResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])

_TASKS = {
    "llm": llm_task,
    "image": image_task,
    "video": video_task,
    "tts": tts_task,
    "lipsync": lipsync_task,
    "qc": qc_task,
}


@router.post("", response_model=TaskEnqueueResponse, status_code=202)
def enqueue_task(payload: TaskEnqueueRequest) -> TaskEnqueueResponse:
    task = _TASKS.get(payload.queue)
    if task is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown queue '{payload.queue}', expected one of {sorted(_TASKS)}",
        )
    async_result = task.delay(payload.payload)
    return TaskEnqueueResponse(task_id=async_result.id, queue=payload.queue, status=async_result.status)


@router.get("/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str) -> TaskStatusResponse:
    async_result = celery_app.AsyncResult(task_id)
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    if async_result.failed():
        error = str(async_result.result)
    elif async_result.successful():
        raw = async_result.result
        result = raw if isinstance(raw, dict) else {"value": raw}
    return TaskStatusResponse(task_id=task_id, status=async_result.status, result=result, error=error)
