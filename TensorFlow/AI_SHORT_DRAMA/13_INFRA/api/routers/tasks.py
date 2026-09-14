from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from ...queue.analytics_task import analytics_task
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

_TASKS = {
    "llm": llm_task,
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
    return {queue: model.model_json_schema() for queue, model in PAYLOAD_MODELS.items()}


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
