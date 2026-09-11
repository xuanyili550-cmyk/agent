from fastapi import APIRouter
from ..core.config import get_settings
from ..core.exceptions import ValidationError
from ..schemas.mod import BatchRequest, ModerateRequest, ModerateResponse
from ..services import jobs, moderate as mod
router = APIRouter(prefix="/api", tags=["moderate"])


@router.post("/moderate", response_model=ModerateResponse)
def moderate_one(req: ModerateRequest):
    return mod.moderate(req.text)


@router.post("/moderate/batch")
def moderate_batch(req: BatchRequest):
    """大批量 → 异步 job,返回 job_id。"""
    if len(req.texts) > get_settings().max_batch:
        raise ValidationError("批量过大")
    jid = jobs.submit(lambda ts: [mod.moderate(t) for t in ts], req.texts)
    return {"job_id": jid}


@router.get("/jobs/{jid}")
def job(jid: str):
    from ..core.exceptions import AppError
    j = jobs.get(jid)
    if not j:
        raise AppError("任务不存在", "not_found", 404)
    return j
