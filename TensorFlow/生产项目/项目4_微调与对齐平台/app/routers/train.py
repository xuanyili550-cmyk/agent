from fastapi import APIRouter
from ..core.exceptions import NotFoundError
from ..schemas.ft import TrainRequest
from ..services import jobs, pipeline, registry
router = APIRouter(prefix="/api", tags=["train"])
@router.post("/train")
def train(req: TrainRequest):
    """启动对齐流水线(异步 job):数据→SFT→评估→注册。"""
    samples = [s.model_dump() for s in req.samples]
    jid = jobs.submit(lambda prog, xs: pipeline.run(xs, prog), samples)
    return {"job_id": jid}
@router.get("/jobs/{jid}")
def job(jid: str):
    j = jobs.get(jid)
    if not j: raise NotFoundError("任务不存在")
    return j
@router.get("/models")
def models():
    return {"models": registry.list_all()}
