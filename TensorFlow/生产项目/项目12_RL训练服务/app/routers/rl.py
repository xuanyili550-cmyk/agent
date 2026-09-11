from fastapi import APIRouter
from ..schemas.rl import EvalRequest, TrainRequest
from ..services import rl
router = APIRouter(prefix="/api", tags=["rl"])
@router.post("/train")
def train(req: TrainRequest): return rl.train(req.env, req.steps)
@router.post("/eval")
def evaluate(req: EvalRequest): return rl.evaluate(req.env)
