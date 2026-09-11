from fastapi import APIRouter
from pydantic import BaseModel, Field
from ..services import evaluator, promptlab
router = APIRouter(prefix="/api", tags=["eval"])
class Case(BaseModel):
    query: str = ""; answer: str = ""; reference: str = ""; context: str = ""
class EvalReq(BaseModel):
    cases: list[Case] = Field(..., min_length=1)
class ABReq(BaseModel):
    cases: list[Case] = Field(..., min_length=1)
    template_a: str = Field(..., min_length=1)
    template_b: str = Field(..., min_length=1)
@router.post("/evaluate")
def evaluate_cases(req: EvalReq): return evaluator.evaluate([c.model_dump() for c in req.cases])
@router.post("/ab_eval")
def ab_eval(req: ABReq):
    return promptlab.ab_eval([c.model_dump() for c in req.cases], req.template_a, req.template_b)
