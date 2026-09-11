from fastapi import APIRouter
from ..schemas.d3d import Gen3DRequest
from ..services import gen3d
router = APIRouter(prefix="/api", tags=["3d"])
@router.post("/generate3d")
def generate3d(req: Gen3DRequest): return gen3d.generate3d(req.prompt)
