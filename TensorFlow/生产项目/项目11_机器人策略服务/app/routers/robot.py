from fastapi import APIRouter
from ..schemas.rbt import ActRequest
from ..services import robot
router = APIRouter(prefix="/api", tags=["robot"])
@router.post("/act")
def act(req: ActRequest): return robot.act(req.observation)
