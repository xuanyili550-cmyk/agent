from fastapi import APIRouter
from ..schemas.agt import AgentRequest
from ..services import agent
router = APIRouter(prefix="/api", tags=["agent"])
@router.post("/agent")
def run(req: AgentRequest): return agent.run(req.query)
