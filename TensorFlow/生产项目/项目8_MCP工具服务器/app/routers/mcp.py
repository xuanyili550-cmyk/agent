from fastapi import APIRouter
from ..schemas.mcp import CallRequest
from ..services import mcp
router = APIRouter(prefix="/api", tags=["mcp"])
@router.get("/tools/list")
def list_tools(): return {"tools": mcp.list_tools()}
@router.post("/tools/call")
def call_tool(req: CallRequest): return mcp.call_tool(req.name, req.args)
