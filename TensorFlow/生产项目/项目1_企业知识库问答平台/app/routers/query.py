from fastapi import APIRouter
from ..schemas.kb import QueryRequest, QueryResponse
from ..services import agent, cache
router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """问答:Agent 决定查库 or 直答;相同问题命中缓存。"""
    k = cache.key("q", req.query)
    hit = cache.get(k)
    return hit or cache.put(k, agent.handle(req.query))
