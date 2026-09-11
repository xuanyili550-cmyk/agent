"""检索路由：POST /api/search —— 查询 → 向量粗召回 →(可选)cross-encoder 重排 → 返回原文+分数。"""
from fastapi import APIRouter

from ..schemas.search import SearchRequest, SearchResponse
from ..services import search as search_svc

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
def api_search(req: SearchRequest):
    hits = search_svc.search(req.query, top_k=req.top_k, use_rerank=req.rerank)
    return SearchResponse(hits=hits)
