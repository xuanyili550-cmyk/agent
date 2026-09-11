"""文档库路由：入库(编码存向量+原文) / 清空 / 统计。"""
from fastapi import APIRouter

from ..schemas.search import IndexRequest, IndexStats
from ..services import search, store
from ..services.index import get_index

router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/documents", response_model=IndexStats)
def api_index(req: IndexRequest):
    """把文档编码入向量库(passage 向量) + 存原文。"""
    stats = search.index_documents([d.model_dump() for d in req.documents])
    return IndexStats(**stats)


@router.get("/documents/stats", response_model=IndexStats)
def api_stats():
    return IndexStats(**get_index().stats())


@router.delete("/documents")
def api_clear():
    get_index().clear()
    store.clear()
    return {"status": "cleared"}
