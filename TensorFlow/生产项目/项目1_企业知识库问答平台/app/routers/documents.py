from fastapi import APIRouter
from ..schemas.kb import IngestRequest, IngestResponse
from ..services import rag
from ..services.index import get_index
router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/documents", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """入库一份文档(解析→切块→嵌入→建索引)。"""
    return IngestResponse(**rag.ingest(req.name, req.text))


@router.delete("/documents")
def clear():
    get_index().clear()
    return {"status": "cleared"}
