from fastapi import APIRouter
from pydantic import BaseModel, Field
from ..services import multimodal
router = APIRouter(prefix="/api", tags=["multimodal"])
class IndexReq(BaseModel):
    items: list[str] = Field(..., min_length=1, description="图像描述/文本(统一到同一空间)")
class SearchReq(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = 3
@router.post("/index")
def index(req: IndexReq): return multimodal.index(req.items)
@router.post("/search")
def search(req: SearchReq): return {"hits": multimodal.search(req.query, req.k)}
