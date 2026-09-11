"""健康检查：返回模型配置 + 向量库规模(探针/排障用)。不触发模型加载(保持懒加载)。"""
from fastapi import APIRouter

from ..core.config import get_settings
from ..services.index import get_index

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok",
            "model": s.model_name, "pooling": s.pooling, "normalize": s.normalize,
            "rerank_enabled": s.rerank_enabled, "index": get_index().stats()}
