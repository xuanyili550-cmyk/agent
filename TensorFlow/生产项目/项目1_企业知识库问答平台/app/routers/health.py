from fastapi import APIRouter
from ..core.config import get_settings
from ..services.index import get_index
router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "embed_backend": s.embed_backend, "llm_backend": s.llm_backend,
            "index": get_index().stats()}
