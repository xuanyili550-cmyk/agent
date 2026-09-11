from fastapi import APIRouter
from ..core.config import get_settings
router = APIRouter(tags=["health"])
@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "backend": s.backend, "default_model": s.default_model}
@router.get("/api/models")
def models():
    return {"data": [{"id": get_settings().default_model}]}
