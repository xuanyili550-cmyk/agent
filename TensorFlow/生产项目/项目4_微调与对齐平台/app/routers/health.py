from fastapi import APIRouter
from ..core.config import get_settings
router = APIRouter(tags=["health"])
@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "train_backend": s.train_backend, "base_model": s.base_model}
