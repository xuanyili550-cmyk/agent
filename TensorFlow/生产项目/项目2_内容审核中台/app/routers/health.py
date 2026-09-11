from fastapi import APIRouter
from ..core.config import get_settings
router = APIRouter(tags=["health"])
@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "risk_backend": s.risk_backend,
            "thresholds": {"block": s.block_threshold, "review": s.review_threshold}}
