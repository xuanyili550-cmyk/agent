# 【构建 9/16 · 接口】依赖 config(1)
"""健康检查(K8s 存活/就绪探针)。"""
from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "app": s.app_name, "env": s.env, "llm_backend": s.llm_backend}
