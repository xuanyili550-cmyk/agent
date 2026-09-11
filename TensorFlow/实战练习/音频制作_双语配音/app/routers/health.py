"""健康检查路由：K8s/负载均衡探针 + 人工排障都看它。
返回各依赖(say/ffmpeg/字体/声音)是否就绪；全就绪=ok，缺任一=degraded(服务仍部分可用)。"""
from fastapi import APIRouter

from ..services.system import healthcheck

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    st = healthcheck()
    return {"status": "ok" if st["ok"] else "degraded", "deps": st}
