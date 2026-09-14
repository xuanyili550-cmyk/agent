"""API key 鉴权。

请求头 ``X-API-Key`` 必须在 settings.api_keys 里（常量时间比较，防时序攻击）。
API_KEYS 没配置时受保护接口返回 503 而不是放行——"忘了配鉴权"在生产上应该是显性故障，
不能静默变成裸奔。/health 和 /metrics 不受保护（探针和 Prometheus 抓取用）。
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from ..config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _matches(candidate: str, keys: list[str]) -> bool:
    return any(hmac.compare_digest(candidate.encode(), k.encode()) for k in keys)


def require_api_key(request: Request, api_key: str | None = Depends(api_key_header)) -> str:
    settings = get_settings()
    if not settings.api_keys:
        raise HTTPException(status_code=503, detail="API_KEYS 未配置，受保护接口不可用")
    if not api_key or not _matches(api_key, settings.api_keys):
        raise HTTPException(status_code=401, detail="缺少或无效的 X-API-Key")
    request.state.api_key = api_key
    return api_key
