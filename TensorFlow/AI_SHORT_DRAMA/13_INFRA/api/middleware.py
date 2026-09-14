"""请求级中间件：request_id + 结构化访问日志 + Prometheus + 限流。

限流用固定窗口计数（每分钟 N 次，按 API key 或客户端 IP）：有 Redis（LLM_CONTEXT_REDIS_URL）
时计数放 Redis，多副本 API 共享配额；没有就进程内字典（单副本开发够用）。
返回 429 时带 Retry-After。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import get_settings
from ..observability import HTTP_REQUESTS, get_logger
from ..observability.logging import request_id_var
from ..observability.metrics import HTTP_LATENCY

log = get_logger("api")
EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}


class _MemoryCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, int], int] = defaultdict(int)

    def incr(self, key: str, window: int) -> int:
        with self._lock:
            # 顺手清掉过期窗口，避免字典无限增长
            for k in [k for k in self._buckets if k[1] < window - 1]:
                del self._buckets[k]
            self._buckets[(key, window)] += 1
            return self._buckets[(key, window)]


class _RedisCounter:
    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url)

    def incr(self, key: str, window: int) -> int:
        redis_key = f"ratelimit:{key}:{window}"
        pipe = self._client.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, 120)
        count, _ = pipe.execute()
        return int(count)


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)
        settings = get_settings()
        self.limit = settings.rate_limit_per_minute
        self._counter = _RedisCounter(settings.llm_context_redis_url) if settings.llm_context_redis_url else _MemoryCounter()

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        path = request.url.path
        route = request.scope.get("route")
        path_label = getattr(route, "path", None) or path
        try:
            if self.limit > 0 and path not in EXEMPT_PATHS:
                identity = request.headers.get("X-API-Key") or (request.client.host if request.client else "anonymous")
                window = int(time.time() // 60)
                try:
                    count = self._counter.incr(identity, window)
                except Exception as exc:  # Redis 挂了不应该把 API 一起带挂：退化为不限流
                    log.warning("限流计数器不可用", extra={"error": str(exc)[:100]})
                    count = 0
                if count > self.limit:
                    HTTP_REQUESTS.labels(request.method, path_label, "429").inc()
                    return Response(
                        content='{"detail":"请求过于频繁"}',
                        status_code=429,
                        media_type="application/json",
                        headers={"Retry-After": str(60 - int(time.time() % 60)), "X-Request-ID": request_id},
                    )
            response = await call_next(request)
        except Exception:
            HTTP_REQUESTS.labels(request.method, path_label, "500").inc()
            log.exception("请求处理异常", extra={"method": request.method, "path": path})
            raise
        finally:
            request_id_var.reset(token)
        elapsed = time.perf_counter() - started
        HTTP_REQUESTS.labels(request.method, path_label, str(response.status_code)).inc()
        HTTP_LATENCY.labels(request.method, path_label).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        if path not in EXEMPT_PATHS:
            log.info("access", extra={"method": request.method, "path": path, "status": response.status_code, "ms": round(elapsed * 1000, 1)})
        return response
