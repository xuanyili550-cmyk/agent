"""请求级中间件：request_id + 结构化访问日志 + Prometheus + 限流。

限流用固定窗口计数（每分钟 N 次，按 API key 或客户端 IP）：有 Redis（LLM_CONTEXT_REDIS_URL）
时计数放 Redis，多副本 API 共享配额；没有就进程内字典（单副本开发够用）。
返回 429 时带 Retry-After。

这一层放在所有路由之外，所以鉴权失败、404、500 也都会被计数和记日志；
request_id 通过 contextvar 传播，请求内所有日志（含 worker 侧透传的 task 日志）都能串起来。
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
# 探针、指标抓取、文档页不限流也不记访问日志：它们由监控系统高频调用，记了只会刷屏
EXEMPT_PATHS = {"/", "/ui", "/health", "/metrics", "/metrics/summary", "/docs", "/openapi.json", "/redoc"}


class _MemoryCounter:
    """进程内固定窗口计数器（无 Redis 时的退化实现）。多副本部署下各副本配额独立，仅适合单副本/本地开发。"""

    def __init__(self) -> None:
        """初始化锁和 ``(identity, window) -> 次数`` 的桶字典。"""
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, int], int] = defaultdict(int)

    def incr(self, key: str, window: int) -> int:
        """给 ``key`` 在 ``window``（分钟号）的计数 +1 并返回新值。加锁是因为 uvicorn 线程池里会并发调用。"""
        with self._lock:
            # 顺手清掉过期窗口，避免字典无限增长
            for k in [k for k in self._buckets if k[1] < window - 1]:
                del self._buckets[k]
            self._buckets[(key, window)] += 1
            return self._buckets[(key, window)]


class _RedisCounter:
    """基于 Redis INCR 的固定窗口计数器：多副本 API 共用一份配额。"""

    def __init__(self, url: str) -> None:
        """按 URL 建 Redis 客户端。redis 包延迟导入：没配 Redis 的部署不需要装它。"""
        import redis

        self._client = redis.Redis.from_url(url)

    def incr(self, key: str, window: int) -> int:
        """INCR + EXPIRE 打包在一个 pipeline 里往返一次；120 秒过期足够覆盖当前和上一个分钟窗口，之后自动清理。"""
        redis_key = f"ratelimit:{key}:{window}"
        pipe = self._client.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, 120)
        count, _ = pipe.execute()
        return int(count)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """每个请求：生成/透传 X-Request-ID、限流、记 Prometheus 指标和访问日志。"""

    def __init__(self, app) -> None:
        """读配置决定限流阈值，并按是否配了 Redis 选择计数器实现。"""
        super().__init__(app)
        settings = get_settings()
        self.limit = settings.rate_limit_per_minute
        self._counter = _RedisCounter(settings.llm_context_redis_url) if settings.llm_context_redis_url else _MemoryCounter()

    async def dispatch(self, request: Request, call_next) -> Response:
        """请求处理主流程：设 request_id 上下文 -> 限流判断 -> 调下游 -> 记指标/日志 -> 回写 X-Request-ID 头。

        指标的 path 标签优先用路由模板（``/projects/{project_id}``）而不是真实路径，否则每个 id 都会变成一个新的时间序列，把 Prometheus 撑爆。
        """
        # 调用方带了 X-Request-ID 就沿用（便于跨服务追踪），否则生成一个
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        path = request.url.path
        route = request.scope.get("route")
        path_label = getattr(route, "path", None) or path
        try:
            if self.limit > 0 and path not in EXEMPT_PATHS:
                # 有 API key 按 key 限流（同一调用方多台机器共享配额），否则退回按客户端 IP
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
                        # Retry-After = 距离下一个分钟窗口还有几秒
                        headers={"Retry-After": str(60 - int(time.time() % 60)), "X-Request-ID": request_id},
                    )
            response = await call_next(request)
        except Exception:
            # 下游抛出的未捕获异常：先计一次 500 并带栈记日志，再原样抛给 FastAPI 的异常处理
            HTTP_REQUESTS.labels(request.method, path_label, "500").inc()
            log.exception("请求处理异常", extra={"method": request.method, "path": path})
            raise
        finally:
            # 无论成功失败都要还原 contextvar，否则协程复用时会串到别的请求
            request_id_var.reset(token)
        elapsed = time.perf_counter() - started
        HTTP_REQUESTS.labels(request.method, path_label, str(response.status_code)).inc()
        HTTP_LATENCY.labels(request.method, path_label).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        if path not in EXEMPT_PATHS:
            log.info("access", extra={"method": request.method, "path": path, "status": response.status_code, "ms": round(elapsed * 1000, 1)})
        return response
