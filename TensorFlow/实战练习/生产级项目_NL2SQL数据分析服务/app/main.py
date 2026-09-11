# 【构建 11/16 · 组装】依赖 core+db+routers（全部上层，把项目串起来）
"""FastAPI 应用工厂：中间件(CORS/请求日志/request-id) + 全局异常 + 路由 + 启动初始化。"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, new_request_id, setup_logging
from app.db.database import init_db
from app.routers import health, query

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    init_db()                                # 生产用 Alembic 迁移；这里演示建表+灌种子
    logger.info("应用启动完成: env=%s llm=%s", s.env, s.llm_backend)
    yield
    logger.info("应用关闭")


def create_app() -> FastAPI:
    s = get_settings()
    setup_logging(s.log_level)
    app = FastAPI(title=s.app_name, version="1.0.0", lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware, allow_origins=[o.strip() for o in s.cors_origins.split(",")],
        allow_methods=["*"], allow_headers=["*"],
    )

    # 请求日志 + request-id(每请求一个 id，贯穿日志便于追踪)
    @app.middleware("http")
    async def _request_context(request: Request, call_next):
        rid = new_request_id()
        t0 = time.time()
        response = await call_next(request)
        logger.info("%s %s -> %d (%.0fms)", request.method, request.url.path,
                    response.status_code, (time.time() - t0) * 1000)
        response.headers["X-Request-ID"] = rid
        return response

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(query.router)
    return app


app = create_app()      # uvicorn app.main:app
