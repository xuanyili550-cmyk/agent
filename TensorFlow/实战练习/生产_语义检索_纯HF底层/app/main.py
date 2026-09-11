"""FastAPI 应用工厂：CORS + 全局异常 + 路由。模型懒加载(启动不加载，首次请求才加载)。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.exceptions import register_exception_handlers
from .core.logging import get_logger, setup_logging
from .routers import documents, embed, health, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.debug)
    get_logger("startup").info("启动: model=%s pooling=%s (模型懒加载，首个请求才载入)",
                               s.model_name, s.pooling)
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title=s.app_name, version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(embed.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    return app


app = create_app()
