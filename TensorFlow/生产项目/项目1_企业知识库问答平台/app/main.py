"""应用工厂:CORS + 全局异常 + 路由 + 静态前端 + 启动自检。模型懒加载。"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .core.config import get_settings
from .core.exceptions import register_exception_handlers
from .core.logging import get_logger, setup_logging
from .routers import documents, health, query

_STATIC = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings(); setup_logging(s.debug)
    get_logger("startup").info("启动: embed=%s llm=%s(懒加载)", s.embed_backend, s.llm_backend)
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title=s.app_name, version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])
    register_exception_handlers(app)
    for r in (health.router, documents.router, query.router):
        app.include_router(r)
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_STATIC, "index.html"))
    return app


app = create_app()
