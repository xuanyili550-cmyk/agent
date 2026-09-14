from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_settings
from ..database.session import init_db
from ..observability import configure_logging, get_logger, metrics_response
from .middleware import RequestContextMiddleware
from .routers import assets, characters, episodes, pipelines, projects, shots, tasks

log = get_logger("api.main")


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    if not settings.is_prod:
        # 只给本地开发/demo 用；生产的表结构变更走 Alembic（13_INFRA/database/alembic），不用 create_all()
        init_db()
    if not settings.api_keys:
        log.warning("API_KEYS 未配置：除 /health /metrics 外所有接口都会返回 503")
    yield


settings = get_settings()
app = FastAPI(title="AI Short Drama API", version="0.2.0", lifespan=_lifespan)
app.add_middleware(RequestContextMiddleware)
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": settings.app_env}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


app.include_router(projects.router)
app.include_router(episodes.router)
app.include_router(characters.router)
app.include_router(shots.router)
app.include_router(assets.router)
app.include_router(tasks.router)
app.include_router(pipelines.router)
