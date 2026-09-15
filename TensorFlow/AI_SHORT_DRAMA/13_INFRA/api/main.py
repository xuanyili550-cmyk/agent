"""FastAPI 应用入口：装配中间件、路由、健康检查、指标接口和 Web 控制台。

启动即用：``uvicorn 13_INFRA.api.main:app``（见 Makefile ``api`` 目标），浏览器打开 http://localhost:8000/
就是控制台（``static/index.html``，零构建单文件页面；页面里的请求同样带 X-API-Key）。
无鉴权接口只有 ``/``、``/ui``、``/health``、``/metrics``、``/metrics/summary``，其余路由都挂了 API key 依赖。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..config import get_settings
from ..database.session import init_db
from ..observability import configure_logging, four_key_metrics, get_logger, metrics_response
from .middleware import RequestContextMiddleware
from .routers import assets, characters, episodes, pipelines, projects, shots, tasks

log = get_logger("api.main")
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """进程启动/关闭钩子：配日志、非生产环境自动建表、提醒缺 API key。"""
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


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
def console() -> FileResponse:
    """Web 控制台：单文件 HTML，直接从 static/ 读取；页面本身不含密钥，数据请求由浏览器带 X-API-Key 发起。"""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")


@app.get("/health")
def health() -> dict:
    """存活探针（Docker HEALTHCHECK / K8s liveness 用）。"""
    return {"status": "ok", "env": settings.app_env}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus 抓取端点。"""
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


@app.get("/metrics/summary")
def metrics_summary() -> dict:
    """四大监控指标（token 消耗 / 工具调用成功率 / 执行时间 / 错误率）的 JSON 汇总，本进程启动以来累计。"""
    return four_key_metrics()


app.include_router(projects.router)
app.include_router(episodes.router)
app.include_router(characters.router)
app.include_router(shots.router)
app.include_router(assets.router)
app.include_router(tasks.router)
app.include_router(pipelines.router)
