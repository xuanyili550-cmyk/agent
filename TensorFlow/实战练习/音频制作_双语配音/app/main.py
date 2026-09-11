"""
================================================================================
 应用工厂 · FastAPI 组装（中间件 + 全局异常 + 路由 + 静态前端 + 启动自检）
================================================================================
 【这个文件做什么】把各层拼成一个可运行的 Web 应用：跨域、统一错误、注册路由、托管前端页面，
   并在启动时做依赖自检。用"应用工厂 create_app()"而不是模块级散装写——测试可独立构造、配置可注入。

 【lifespan 启动钩子里做什么(为什么)】
   · 初始化日志(按 debug 决定级别)。
   · 依赖自检 healthcheck()：把 say/字体/ffmpeg 是否就绪打进日志；缺了不崩溃(仍能用 /analyze 这类
     不依赖音视频的接口)，但会告警、/health 显示 degraded —— "尽量可用"而非"一票否决"。
   · 清一次过期缓存和旧任务，避免上次残留堆积。

 【前端怎么托管】/static 挂静态文件；GET / 直接返回 index.html —— 前后端同源，前端 fetch('/api/..') 免跨域配置。
================================================================================
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .core.config import get_settings
from .core.exceptions import register_exception_handlers
from .core.logging import get_logger, setup_logging
from .routers import audio, health, video
from .services import cache, jobs
from .services.system import healthcheck

_STATIC = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启停钩子：yield 之前=启动初始化，之后=关闭清理(此处无需特别清理)。"""
    s = get_settings()
    setup_logging(s.debug)
    log = get_logger("startup")
    st = healthcheck()
    log.info("依赖自检: %s", st)
    if not st["ok"]:                       # 缺依赖不崩，只告警(降级可用)
        log.warning("部分依赖缺失，音频/视频功能可能不可用(见 /health)。")
    cache.cleanup()                        # 清上次残留的过期缓存/任务
    jobs.cleanup()
    yield


def create_app() -> FastAPI:
    """构造并返回 FastAPI 应用(测试里也用它拿到干净实例)。"""
    s = get_settings()
    app = FastAPI(title=s.app_name, version="1.0.0", lifespan=lifespan)
    # 跨域：前端和 API 若不同源(开发/嵌入)也能调；生产应把 cors_origins 收紧到白名单
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"],
                       allow_headers=["*"])
    register_exception_handlers(app)       # 统一错误响应(见 core/exceptions)
    app.include_router(health.router)
    app.include_router(audio.router)
    app.include_router(video.router)
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_STATIC, "index.html"))   # 首页=前端单页

    return app


app = create_app()
