from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import get_settings
from .core.exceptions import register_exception_handlers
from .core.logging import get_logger, setup_logging
from .routers import health, image
@asynccontextmanager
async def lifespan(app):
    s = get_settings(); setup_logging(s.debug)
    get_logger("startup").info("启动 %s backend=%s", s.app_name, s.backend); yield
def create_app():
    s = get_settings(); app = FastAPI(title=s.app_name, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])
    register_exception_handlers(app)
    app.include_router(health.router); app.include_router(image.router)
    return app
app = create_app()
