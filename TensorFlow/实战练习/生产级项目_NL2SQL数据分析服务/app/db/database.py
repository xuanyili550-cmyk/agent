# 【构建 5/16 · 数据】依赖 config(1)+models(4)
"""数据库引擎/会话 + 依赖注入 + 初始化灌种子。生产改 DATABASE_URL 即切 MySQL(带连接池)。"""
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Base, SEED_ORDERS, SEED_PRODUCTS

logger = get_logger(__name__)
_settings = get_settings()

# SQLite 需要 check_same_thread=False；MySQL/PG 会用连接池(pool_size 等)
_connect_args = {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}
engine = create_engine(_settings.database_url, connect_args=_connect_args, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每请求一个会话，用完即关。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """建表 + (可选)灌示例数据。生产用 Alembic 迁移，不在启动时建表。"""
    Base.metadata.create_all(engine)
    if _settings.db_seed:
        with SessionLocal() as db:
            if db.execute(text("SELECT COUNT(*) FROM products")).scalar() == 0:
                db.add_all(SEED_PRODUCTS + SEED_ORDERS)
                db.commit()
                logger.info("已灌入示例数据")
