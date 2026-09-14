from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..config import get_settings
from .base import Base

# 生产：DATABASE_URL 指向 Postgres，例如
#   postgresql+psycopg2://user:password@postgres:5432/ai_short_drama
# 本地开发默认值（没起 Docker/Postgres）：模块旁边的一个 SQLite 文件。
DATABASE_URL = get_settings().database_url

_is_sqlite = DATABASE_URL.startswith("sqlite")
_is_sqlite_memory = _is_sqlite and ":memory:" in DATABASE_URL
_connect_args = {"check_same_thread": False} if _is_sqlite else {}
# 纯 sqlite ":memory:" 是"每个连接一个库"；FastAPI 的同步接口跑在线程池里，不固定成
# 单一连接的话每个请求可能看到一个全新的空库。StaticPool 把所有 session 钉在同一条连接上。
_engine_kwargs = {"poolclass": StaticPool} if _is_sqlite_memory else {"pool_pre_ping": True}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """队列任务里用：``with session_scope() as db:`` 出错自动回滚、结束自动关闭。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
