"""数据库连接与会话：按 settings.database_url 建一个全局 engine，并给 API / 队列任务各提供一种拿 Session 的方式。

- ``get_db()``：FastAPI ``Depends`` 用的生成器，请求结束自动 close（是否 commit 由接口自己决定）；
- ``session_scope()``：队列任务用的上下文管理器，成功 commit、异常 rollback；
- ``init_db()``：开发/测试直接建表，生产用 alembic 迁移。
SQLite 内存库的 StaticPool 特殊处理见下方注释。
"""

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
# SQLite 连接默认只允许创建它的线程使用；FastAPI 线程池 / Celery 都会跨线程用同一连接，必须关掉这个检查
_connect_args = {"check_same_thread": False} if _is_sqlite else {}
# 纯 sqlite ":memory:" 是"每个连接一个库"；FastAPI 的同步接口跑在线程池里，不固定成
# 单一连接的话每个请求可能看到一个全新的空库。StaticPool 把所有 session 钉在同一条连接上。
# Postgres 走 pool_pre_ping：从连接池取连接前先 ping 一下，避免拿到被服务端断掉的死连接。
_engine_kwargs = {"poolclass": StaticPool} if _is_sqlite_memory else {"pool_pre_ping": True}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True, **_engine_kwargs)
# autoflush=False：避免查询时隐式 flush 触发意外的约束错误；提交时机统一由 session_scope / 接口控制
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """按 models 里的定义直接建表（已存在的表跳过）。只给开发/测试用；生产环境的表结构变更走 alembic。"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每个请求一个 Session，请求结束时关闭。不在这里 commit，由接口按需显式提交。"""
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
