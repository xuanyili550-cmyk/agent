from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .base import Base

# Production: set DATABASE_URL to a Postgres DSN, e.g.
#   postgresql+psycopg2://user:password@postgres:5432/ai_short_drama
# Local dev fallback (no Docker/Postgres running): a SQLite file next to this module.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "sqlite:///./13_infra_dev.db"
)

_is_sqlite = DATABASE_URL.startswith("sqlite")
_is_sqlite_memory = _is_sqlite and ":memory:" in DATABASE_URL
_connect_args = {"check_same_thread": False} if _is_sqlite else {}
# A plain sqlite ":memory:" DB is per-connection; FastAPI's sync endpoints run in a
# threadpool, so without a single shared connection each request could see a fresh,
# empty database. StaticPool pins all sessions to one connection to avoid that.
_engine_kwargs = {"poolclass": StaticPool} if _is_sqlite_memory else {}

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
