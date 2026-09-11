from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from ..database.session import init_db
from .routers import assets, characters, episodes, projects, shots, tasks


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Convenience for local dev/demo only; production schema changes go through
    # Alembic migrations (13_INFRA/database/alembic), not create_all().
    init_db()
    yield


app = FastAPI(title="AI Short Drama API", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(projects.router)
app.include_router(episodes.router)
app.include_router(characters.router)
app.include_router(shots.router)
app.include_router(assets.router)
app.include_router(tasks.router)
