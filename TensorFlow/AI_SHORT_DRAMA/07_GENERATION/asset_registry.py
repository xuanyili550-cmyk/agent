"""
Local file-backed asset ledger. Field design mirrors what would eventually be
a PostgreSQL `assets` table -- this just appends to a JSONL file for now.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class AssetRecord(BaseModel):
    asset_id: str
    file_path: str
    character_id: Optional[str] = None
    episode_id: Optional[str] = None
    shot_id: Optional[str] = None
    model: str
    prompt: Optional[str] = None
    seed: Optional[int] = None
    version: str = "0.1.0"
    license: Optional[str] = None
    created_at: str


def new_asset_id(prefix: str = "asset") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_asset_record(record: AssetRecord, jsonl_path: str | Path) -> None:
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
