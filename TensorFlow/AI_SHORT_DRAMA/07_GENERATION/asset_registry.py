"""
Local file-backed asset ledger. Field design mirrors what would eventually be
a PostgreSQL `assets` table -- this just appends to a JSONL file for now.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
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
    return datetime.now(UTC).isoformat()


def write_asset_record(record: AssetRecord, jsonl_path: str | Path) -> None:
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


# 同一个模块可能以两种名字被 import（07 内部 `from errors import ...` 走 sys.path，13_INFRA 走
# importlib "07_GENERATION.errors"）。不做别名就会出现两份类对象，except 捕获不到对方抛的异常。
import sys as _sys

for _name in ("asset_registry", "07_GENERATION.asset_registry"):
    _sys.modules.setdefault(_name, _sys.modules[__name__])
