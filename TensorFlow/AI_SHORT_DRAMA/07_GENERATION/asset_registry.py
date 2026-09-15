"""本地文件版素材台账（asset ledger）。

字段设计对齐将来 PostgreSQL 里的 `assets` 表——现阶段只是往一个 JSONL 文件追加记录。

流水线位置：07_GENERATION 每个生成器（图像 / 视频 / TTS / 口型 / 声音克隆）产出一个文件后，
都调用 ``write_asset_record`` 登记一条 AssetRecord，后续 08_QC、09_POST 和 13_INFRA 的数据库
导入都以这条记录为准。
为什么用 JSONL 追加写：离线 demo 不需要数据库；追加写天然幂等、可并发、可随时用 ``jq`` 查看，
且每行就是一个 AssetRecord，迁移到数据库时逐行读取即可。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class AssetRecord(BaseModel):
    """一条素材记录：文件在哪、属于哪个角色 / 剧集 / 镜头、用什么模型和 prompt / seed 生成的。

    保留 ``seed`` 与 ``prompt`` 是为了可复现：QC 不过需要重跑时能拿到完全相同的输入。
    ``license`` 记录素材许可（模型输出许可各不相同，商用前必须能追溯）。
    """

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
    """生成形如 ``img_3f9a1c2b7d4e`` 的素材 ID；前缀标明类型，12 位十六进制随机数足以避免碰撞。"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    """当前 UTC 时间的 ISO 8601 字符串；统一用 UTC 避免多机器时区不一致。"""
    return datetime.now(UTC).isoformat()


def write_asset_record(record: AssetRecord, jsonl_path: str | Path) -> None:
    """把一条记录以 JSON 单行追加写入台账文件；父目录不存在时自动创建。"""
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


# 同一个模块可能以两种名字被 import（07 内部 `from errors import ...` 走 sys.path，13_INFRA 走
# importlib "07_GENERATION.errors"）。不做别名就会出现两份类对象，except 捕获不到对方抛的异常。
import sys as _sys

# setdefault：只在该名字还没被注册时才写入，避免覆盖已经以那个名字正常 import 的模块对象
for _name in ("asset_registry", "07_GENERATION.asset_registry"):
    _sys.modules.setdefault(_name, _sys.modules[__name__])
