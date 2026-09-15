"""存储后端工厂：按环境变量 ``STORAGE_BACKEND`` 决定用本地磁盘还是 S3。

业务代码统一调 ``get_storage()`` 拿适配器，部署时只改环境变量就能切换后端，不动代码。
"""

from __future__ import annotations

import os

from .base import StorageAdapter
from .local import LocalStorageAdapter
from .s3 import S3StorageAdapter


def get_storage() -> StorageAdapter:
    """根据环境变量选择存储后端。

    STORAGE_BACKEND=local（默认）：LocalStorageAdapter，根目录取 STORAGE_LOCAL_ROOT（默认 "./storage_data"）。
    STORAGE_BACKEND=s3：S3StorageAdapter，读取 STORAGE_BUCKET、STORAGE_ENDPOINT_URL、
    STORAGE_ACCESS_KEY、STORAGE_SECRET_KEY、STORAGE_REGION。

    s3 模式下 STORAGE_BUCKET 缺失直接抛错而不是给默认值：bucket 配错会把成片写到不该去的地方，必须显式配置。
    """
    backend = os.environ.get("STORAGE_BACKEND", "local").lower()
    if backend == "local":
        root = os.environ.get("STORAGE_LOCAL_ROOT", "./storage_data")
        return LocalStorageAdapter(root)
    if backend == "s3":
        bucket = os.environ.get("STORAGE_BUCKET")
        if not bucket:
            raise RuntimeError("STORAGE_BUCKET env var is required when STORAGE_BACKEND=s3")
        return S3StorageAdapter(
            bucket=bucket,
            endpoint_url=os.environ.get("STORAGE_ENDPOINT_URL"),
            access_key=os.environ.get("STORAGE_ACCESS_KEY"),
            secret_key=os.environ.get("STORAGE_SECRET_KEY"),
            region=os.environ.get("STORAGE_REGION"),
        )
    raise ValueError(f"Unknown STORAGE_BACKEND '{backend}', expected 'local' or 's3'")
