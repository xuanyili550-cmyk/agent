from __future__ import annotations

import os

from .base import StorageAdapter
from .local import LocalStorageAdapter
from .s3 import S3StorageAdapter


def get_storage() -> StorageAdapter:
    """Picks a storage backend from env vars.

    STORAGE_BACKEND=local (default): LocalStorageAdapter rooted at STORAGE_LOCAL_ROOT
    (default "./storage_data").
    STORAGE_BACKEND=s3: S3StorageAdapter reading STORAGE_BUCKET, STORAGE_ENDPOINT_URL,
    STORAGE_ACCESS_KEY, STORAGE_SECRET_KEY, STORAGE_REGION.
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
