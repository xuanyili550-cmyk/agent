from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO, Union

from .base import StorageAdapter

PathLike = Union[str, Path]


class LocalStorageAdapter(StorageAdapter):
    """Stores files on local disk under `root_dir`. Default backend for local dev
    (no S3/R2 credentials required)."""

    def __init__(self, root_dir: PathLike):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = (self.root_dir / key).resolve()
        if self.root_dir.resolve() not in path.parents and path != self.root_dir.resolve():
            raise ValueError(f"key '{key}' escapes storage root")
        return path

    def upload_file(self, local_path: PathLike, key: str) -> str:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest)
        return str(dest)

    def upload_fileobj(self, fileobj: BinaryIO, key: str) -> str:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            shutil.copyfileobj(fileobj, f)
        return str(dest)

    def download_file(self, key: str, local_path: PathLike) -> Path:
        src = self._resolve(key)
        if not src.exists():
            raise FileNotFoundError(f"Object not found: {key}")
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local_path)
        return local_path

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        path.unlink(missing_ok=True)

    def url(self, key: str) -> str:
        return str(self._resolve(key))
