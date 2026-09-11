from __future__ import annotations

import abc
from pathlib import Path
from typing import BinaryIO, Union

PathLike = Union[str, Path]


class StorageAdapter(abc.ABC):
    """Storage backend interface shared by the local-disk and S3/R2-compatible adapters.

    Keys are always POSIX-style relative paths (e.g. "episodes/EP001/Episode.mp4"),
    regardless of backend.
    """

    @abc.abstractmethod
    def upload_file(self, local_path: PathLike, key: str) -> str:
        """Uploads/copies a local file to the store under `key`; returns a locator
        (local absolute path or s3:// URI) that can be persisted in the DB."""

    @abc.abstractmethod
    def upload_fileobj(self, fileobj: BinaryIO, key: str) -> str:
        ...

    @abc.abstractmethod
    def download_file(self, key: str, local_path: PathLike) -> Path:
        ...

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abc.abstractmethod
    def url(self, key: str) -> str:
        """Returns a locator/URL for the object (local path or presigned/public S3 URL)."""
