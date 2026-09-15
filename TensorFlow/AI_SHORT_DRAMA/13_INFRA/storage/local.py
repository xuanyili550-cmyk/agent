"""本地磁盘存储适配器：开发/测试用的默认后端，零外部依赖。

把 key 映射成 ``root_dir/key`` 的文件路径；并做路径越界检查，防止 ``../`` 形式的 key 写到存储根目录之外。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO, Union

from .base import StorageAdapter

PathLike = Union[str, Path]


class LocalStorageAdapter(StorageAdapter):
    """把文件存在本地磁盘的 ``root_dir`` 下。本地开发的默认后端（不需要 S3/R2 凭据）。"""

    def __init__(self, root_dir: PathLike):
        """记录根目录并确保它存在（parents=True：一次性建好多级目录）。"""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """把 key 换算成根目录下的绝对路径，并拒绝逃逸出根目录的 key。

        ``resolve()`` 会展开 ``..`` 和符号链接，所以 "a/../../etc/passwd" 这类 key 解析后不在 root 之下，直接抛 ValueError。
        """
        path = (self.root_dir / key).resolve()
        if self.root_dir.resolve() not in path.parents and path != self.root_dir.resolve():
            raise ValueError(f"key '{key}' escapes storage root")
        return path

    def upload_file(self, local_path: PathLike, key: str) -> str:
        """复制本地文件到 ``root_dir/key``，返回目标绝对路径。"""
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest)
        return str(dest)

    def upload_fileobj(self, fileobj: BinaryIO, key: str) -> str:
        """把文件对象内容流式写到 ``root_dir/key``（copyfileobj 分块拷贝，大文件不会一次性读进内存）。"""
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            shutil.copyfileobj(fileobj, f)
        return str(dest)

    def download_file(self, key: str, local_path: PathLike) -> Path:
        """把存储里的文件复制到 ``local_path``；对象不存在时抛 FileNotFoundError，和 S3 后端行为对齐。"""
        src = self._resolve(key)
        if not src.exists():
            raise FileNotFoundError(f"Object not found: {key}")
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local_path)
        return local_path

    def exists(self, key: str) -> bool:
        """文件是否存在。"""
        return self._resolve(key).exists()

    def delete(self, key: str) -> None:
        """删除文件；``missing_ok=True`` 让重复删除不报错（幂等）。"""
        path = self._resolve(key)
        path.unlink(missing_ok=True)

    def url(self, key: str) -> str:
        """本地后端的"URL"就是绝对路径。"""
        return str(self._resolve(key))
