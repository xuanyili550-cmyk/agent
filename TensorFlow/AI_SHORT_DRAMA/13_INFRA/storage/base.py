"""存储后端的抽象接口。

渲染成片、manifest、生成素材等文件最终都要落到"某个存储"里；本地开发写磁盘，生产写 S3/R2。
把读写操作收敛成这一个接口，业务代码（render/manifest/publish 任务）只认 key，不关心后端是什么。
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import BinaryIO, Union

PathLike = Union[str, Path]


class StorageAdapter(abc.ABC):
    """本地磁盘与 S3/R2 兼容适配器共用的存储后端接口。

    无论后端是什么，key 一律是 POSIX 风格的相对路径（例如 "episodes/EP001/Episode.mp4"），
    这样同一份业务代码切换后端时 key 不需要改。
    """

    @abc.abstractmethod
    def upload_file(self, local_path: PathLike, key: str) -> str:
        """把本地文件上传/复制到存储的 ``key`` 位置；返回可以持久化进数据库的定位符
        （本地绝对路径或 s3:// URI）。"""

    @abc.abstractmethod
    def upload_fileobj(self, fileobj: BinaryIO, key: str) -> str:
        """从已打开的二进制文件对象上传到 ``key``；返回定位符。适合内存中生成的内容，不用先落临时文件。"""
        ...

    @abc.abstractmethod
    def download_file(self, key: str, local_path: PathLike) -> Path:
        """把 ``key`` 对应的对象下载到本地 ``local_path``；返回本地 Path。"""
        ...

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """``key`` 对应的对象是否存在。"""
        ...

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """删除 ``key`` 对应的对象；不存在时应静默成功（幂等）。"""
        ...

    @abc.abstractmethod
    def url(self, key: str) -> str:
        """返回对象的定位符/URL（本地路径，或预签名/公开的 S3 URL）。"""
