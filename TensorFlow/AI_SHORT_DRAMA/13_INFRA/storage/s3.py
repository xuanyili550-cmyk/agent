"""S3 兼容对象存储适配器（AWS S3 / Cloudflare R2 / MinIO），生产环境后端。

通过 boto3 访问；``endpoint_url`` 指向不同供应商即可切换，业务代码不感知差异。
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Optional, Union

from .base import StorageAdapter

PathLike = Union[str, Path]


class S3StorageAdapter(StorageAdapter):
    """基于 boto3 的 S3/R2 兼容对象存储后端。把 ``endpoint_url`` 指向对应供应商即可用于
    AWS S3、Cloudflare R2、MinIO 等。

    凭据/配置来自构造参数；参数省略时 boto3 会按它自己的默认链（环境变量、~/.aws、实例角色）查找。
    工厂 ``get_storage()`` 负责从 STORAGE_ENDPOINT_URL、STORAGE_ACCESS_KEY、STORAGE_SECRET_KEY、
    STORAGE_BUCKET、STORAGE_REGION 这些环境变量读值传进来。
    """

    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """建 boto3 S3 client。

        boto3 延迟导入并给出友好的安装提示：本地开发只用 LocalStorageAdapter 时不必装 boto3，
        导入本模块（storage/__init__ 会导入）也不会失败。
        """
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3StorageAdapter. Install it with `pip install -r 13_INFRA/requirements-infra.txt`.") from exc

        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def upload_file(self, local_path: PathLike, key: str) -> str:
        """上传本地文件到 bucket 的 ``key``（boto3 内部自动分片处理大文件），返回 s3:// URI。"""
        self._client.upload_file(str(local_path), self.bucket, key)
        return self.url(key)

    def upload_fileobj(self, fileobj: BinaryIO, key: str) -> str:
        """从文件对象流式上传，返回 s3:// URI。"""
        self._client.upload_fileobj(fileobj, self.bucket, key)
        return self.url(key)

    def download_file(self, key: str, local_path: PathLike) -> Path:
        """下载对象到本地 ``local_path``（先建好父目录），返回本地 Path。"""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(local_path))
        return local_path

    def exists(self, key: str) -> bool:
        """用 HEAD 请求判断对象是否存在。

        任何异常都当作"不存在"：boto3 对 404 抛 ClientError，对网络/权限问题抛别的异常，
        对调用方来说都意味着"现在拿不到这个对象"。
        """
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        """删除对象；S3 的 delete_object 对不存在的 key 也返回成功，天然幂等。"""
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def url(self, key: str) -> str:
        """返回 ``s3://bucket/key`` 形式的定位符（入库用；对外访问需另行生成预签名 URL）。"""
        return f"s3://{self.bucket}/{key}"
