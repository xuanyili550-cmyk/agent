from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Optional, Union

from .base import StorageAdapter

PathLike = Union[str, Path]


class S3StorageAdapter(StorageAdapter):
    """S3/R2-compatible object storage backend via boto3. Works against AWS S3,
    Cloudflare R2, MinIO, etc. by pointing `endpoint_url` at the provider.

    Reads credentials/config from constructor args or, if omitted, from env vars:
    STORAGE_ENDPOINT_URL, STORAGE_ACCESS_KEY, STORAGE_SECRET_KEY, STORAGE_BUCKET,
    STORAGE_REGION.
    """

    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
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
        self._client.upload_file(str(local_path), self.bucket, key)
        return self.url(key)

    def upload_fileobj(self, fileobj: BinaryIO, key: str) -> str:
        self._client.upload_fileobj(fileobj, self.bucket, key)
        return self.url(key)

    def download_file(self, key: str, local_path: PathLike) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(local_path))
        return local_path

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def url(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"
