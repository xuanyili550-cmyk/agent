"""13_INFRA.storage：对象存储抽象层。

导出统一接口 ``StorageAdapter``、两种实现 ``LocalStorageAdapter``（本地磁盘，开发用）和
``S3StorageAdapter``（S3 / R2 / MinIO，生产用），以及按环境变量选实现的 ``get_storage()``。
业务代码只依赖接口和工厂，不直接 import 具体实现，换后端不用改调用方。
"""

from .base import StorageAdapter
from .factory import get_storage
from .local import LocalStorageAdapter
from .s3 import S3StorageAdapter

__all__ = ["StorageAdapter", "LocalStorageAdapter", "S3StorageAdapter", "get_storage"]
