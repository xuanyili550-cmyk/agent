"""全项目测试的公共环境：在任何 13_INFRA 模块被 import 之前把环境变量定死。

- SQLite 内存库 + Celery eager：不需要 Postgres/Redis；
- LLM_PROVIDER=mock、IMAGE_BACKEND=dummy、QC_BACKEND=none：不需要 API key、GPU、模型权重；
- 产物写到临时目录，不污染仓库。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="ai_short_drama_tests_")
_ENV = {
    "APP_ENV": "test",
    "DATABASE_URL": "sqlite:///:memory:",
    "CELERY_TASK_ALWAYS_EAGER": "true",
    "CELERY_BROKER_URL": "memory://",
    "CELERY_RESULT_BACKEND": "cache+memory://",
    "STORAGE_BACKEND": "local",
    "STORAGE_LOCAL_ROOT": str(Path(_TMP) / "storage"),
    "ARTIFACTS_ROOT": str(Path(_TMP) / "artifacts"),
    "LLM_CONTEXT_DIR": str(Path(_TMP) / "contexts"),
    "API_KEYS": "test-key",
    "LLM_PROVIDER": "mock",
    "IMAGE_BACKEND": "dummy",
    "QC_BACKEND": "none",
    "METRICS_ENABLED": "false",
    "RATE_LIMIT_PER_MINUTE": "0",
    "PUBLISH_DRY_RUN": "true",
    "EAGER_BACKOFF_SCALE": "0",
    "RETRY_RESOLUTION_LADDER": "256x448,192x336,128x224",
    "IMAGE_WIDTH": "256",
    "IMAGE_HEIGHT": "448",
}
for key, value in _ENV.items():
    os.environ.setdefault(key, value)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def api_headers() -> dict[str, str]:
    """给测试用的 API Key 请求头：取环境变量 API_KEYS 里的第一个 key，供 TestClient 统一带上。"""
    return {"X-API-Key": os.environ["API_KEYS"].split(",")[0]}


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    """本次测试会话的临时根目录（存储/产物/上下文都在它下面），测试里要核对落盘路径时可以直接用。"""
    return Path(_TMP)


@pytest.fixture(scope="session", autouse=True)
def _init_database():
    """内存 SQLite 每个进程一份：先建表，任何测试单独跑也不会遇到 no such table。"""
    import importlib

    importlib.import_module("13_INFRA.database.session").init_db()
    yield
