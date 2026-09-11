# 【构建 12/16 · 测试】依赖 main(11)+database(5)+models(4)
"""pytest 配置：测试用 stub LLM(确定性、不下模型) + 独立 SQLite 测试库。
必须在导入 app 之前设好环境变量(配置 lru_cache + 引擎在模块级创建)。"""
import os

os.environ["APP_LLM_BACKEND"] = "stub"
os.environ["APP_DATABASE_URL"] = "sqlite:///./test_nl2sql.db"
os.environ["APP_DB_SEED"] = "true"

import pytest                                       # noqa: E402
from fastapi.testclient import TestClient          # noqa: E402

from app.db.database import SessionLocal, engine, init_db   # noqa: E402
from app.db.models import Base                     # noqa: E402
from app.main import create_app                    # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    init_db()
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
