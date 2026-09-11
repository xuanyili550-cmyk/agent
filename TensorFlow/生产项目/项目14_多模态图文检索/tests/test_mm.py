from fastapi.testclient import TestClient
from app.main import app
from app.services import multimodal
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_index_search():
    multimodal.clear()
    client.post("/api/index", json={"items": ["一只橘猫的照片", "雪山风景", "城市夜景"]})
    hits = client.post("/api/search", json={"query": "猫", "k": 2}).json()["hits"]
    assert hits and "猫" in hits[0]["item"]         # 图文对齐:查"猫"命中猫的描述
def test_validation(): assert client.post("/api/search", json={"query": ""}).status_code == 422
