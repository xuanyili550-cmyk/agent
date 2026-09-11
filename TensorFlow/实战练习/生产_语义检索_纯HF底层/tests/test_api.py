"""API 层单测：只测不加载模型的路径(健康检查、入参校验)。"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_no_model_load():
    r = client.get("/health")
    assert r.status_code == 200 and "model" in r.json()


def test_search_empty_query_422():
    assert client.post("/api/search", json={"query": ""}).status_code == 422


def test_embed_empty_list_422():
    assert client.post("/api/embed", json={"texts": []}).status_code == 422
