from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def _chat(msg): return client.post("/api/chat", json={"messages": [{"role": "user", "content": msg}]})
def test_health(): assert client.get("/health").status_code == 200
def test_chat_stub():
    r = _chat("你好呀")
    assert r.status_code == 200 and "你好呀" in r.json()["content"] and r.json()["cached"] is False
def test_cache_hit():
    _chat("同一个问题")
    assert _chat("同一个问题").json()["cached"] is True     # 第二次命中缓存
def test_validation():
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
