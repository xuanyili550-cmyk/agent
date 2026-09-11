from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_tool_route():
    r = client.post("/api/agent", json={"query": "帮我算 12*(3+4)"}).json()
    assert "84" in r["answer"] and r["route"] == "tool"
def test_direct(): assert client.post("/api/agent", json={"query": "你好"}).json()["route"] == "direct"
def test_validation(): assert client.post("/api/agent", json={"query": ""}).status_code == 422
