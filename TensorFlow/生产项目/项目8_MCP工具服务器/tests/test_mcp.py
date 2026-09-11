from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_list(): assert len(client.get("/api/tools/list").json()["tools"]) >= 2
def test_call_add():
    assert client.post("/api/tools/call", json={"name": "add", "args": {"a": 2, "b": 3}}).json()["result"] == 5
def test_call_unknown():
    assert client.post("/api/tools/call", json={"name": "nope", "args": {}}).status_code == 404
