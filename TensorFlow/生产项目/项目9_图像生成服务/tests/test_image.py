from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_generate():
    r = client.post("/api/generate", json={"prompt": "a cat", "steps": 20}).json()
    assert len(r["image_id"]) == 12 and r["steps"] == 20
def test_determinism():
    a = client.post("/api/generate", json={"prompt": "x"}).json()["image_id"]
    b = client.post("/api/generate", json={"prompt": "x"}).json()["image_id"]
    assert a == b
def test_validation(): assert client.post("/api/generate", json={"prompt": ""}).status_code == 422
