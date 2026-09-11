from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_gen3d():
    r = client.post("/api/generate3d", json={"prompt": "a toy car"}).json()
    assert r["num_gaussians"] == 2048 and r["params_per_gaussian"] == 14
def test_validation(): assert client.post("/api/generate3d", json={"prompt": ""}).status_code == 422
