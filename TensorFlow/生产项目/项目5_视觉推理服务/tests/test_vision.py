from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_classify():
    r = client.post("/api/classify", json={"image_b64": "fakeimg"})
    assert r.status_code == 200 and r.json()["label"] in ["cat","dog","car","flower","person"]
def test_detect():
    assert client.post("/api/detect", json={"image_b64": "x"}).json()["boxes"]
def test_validation(): assert client.post("/api/classify", json={"image_b64": ""}).status_code == 422
