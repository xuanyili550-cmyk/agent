from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_asr(): assert "转写" in client.post("/api/asr", json={"audio_b64": "aaa"}).json()["text"]
def test_tts():
    r = client.post("/api/tts", json={"text": "你好世界"}).json()
    assert r["sample_rate"] == 22050 and r["duration_s"] > 0
def test_validation(): assert client.post("/api/tts", json={"text": ""}).status_code == 422
