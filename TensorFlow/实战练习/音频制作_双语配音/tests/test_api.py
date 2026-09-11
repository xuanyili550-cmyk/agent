from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and "deps" in r.json()


def test_analyze():
    r = client.post("/api/analyze", json={"text": "Hello 你好，welcome。"})
    assert r.status_code == 200
    body = r.json()
    assert body["stats"]["en_words"] >= 2
    assert any(t.get("ipa") for s in body["sentences"] for t in s["tokens"])


def test_analyze_empty_422():
    assert client.post("/api/analyze", json={"text": ""}).status_code == 422


def test_job_not_found():
    assert client.get("/api/jobs/nope").status_code == 404
