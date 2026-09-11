from fastapi.testclient import TestClient
from app.main import app
from app.services import moderate as mod
client = TestClient(app)


def test_pass_clean():
    assert mod.moderate("今天天气不错")["grade"] == "pass"


def test_block_sensitive():
    r = mod.moderate("这是诈骗暴力违禁内容")    # 3 个敏感词 → score≥0.7 → block
    assert r["grade"] == "block" and len(r["sensitive"]) >= 2


def test_pii_mask():
    r = mod.moderate("联系我 13812345678 或 a@b.com")
    assert r["pii"]["phone"] and "****" in r["masked"] and "***@" in r["masked"]


def test_api_and_validation():
    assert client.get("/health").status_code == 200
    assert client.post("/api/moderate", json={"text": "正常内容"}).json()["grade"] == "pass"
    assert client.post("/api/moderate", json={"text": ""}).status_code == 422


def test_batch_job():
    jid = client.post("/api/moderate/batch", json={"texts": ["正常", "诈骗赌博违禁"]}).json()["job_id"]
    import time; time.sleep(0.2)
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["status"] in ("done", "running")
