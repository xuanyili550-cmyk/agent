from fastapi.testclient import TestClient
from app.main import app
from app.services import pipeline, registry, dataset
from app.core.exceptions import ValidationError
import pytest, time
client = TestClient(app)
SAMPLES = [{"instruction": "你好", "output": "你好呀"}, {"instruction": "1+1", "output": "2"}]


def test_dataset_validation():
    assert dataset.validate(SAMPLES) == 2
    with pytest.raises(ValidationError):
        dataset.validate([{"instruction": "x"}])        # 缺 output


def test_pipeline_offline():
    registry.clear()
    r = pipeline.run(SAMPLES)
    m = r["metrics"]
    assert m["final_loss"] < m["loss_curve"][0]          # loss 递减
    assert r["version"] == "v1" and registry.list_all()  # 已注册


def test_api_train_flow():
    jid = client.post("/api/train", json={"samples": SAMPLES}).json()["job_id"]
    time.sleep(0.3)
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["status"] in ("done", "running")
    assert client.get("/health").status_code == 200
