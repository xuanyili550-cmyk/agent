from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_train_curve():
    r = client.post("/api/train", json={"env": "CartPole", "steps": 5}).json()
    assert r["reward_curve"][-1] > r["reward_curve"][0]    # 奖励递增
def test_eval(): assert client.post("/api/eval", json={"env": "CartPole"}).json()["avg_reward"] > 0
def test_validation(): assert client.post("/api/train", json={"env": ""}).status_code == 422
