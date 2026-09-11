from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_act_positive():
    r = client.post("/api/act", json={"observation": [1.0, 2.0, 3.0]}).json()
    assert r["action"][0] == 1.0
def test_act_negative():
    assert client.post("/api/act", json={"observation": [-5.0, -1.0]}).json()["action"][0] == -1.0
def test_validation(): assert client.post("/api/act", json={"observation": []}).status_code == 422
