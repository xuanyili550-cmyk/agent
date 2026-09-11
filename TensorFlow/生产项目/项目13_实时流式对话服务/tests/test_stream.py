from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_ws_stream():
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": "你好"})
        toks=[]
        while True:
            d = ws.receive_json()
            if d.get("done"): break
            toks.append(d["token"])
        assert "".join(toks).startswith("收到:你好")     # 逐字流式拼回完整回复
def test_ws_empty():
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"message": ""})
        assert ws.receive_json().get("error")            # 空输入返回 error
