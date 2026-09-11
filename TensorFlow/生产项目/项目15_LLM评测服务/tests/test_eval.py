from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
def test_health(): assert client.get("/health").status_code == 200
def test_evaluate():
    cases=[{"answer":"7天无理由","reference":"7天","context":"退货7天无理由"},
           {"answer":"保修三年","reference":"一年","context":"整机保修一年"}]
    r=client.post("/api/evaluate", json={"cases":cases}).json()
    assert r["n"]==2 and 0<=r["hallucination_rate"]<=1 and r["accuracy"]==0.5  # 第2条幻觉且不准
def test_validation(): assert client.post("/api/evaluate", json={"cases":[]}).status_code == 422


def test_prompt_ab():
    cases = [
        {"reference": "7天", "context": "退货7天无理由"},          # 有据
        {"reference": "一年", "context": "保修政策暂未公布"},       # 未知(陷阱:资料没说)
    ]
    A = "你是助手。只根据资料回答,无据就说未提及。资料:{context} 问题:{query}"   # 含据实约束
    B = "你是助手。回答问题。资料:{context} 问题:{query}"                     # 无约束
    r = client.post("/api/ab_eval", json={"cases": cases, "template_a": A, "template_b": B}).json()
    assert r["winner"] == "A"                                  # 据实模板胜
    assert r["A"]["hallucination_rate"] < r["B"]["hallucination_rate"]   # A 幻觉更低
