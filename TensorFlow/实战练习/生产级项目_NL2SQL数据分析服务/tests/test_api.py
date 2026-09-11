# 【构建 14/16 · 测试】依赖 main(11) 经 TestClient
"""接口层测试：健康检查 + 问数 + 输入校验 + 请求头。"""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_city_sales(client):
    r = client.post("/api/query", json={"question": "各城市销售额是多少？"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["sql"].upper().startswith("SELECT")
    cities = {row["city"] for row in body["rows"]}
    assert {"北京", "上海", "广州"} <= cities
    assert "X-Request-ID" in r.headers                # 中间件注入了 request-id


def test_query_category_count(client):
    r = client.post("/api/query", json={"question": "每个类目有几个商品？"})
    assert r.status_code == 200
    assert any("category" in row for row in r.json()["rows"])


def test_empty_question_rejected(client):
    r = client.post("/api/query", json={"question": ""})
    assert r.status_code == 422                        # pydantic min_length 校验
