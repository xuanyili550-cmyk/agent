# 【构建 13/16 · 测试】依赖 services(8)
"""服务层测试：元数据检索 / SQL 安全校验 / 端到端 / 兜底。"""
from app.services.nl2sql import NL2SQLService, retrieve_tables


def test_retrieve_tables():
    assert retrieve_tables("各城市销售额") == ["orders"]
    assert retrieve_tables("每个类目商品数") == ["products"]
    assert set(retrieve_tables("你好")) == {"products", "orders"}   # 都不命中→全给


def test_sql_validation_blocks_writes(db):
    svc = NL2SQLService(db)
    assert svc._validate("SELECT * FROM products") is True
    assert svc._validate("DELETE FROM orders") is False           # 禁写
    assert svc._validate("DROP TABLE products") is False
    assert svc._validate("SELECT * FROM nope") is False           # 表不存在→解析失败
    assert svc._validate("") is False


def test_end_to_end(db):
    r = NL2SQLService(db).run("各城市销售额是多少？")
    assert r["sql"].upper().startswith("SELECT")
    assert r["rows"] and {"北京", "上海", "广州"} <= {row["city"] for row in r["rows"]}
    assert any("执行" in s for s in r["steps"])                   # 走了执行节点
    assert r["answer"]


def test_fallback_on_invalid(db):
    """LLM 给不出合法 SQL 时，走规则兜底仍能出结果(这里直接测兜底 SQL 可执行)。"""
    from app.services.nl2sql import _template_sql
    svc = NL2SQLService(db)
    sql = _template_sql("最贵的商品")
    assert svc._validate(sql) is True
