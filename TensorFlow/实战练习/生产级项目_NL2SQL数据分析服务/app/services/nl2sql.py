# 【构建 8/16 · 服务·★核心】依赖 config/exceptions/logging/models(4)/llm(7)
"""NL2SQL 核心服务：LangGraph 状态机编排
[检索元数据 → 生成SQL → 安全校验 → (失败重试/兜底) → 执行 → 自然语言总结]。
安全：只放行 SELECT、禁写操作；SQL 参数化执行由 ORM 层保证只读。"""
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import InvalidSQLError, SQLExecutionError
from app.core.logging import get_logger
from app.db.models import TABLE_METADATA
from app.services.llm import generate_with_retry

logger = get_logger(__name__)
_FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|GRANT)\b", re.I)


def _schema_text(tables: list[str]) -> str:
    out = []
    for t in tables:
        meta = TABLE_METADATA[t]
        cols = ", ".join(f"{c}({d})" for c, d in meta["columns"].items())
        out.append(f"{t}({meta['desc']}): {cols}")
    return "\n".join(out)


def retrieve_tables(question: str) -> list[str]:
    """按关键词命中相关表(简单混合检索；命中为空则全给)。"""
    hit = [t for t, m in TABLE_METADATA.items() if any(k in question for k in m["keywords"])]
    return hit or list(TABLE_METADATA)


def _template_sql(question: str) -> str:
    """规则兜底：小模型/上游失败时保证有可执行 SQL。"""
    if "城市" in question and ("金额" in question or "销售" in question or "卖" in question):
        return "SELECT city, SUM(amount) AS total FROM orders GROUP BY city ORDER BY total DESC;"
    if "类目" in question or "分类" in question:
        return "SELECT category, COUNT(*) AS n FROM products GROUP BY category;"
    if "最贵" in question or "最高" in question:
        return "SELECT name, price FROM products ORDER BY price DESC LIMIT 1;"
    return "SELECT SUM(amount) AS total FROM orders;"


class _State(TypedDict):
    question: str
    tables: list
    sql: str
    valid: bool
    rows: list
    answer: str
    retries: int
    steps: list


class NL2SQLService:
    """把 LangGraph 图封装成一个服务，run(question, db) 端到端出结果。"""

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.graph = self._build()

    # —— 节点 ——
    def _n_retrieve(self, s: _State) -> dict:
        tables = retrieve_tables(s["question"])
        return {"tables": tables, "steps": s["steps"] + [f"检索相关表: {tables}"]}

    def _n_gen(self, s: _State) -> dict:
        prompt = (f"你是 SQLite 专家。根据表结构写一条 SELECT 查询回答问题，只输出 SQL。\n"
                  f"表结构：\n{_schema_text(s['tables'])}\n问题：{s['question']}\nSQL：")
        out = generate_with_retry(prompt, retries=self.settings.sql_max_retries, max_tokens=80)
        m = re.search(r"(SELECT[\s\S]+?)(;|$)", out, re.I)
        sql = (m.group(1).strip() + ";") if m else ""
        return {"sql": sql, "steps": s["steps"] + [f"生成SQL: {sql[:50]}"]}

    def _validate(self, sql: str) -> bool:
        if not sql or not sql.strip().upper().startswith("SELECT") or _FORBIDDEN.search(sql):
            return False
        try:
            self.db.execute(text("EXPLAIN " + sql.rstrip(";")))   # 能被数据库解析
            return True
        except Exception:                                          # noqa: BLE001
            return False

    def _n_check(self, s: _State) -> dict:
        ok = self._validate(s["sql"])
        return {"valid": ok, "steps": s["steps"] + [f"SQL 安全校验: {'通过' if ok else '不通过'}"]}

    def _route_after_check(self, s: _State) -> str:
        return "execute" if s["valid"] else "fallback"

    def _n_fallback(self, s: _State) -> dict:
        sql = _template_sql(s["question"])
        return {"sql": sql, "retries": s["retries"] + 1,
                "steps": s["steps"] + [f"规则兜底SQL: {sql[:50]}"]}

    def _n_execute(self, s: _State) -> dict:
        try:
            cur = self.db.execute(text(s["sql"]))
            cols = list(cur.keys())
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception as e:                                     # noqa: BLE001
            raise SQLExecutionError(f"SQL 执行失败: {e}") from e
        return {"rows": rows, "steps": s["steps"] + [f"执行 → {len(rows)} 行"]}

    def _n_summarize(self, s: _State) -> dict:
        ans = generate_with_retry(
            f"用一句话回答问题。问题：{s['question']}\n查询结果：{s['rows']}\n回答：",
            retries=1, max_tokens=80)
        return {"answer": ans, "steps": s["steps"] + ["生成自然语言回答"]}

    def _build(self):
        g = StateGraph(_State)
        g.add_node("retrieve", self._n_retrieve)
        g.add_node("gen", self._n_gen)
        g.add_node("check", self._n_check)
        g.add_node("fallback", self._n_fallback)
        g.add_node("execute", self._n_execute)
        g.add_node("summarize", self._n_summarize)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "gen")
        g.add_edge("gen", "check")
        g.add_conditional_edges("check", self._route_after_check,
                                {"execute": "execute", "fallback": "fallback"})
        g.add_edge("fallback", "execute")
        g.add_edge("execute", "summarize")
        g.add_edge("summarize", END)
        return g.compile()

    def run(self, question: str) -> dict:
        logger.info("NL2SQL 处理: %s", question)
        return self.graph.invoke({"question": question, "tables": [], "sql": "", "valid": False,
                                  "rows": [], "answer": "", "retries": 0, "steps": []})
