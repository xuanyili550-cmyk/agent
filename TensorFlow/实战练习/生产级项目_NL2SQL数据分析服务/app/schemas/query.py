# 【构建 6/16 · 契约】无内部依赖；供 routers 用
"""请求/响应模型(pydantic)：API 契约 + 自动校验 + 文档。"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="自然语言问数据的问题")


class QueryResponse(BaseModel):
    ok: bool = True
    question: str
    sql: str
    rows: list[dict]
    answer: str
    steps: list[str] = Field(default_factory=list, description="Agent 执行轨迹")
    request_id: str = "-"
