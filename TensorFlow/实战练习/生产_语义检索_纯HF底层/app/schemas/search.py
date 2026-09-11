"""API 请求/响应模型(pydantic)：入参校验 + 接口契约 + 自动文档。"""
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="待编码文本列表")


class EmbedResponse(BaseModel):
    dim: int
    vectors: list[list[float]]


class Document(BaseModel):
    id: str
    text: str = Field(..., min_length=1)


class IndexRequest(BaseModel):
    documents: list[Document] = Field(..., min_length=1)


class IndexStats(BaseModel):
    count: int
    dim: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=100)
    rerank: bool | None = None


class Hit(BaseModel):
    id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[Hit]
