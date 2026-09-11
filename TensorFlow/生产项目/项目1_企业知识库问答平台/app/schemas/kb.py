"""API 契约(pydantic):校验 + 文档。"""
from pydantic import BaseModel, Field
class IngestRequest(BaseModel):
    name: str = Field(..., min_length=1); text: str = Field(..., min_length=1)
class IngestResponse(BaseModel):
    chunks: int; dim: int; security: dict = {}
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
class Source(BaseModel):
    id: str; score: float
class QueryResponse(BaseModel):
    answer: str; route: str; sources: list[Source]; security: dict = {}
