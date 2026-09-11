# 【构建 10/16 · 接口】依赖 schemas(6)+database(5)+nl2sql(8)
"""问数接口：POST /api/query。依赖注入 DB 会话，调 NL2SQL 服务。"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.logging import request_id_ctx
from app.db.database import get_db
from app.schemas.query import QueryRequest, QueryResponse
from app.services.nl2sql import NL2SQLService

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, db: Session = Depends(get_db)):
    result = NL2SQLService(db).run(req.question)
    return QueryResponse(
        question=req.question, sql=result["sql"], rows=result["rows"],
        answer=result["answer"], steps=result["steps"], request_id=request_id_ctx.get(),
    )
