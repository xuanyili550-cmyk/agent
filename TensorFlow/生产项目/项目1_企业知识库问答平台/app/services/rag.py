"""RAG 编排:入库(解析→切块→嵌入→索引) + 问答(嵌入→检索→重排→LLM)。接入注入防御。"""
from ..core.config import get_settings
from ..core.exceptions import ValidationError
from . import chunk as chunk_mod, docparse, embedding, llm, rerank, security
from .index import get_index


def ingest(name: str, raw_text: str) -> dict:
    text = docparse.parse(raw_text, is_text=True)
    chunks = chunk_mod.chunk(text)
    vecs = embedding.embed(chunks)
    ids = [f"{name}#{i}" for i in range(len(chunks))]
    get_index().add(ids, chunks, vecs)
    stats = get_index().stats()
    stats["security"] = security.scan(raw_text)        # ④ 入库扫描文档是否藏指令
    return stats


def answer(query: str) -> dict:
    s = get_settings()
    if not (query or "").strip():
        raise ValidationError("问题不能为空")
    if len(query) > s.max_query_chars:
        raise ValidationError("问题过长")
    q_inj = security.detect_injection(query)            # ① 检测查询注入
    qvec = embedding.embed([query])[0]
    hits = get_index().search(qvec, s.top_k * 3 if s.rerank_enabled else s.top_k)
    hits = rerank.rerank(query, hits, s.top_k)
    raw_ctx = "\n".join(f"[{i+1}] {t}" for i, (_, t, _) in enumerate(hits))
    context = security.wrap_as_data(raw_ctx)            # ② 指令/数据分离后再喂 LLM
    ans = llm.answer(query, context)
    return {"answer": ans,
            "sources": [{"id": i, "score": round(sc, 3)} for i, _, sc in hits],
            "security": {"query_injection": bool(q_inj)}}
