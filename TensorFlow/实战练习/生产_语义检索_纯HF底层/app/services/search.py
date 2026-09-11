"""
================================================================================
 服务层 · 检索编排（嵌入 → 建库 → 检索 →(可选)重排）
================================================================================
 把底层能力串成业务流：入库时把文档编码成向量存进向量库+存原文；检索时把查询编码→向量库粗召回
 →(可选)cross-encoder 精排→回填原文返回。这就是一个最小但完整的"语义检索"闭环。
================================================================================
"""
from ..core.config import get_settings
from ..core.exceptions import ValidationError
from . import embedding, rerank, store
from .index import get_index


def index_documents(documents: list[dict]) -> dict:
    """documents:[{id, text}] → 编码入库 + 存原文。"""
    s = get_settings()
    if not documents:
        raise ValidationError("documents 不能为空。")
    if len(documents) > s.max_texts_per_request:
        raise ValidationError(f"单次入库过多(>{s.max_texts_per_request})。")
    ids = [str(d["id"]) for d in documents]
    texts = [d["text"] for d in documents]
    vecs = embedding.embed_passages(texts)           # 文档向量(passage 前缀)
    get_index().add(ids, vecs)
    store.put(dict(zip(ids, texts)))
    return get_index().stats()


def search(query: str, top_k: int | None = None, use_rerank: bool | None = None) -> list[dict]:
    """查询 → 向量粗召回 →(可选)重排 → 返回 [{id, text, score}]。"""
    s = get_settings()
    if not query or not query.strip():
        raise ValidationError("query 不能为空。")
    top_k = top_k or s.default_top_k
    do_rerank = s.rerank_enabled if use_rerank is None else use_rerank

    qvec = embedding.embed_queries([query])[0]       # 查询向量
    # 要重排就多召回一些(如 5*k)给精排挑，否则直接召回 k
    recall = get_index().search(qvec, top_k * 5 if do_rerank else top_k)
    hits = [(i, store.get(i), sc) for i, sc in recall]

    if do_rerank and hits:
        scored = rerank.rerank(query, [(i, t) for i, t, _ in hits], top_k)
        text_of = {i: t for i, t, _ in hits}
        return [{"id": i, "text": text_of.get(i, ""), "score": round(sc, 4)} for i, sc in scored]
    return [{"id": i, "text": t, "score": round(sc, 4)} for i, t, sc in hits[:top_k]]
