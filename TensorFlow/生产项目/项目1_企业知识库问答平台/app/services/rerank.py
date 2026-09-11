"""重排:默认轻量词面重合重排(离线);生产可开 cross-encoder(🔴需模型)。"""
import re
from ..core.config import get_settings
_TOK = re.compile(r"[a-z0-9]+|[一-鿿]")


def _overlap(q, t):
    qs, ts = set(_TOK.findall(q.lower())), set(_TOK.findall(t.lower()))
    return len(qs & ts) / (len(qs) + 1e-9)


def rerank(query, hits, top_k):
    """hits=[(id,text,score)] → 结合向量分 + 词面重合 重排,取 top_k。"""
    if not get_settings().rerank_enabled:
        return hits[:top_k]
    scored = [(i, t, 0.5 * s + 0.5 * _overlap(query, t)) for i, t, s in hits]
    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]
