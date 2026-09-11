"""
================================================================================
 服务层 · 重排 rerank（cross-encoder，底层原理：AutoModelForSequenceClassification → logit）
================================================================================
 【为什么要重排(两阶段检索)】向量检索是"双塔"：query 和 doc 各自独立编码，快但粗(建模不到细粒度交互)。
   cross-encoder 把 [query, doc] 拼在一起过一遍模型，直接输出一个"相关性 logit"，更准但慢。
   所以生产做法：向量先粗召回 top-N(快) → cross-encoder 精排取 top-k(准)，兼顾快与准。
 【底层】用 AutoModelForSequenceClassification(num_labels=1)：输入句对 → logits[:,0] 就是相关性分数，
   sigmoid 压到 0~1 便于阈值/展示。只排序的话 sigmoid 单调、直接按 logit 排也行。
================================================================================
"""
from ..core.config import get_settings
from .model import get_reranker


def rerank(query: str, candidates: list[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
    """candidates:[(id, text)] → 用 cross-encoder 重新打分并排序，返回 [(id, score)] 前 top_k。"""
    import torch
    if not candidates:
        return []
    s = get_settings()
    tok, model, dev = get_reranker()
    pairs = [[query, text] for _, text in candidates]        # 句对：查询 + 每个候选文档
    enc = tok(pairs, padding=True, truncation=True, max_length=s.rerank_max_seq_len,
              return_tensors="pt").to(dev)
    with torch.inference_mode():
        logits = model(**enc).logits                          # [N, 1] 相关性 logit
    scores = torch.sigmoid(logits.squeeze(-1)).cpu().tolist() # 压到 0~1
    ranked = sorted(zip((c[0] for c in candidates), scores), key=lambda x: -x[1])
    return ranked[:top_k]
