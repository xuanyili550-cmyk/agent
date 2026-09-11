"""
================================================================================
 服务层 · 文本 → 向量（手工前向：分词 → AutoModel → 池化 → 归一化 → 批处理）
================================================================================
 【这就是"pipeline 内部三步"的生产手写版】
   ① 分词：AutoTokenizer(padding 对齐、truncation 截断、返回 attention_mask 标出真实 token)。
   ② 前向：AutoModel 出 last_hidden_state([B,L,H])；inference_mode 关梯度(省显存/更快)。
   ③ 池化+归一化：mask 加权 mean 或 CLS(见 pooling.py) → L2 归一化 → 得到可检索的句向量。
 【动态批处理】文本按 batch_size 分批前向——一次算一批 GPU 利用率高;不是逐条(慢、GPU 空转)。
 【前缀】有的模型(E5/bge)要给 query/passage 加指令前缀才发挥最好——可配。
================================================================================
"""
import numpy as np

from ..core.config import get_settings
from . import pooling
from .model import get_encoder


def _encode(texts: list[str], prefix: str) -> np.ndarray:
    import torch
    s = get_settings()
    tok, model, dev = get_encoder()
    vectors = []
    for i in range(0, len(texts), s.batch_size):                 # 动态批处理
        batch = [prefix + t for t in texts[i:i + s.batch_size]]  # 需要就加前缀
        # ① 分词：padding 到本批最长、超长截断、返回 attention_mask
        enc = tok(batch, padding=True, truncation=True, max_length=s.max_seq_len,
                  return_tensors="pt").to(dev)
        # ② 前向：拿每个 token 的隐藏向量(不需要梯度)
        with torch.inference_mode():
            out = model(**enc)
        # ③ 池化(mask 加权/CLS) → 可选 L2 归一化
        vec = pooling.pool(out.last_hidden_state, enc["attention_mask"], s.pooling)
        if s.normalize:
            vec = pooling.l2_normalize(vec)
        vectors.append(vec.cpu().float().numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), dtype="float32")


def embed_queries(texts: list[str]) -> np.ndarray:
    """把"查询"编码成向量(用 query 前缀)。"""
    return _encode(texts, get_settings().query_prefix)


def embed_passages(texts: list[str]) -> np.ndarray:
    """把"文档/段落"编码成向量(用 passage 前缀)。"""
    return _encode(texts, get_settings().passage_prefix)
