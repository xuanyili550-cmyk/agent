"""
================================================================================
 服务层 · 向量库（numpy 余弦检索 + 持久化，线程安全）
================================================================================
 【为什么点积就够】上游已把向量 L2 归一化(单位向量)，所以 余弦 = 点积。整库检索 = 一次矩阵乘
   scores = 库矩阵[N,H] @ 查询[H] → 每个文档一个相似度，再取 top-k。简单、快、够小中规模用。
 【生产演进】上百万向量用近似最近邻(FAISS/HNSW、Qdrant/Milvus)；这里 numpy 精确检索，接口一致、可平替。
 【持久化】ids + 矩阵存成 .npz，重启不丢；加锁防并发 add/search 竞态。
================================================================================
"""
import os
import threading

import numpy as np

from ..core.config import get_settings


class VectorIndex:
    def __init__(self):
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None      # [N, H] float32，行=已归一化的文档向量
        self._lock = threading.Lock()
        self._load()

    def _path(self):
        d = get_settings().index_dir
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "index.npz")

    def _load(self):
        p = self._path()
        if os.path.exists(p):
            data = np.load(p, allow_pickle=True)
            self._ids = list(data["ids"])
            self._mat = data["mat"].astype("float32")

    def _save(self):
        np.savez(self._path(), ids=np.array(self._ids, dtype=object),
                 mat=self._mat if self._mat is not None else np.zeros((0, 0), "float32"))

    def add(self, ids: list[str], vectors: np.ndarray):
        """加入(或按 id 覆盖)一批向量。vectors:[M,H] 已归一化。"""
        with self._lock:
            vectors = vectors.astype("float32")
            for i, _id in enumerate(ids):
                if _id in self._ids:                          # 已存在 → 覆盖该行
                    self._mat[self._ids.index(_id)] = vectors[i]
                else:
                    self._ids.append(_id)
                    self._mat = vectors[i:i + 1] if self._mat is None else np.vstack([self._mat, vectors[i:i + 1]])
            self._save()

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        """余弦(=点积)检索 top-k。query_vec:[H] 已归一化 → 返回 [(id, score)]。"""
        with self._lock:
            if self._mat is None or not len(self._ids):
                return []
            scores = self._mat @ query_vec.astype("float32")   # [N] 每个文档的余弦
            k = min(k, len(self._ids))
            top = np.argpartition(-scores, k - 1)[:k]           # 部分排序取前 k(比全排快)
            top = top[np.argsort(-scores[top])]                 # 前 k 内再精排
            return [(self._ids[i], float(scores[i])) for i in top]

    def stats(self):
        return {"count": len(self._ids), "dim": int(self._mat.shape[1]) if self._mat is not None and self._mat.size else 0}

    def clear(self):
        with self._lock:
            self._ids, self._mat = [], None
            self._save()


_INDEX = None


def get_index() -> VectorIndex:
    """单例向量库。"""
    global _INDEX
    if _INDEX is None:
        _INDEX = VectorIndex()
    return _INDEX
