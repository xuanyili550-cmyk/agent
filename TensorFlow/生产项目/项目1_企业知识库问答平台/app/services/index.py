"""内存向量库:余弦(=归一化后点积)检索。大规模可平替 FAISS/Qdrant(接口一致)。"""
import threading
import numpy as np


class KBIndex:
    def __init__(self):
        self._ids, self._texts, self._mat = [], [], None
        self._lock = threading.Lock()

    def add(self, ids, texts, vecs):
        with self._lock:
            vecs = vecs.astype("float32")
            self._ids += list(ids); self._texts += list(texts)
            self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])

    def search(self, qvec, k):
        with self._lock:
            if self._mat is None:
                return []
            scores = self._mat @ qvec.astype("float32")
            k = min(k, len(self._ids))
            top = np.argsort(-scores)[:k]
            return [(self._ids[i], self._texts[i], float(scores[i])) for i in top]

    def stats(self):
        return {"chunks": len(self._ids), "dim": int(self._mat.shape[1]) if self._mat is not None else 0}

    def clear(self):
        with self._lock:
            self._ids, self._texts, self._mat = [], [], None


_INDEX = None


def get_index() -> KBIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = KBIndex()
    return _INDEX
