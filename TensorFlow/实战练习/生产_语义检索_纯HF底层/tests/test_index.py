"""向量库单测：合成归一化向量，验证余弦检索返回最近邻(不下模型)。"""
import numpy as np

from app.services.index import VectorIndex


def _unit(v):
    v = np.array(v, dtype="float32")
    return v / np.linalg.norm(v)


def test_search_returns_nearest():
    idx = VectorIndex()
    idx.clear()
    idx.add(["a", "b", "c"], np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])]))
    res = idx.search(_unit([0.9, 0.1]), k=2)      # 最接近 a([1,0])
    assert res[0][0] == "a"
    assert res[0][1] >= res[1][1]                 # 分数降序
    assert idx.stats()["count"] == 3 and idx.stats()["dim"] == 2


def test_add_overwrites_same_id():
    idx = VectorIndex()
    idx.clear()
    idx.add(["x"], np.stack([_unit([1, 0])]))
    idx.add(["x"], np.stack([_unit([0, 1])]))     # 覆盖同 id
    assert idx.stats()["count"] == 1
