"""
 CV Course · Ch1 · 基础案例：Lowe 比率测试筛描述子匹配(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(1)
A = rng.standard_normal((3, 4))
B = np.vstack([A + 0.01, rng.standard_normal((3, 4))])   # 前 3 个是 A 的近似
for i, a in enumerate(A):
    d = np.linalg.norm(B - a, axis=1)
    j1, j2 = np.argsort(d)[:2]
    ok = d[j1] < 0.75 * d[j2]
    print(f"A[{i}] → B[{j1}]  {'✅好匹配' if ok else '✗ 模糊,丢弃'}")
