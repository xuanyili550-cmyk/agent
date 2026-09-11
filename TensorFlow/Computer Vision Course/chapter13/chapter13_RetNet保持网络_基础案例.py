"""
 CV Course · Ch13 · 基础案例：retention 循环式状态递推(纯 numpy,可跑)
 S_n = γ·S_{n-1} + K_nᵀV_n,out_n = Q_n·S_n —— 推理 O(1) 内存,免 KV Cache。跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(0)
T, d = 4, 3
Q, K, V = rng.standard_normal((T, d)), rng.standard_normal((T, d)), rng.standard_normal((T, d))
gamma, S, out = 0.9, np.zeros((d, d)), []
for n in range(T):
    S = gamma * S + np.outer(K[n], V[n])     # 只存一个状态 S,不存历史(O(1))
    out.append(Q[n] @ S)
print("循环式输出 shape:", np.array(out).shape)
print("✅ 状态递推:每步只更新固定大小的 S → 推理 O(1) 内存,免 KV Cache")
