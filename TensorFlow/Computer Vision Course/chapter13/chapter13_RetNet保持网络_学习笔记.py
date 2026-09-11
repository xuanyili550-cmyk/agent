"""
================================================================================
 CV Course · Chapter 13 · RetNet 保持网络（学习笔记 · 并行=循环 numpy 可跑）
================================================================================
 一句话：RetNet 用"保持(Retention)"替代注意力,同一套权重可【并行训练】也可【循环推理(O(1))】。
 本章讲(纯 numpy 证明 retention 的并行式与循环式给出相同结果——这是 RetNet 的核心卖点)：
   ① 并行式:out = (Q·Kᵀ ⊙ D)·V,D 是带衰减 γ 的因果衰减矩阵(训练时 GPU 并行)。
   ② 循环式:状态 S_n = γ·S_{n-1} + K_nᵀ·V_n,out_n = Q_n·S_n(推理时 O(1) 内存,免 KV Cache)。
   ③ 两者数学等价 → 训练用并行、推理用循环,鱼和熊掌兼得。
 要点：并行训练 + 循环推理"双形态"是 RetNet 对 Transformer 的核心改进。
 跑：python3 chapter13_RetNet保持网络_学习笔记.py   （并行/循环等价性纯 numpy 真跑)
================================================================================
"""
import numpy as np


def retention_parallel(Q, K, V, gamma):
    n = len(Q)
    D = np.array([[gamma ** (i - j) if i >= j else 0.0 for j in range(n)] for i in range(n)])
    return ((Q @ K.T) * D) @ V                     # 因果衰减矩阵 D 实现"越远衰减越多"


def retention_recurrent(Q, K, V, gamma):
    d_k, d_v = K.shape[1], V.shape[1]
    S = np.zeros((d_k, d_v)); out = []
    for n in range(len(Q)):
        S = gamma * S + np.outer(K[n], V[n])       # 状态递推(O(1) 内存)
        out.append(Q[n] @ S)
    return np.array(out)


def main():
    rng = np.random.default_rng(0)
    T, d = 6, 4
    Q, K, V = rng.standard_normal((T, d)), rng.standard_normal((T, d)), rng.standard_normal((T, d))
    gamma = 0.9
    par = retention_parallel(Q, K, V, gamma)
    rec = retention_recurrent(Q, K, V, gamma)
    assert np.allclose(par, rec, atol=1e-10)        # ★并行式与循环式结果完全一致
    print(f"✅ Ch13 跑通：retention 并行式 == 循环式(误差 {np.abs(par-rec).max():.1e}) → 训练并行+推理O(1)")
    # 面试：Q RetNet 核心卖点? A 同权重并行训练+循环推理(免KVCache,O(1)); Q 对比注意力? A 用衰减保持替代 softmax 注意力。


if __name__ == "__main__":
    main()
