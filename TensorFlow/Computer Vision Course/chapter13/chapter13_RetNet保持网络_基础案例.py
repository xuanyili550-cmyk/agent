"""
 CV Course · Ch13 · 基础案例：RetNet retention 的并行式 == 循环式(纯 numpy,可跑)
 RetNet 的杀手锏:同一套计算有两种等价形态。训练用「并行式」(像注意力,一次算完全序列、可并行);
 推理用「循环式」(像 RNN,只维护固定大小状态 S,O(1) 内存、免 KV Cache)。二者数学上完全相等。
   循环式: S_n = γ·S_{n-1} + K_nᵀV_n,  out_n = Q_n·S_n
   并行式: Out = (QKᵀ ⊙ D)·V,  其中衰减矩阵 D[n,m] = γ^(n-m) (n≥m) 否则 0(因果掩码+距离衰减)
 这里两条路径都从零算,验证输出逐元素相等——这就是 RetNet"训练可并行、推理省内存"的根基。跑：python3 本文件
"""
import numpy as np


def retention_recurrent(Q, K, V, gamma):
    """循环式:逐步递推固定大小状态 S(d×d),只存一个 S 不存历史 → 推理 O(1) 内存。"""
    T, d = Q.shape
    S, out = np.zeros((d, d)), []
    for n in range(T):
        S = gamma * S + np.outer(K[n], V[n])      # K_nᵀV_n 累加,旧状态按 γ 衰减
        out.append(Q[n] @ S)                      # out_n = Q_n·S_n
    return np.array(out)


def retention_parallel(Q, K, V, gamma):
    """并行式:一次算完整序列。衰减矩阵 D 同时充当因果掩码(下三角)+ 距离衰减(γ^(n-m))。"""
    T = Q.shape[0]
    idx = np.arange(T)
    exponent = idx[:, None] - idx[None, :]        # D[n,m] 的指数 = n-m
    D = np.where(exponent >= 0, gamma ** np.clip(exponent, 0, None), 0.0)  # 上三角(未来)置 0
    A = (Q @ K.T) * D                             # 相关性 QKᵀ 逐元素乘衰减矩阵
    return A @ V


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d, gamma = 6, 4, 0.9
    Q, K, V = (rng.standard_normal((T, d)) for _ in range(3))

    out_rec = retention_recurrent(Q, K, V, gamma)
    out_par = retention_parallel(Q, K, V, gamma)

    assert out_rec.shape == out_par.shape == (T, d)
    assert np.allclose(out_rec, out_par, atol=1e-10)   # ★ 两种形态逐元素相等,这是 RetNet 的核心恒等式
    max_diff = np.abs(out_rec - out_par).max()

    print("循环式输出 shape:", out_rec.shape, " 并行式输出 shape:", out_par.shape)
    print(f"两种形态最大逐元素差: {max_diff:.2e}  (≈0 → 完全等价)")
    print("循环式状态 S 恒为 d×d =", (d, d), "→ 与序列长 T 无关,推理内存 O(1)、免 KV Cache")
    print("✅ 并行式(训练可并行)== 循环式(推理省内存):同一计算两副面孔,正是 RetNet 的根基")
    # 面试 Q&A：RetNet 相比标准注意力+KV Cache 好在哪?——注意力推理要缓存全部历史 K/V,内存随序列线性增长;
    #          RetNet 循环式只维护固定 d×d 状态 S,内存 O(1) 且每步 O(d²),长序列推理更省内存、更稳。
