"""
 Audio Course · Ch3 · 基础案例：从零手写自注意力（音频 Transformer 的心脏）
 音频 Transformer(Whisper/wav2vec)= CNN 前端把波形降采样成一串帧向量 → 若干层自注意力。
 自注意力做的事："每一帧用自己的 Query 去和所有帧的 Key 算相似度，softmax 成权重，
 再对所有帧的 Value 加权求和"，从而让每帧融合全局上下文。这里纯 numpy 把这套真算出来。
 跑：python3 本文件（纯 numpy，无需联网/GPU）
"""
import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))         # 减最大值：防指数溢出
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """缩放点积注意力：核心公式 softmax(QKᵀ/√d) · V。这是整个 Transformer 最关键的一步。"""
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)                           # 相似度矩阵 [q帧, k帧]，除√d 稳梯度
    if mask is not None:
        scores = np.where(mask, scores, -1e9)               # 掩码位置置 -∞ → softmax 后≈0
    weights = softmax(scores, axis=-1)                      # 每行(每个 query)权重和=1
    return weights @ V, weights                             # 用权重对 Value 加权求和


def multi_head_attention(X, Wq, Wk, Wv, n_heads):
    """多头：把 d 维拆成 n_heads 份并行做注意力再拼回 —— 让模型同时关注不同类型的关系。"""
    T, d = X.shape
    dh = d // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                        # 线性投影出 Q/K/V
    outs = []
    for h in range(n_heads):                                # 每个头看自己那 dh 维的子空间
        s = slice(h * dh, (h + 1) * dh)
        o, _ = scaled_dot_product_attention(Q[:, s], K[:, s], V[:, s])
        outs.append(o)
    return np.concatenate(outs, axis=1)                     # 拼回 [T, d]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, d, n_heads = 5, 8, 2                                 # 5 帧音频、每帧 8 维、2 头
    X = rng.standard_normal((T, d))
    Wq, Wk, Wv = (rng.standard_normal((d, d)) for _ in range(3))

    # ① 单头自注意力：验证注意力权重是合法概率分布（每行和=1）
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    ctx, weights = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(weights.sum(axis=1), 1.0)            # 每帧对所有帧的权重和恰为 1
    assert ctx.shape == (T, d)

    # ② 查询会关注和它最相似的键：把 K 归一化成单位向量，令第0帧的 Query 正好指向第2帧的 Key
    Kn = K / np.linalg.norm(K, axis=1, keepdims=True)      # 单位化，点积=余弦相似度
    Q2 = Q.copy(); Q2[0] = Kn[2]                           # Query0 与 Key2 方向完全一致
    _, w2 = scaled_dot_product_attention(Q2, Kn, V)
    assert w2[0].argmax() == 2                             # 故第0帧的注意力峰值必落在第2帧

    # ③ 多头输出形状与单头一致，可堆叠成多层
    mh = multi_head_attention(X, Wq, Wk, Wv, n_heads)
    assert mh.shape == (T, d)
    print(f"✅ 自注意力跑通：权重每行和=1、相似帧互相高关注、多头输出 shape={mh.shape}")
    print(f"   第0帧注意力分布(和={weights[0].sum():.2f})：{np.round(weights[0], 2)}")

# 面试 Q&A：注意力里为什么要除以 √d？点积会随维度 d 增大而方差变大，使 softmax 进入饱和区
# (梯度趋零、退化成 one-hot)；除 √d 把方差拉回，保持权重平滑、梯度健康。
