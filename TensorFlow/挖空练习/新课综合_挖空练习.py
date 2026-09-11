"""
================================================================================
 新课综合 挖空练习 · 扩散前向 / 自注意力 / mask 池化（Audio+CV+Diffusion 综合）
================================================================================
 把几门新课的核心公式串一遍。玩法:遮住「练习N」下一行默写,再运行。python3 本文件 (纯 numpy)
================================================================================
"""
import numpy as np

# —— 练习1：扩散前向加噪 x_t = √ᾱ·x₀ + √(1-ᾱ)·ε ——
def q_sample(x0, abar_t, eps):
    return np.sqrt(abar_t) * x0 + np.sqrt(1 - abar_t) * eps   # ← 练习1

# —— 练习2：数值稳定 softmax(减最大值) ——
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)                   # ← 练习2:减最大值防上溢
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

# —— 练习3：自注意力 softmax(QKᵀ/√d)·V ——
def attention(Q, K, V):
    d = Q.shape[-1]
    return softmax(Q @ K.T / np.sqrt(d)) @ V                  # ← 练习3

# —— 练习4：mask 加权 mean 池化(排除 PAD) ——
def mean_pool(hidden, mask):
    m = mask[:, :, None]
    return (hidden * m).sum(1) / m.sum(1).clip(min=1e-9)      # ← 练习4


def main():
    rng = np.random.default_rng(0)
    assert abs(q_sample(np.array([1.0]), 0.25, np.array([0.0]))[0] - 0.5) < 1e-9
    assert abs(softmax(np.array([1000.0, 1000, 1000])).sum() - 1.0) < 1e-9   # 大数不溢出
    X = rng.standard_normal((4, 8))
    assert attention(X, X, X).shape == (4, 8)
    h = rng.standard_normal((2, 5, 8)); mask = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    assert mean_pool(h, mask).shape == (2, 8)
    print("✅ 新课综合挖空全部通过:扩散前向 + softmax + 自注意力 + mask 池化")


if __name__ == "__main__":
    main()

# ================================ 答案要点 ====================================
# 练习1: np.sqrt(abar_t)*x0 + np.sqrt(1-abar_t)*eps
# 练习2: x = x - x.max(axis=axis, keepdims=True)
# 练习3: softmax(Q @ K.T / np.sqrt(d)) @ V
# 练习4: (hidden*m).sum(1) / m.sum(1).clip(min=1e-9)
# ============================================================================
