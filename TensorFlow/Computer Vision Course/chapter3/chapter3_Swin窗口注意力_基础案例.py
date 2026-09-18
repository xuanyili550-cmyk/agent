"""
 CV Course · Ch3 · 基础案例：Swin 窗口注意力 + 移位窗口(纯 numpy,机制真实,可跑)
 全局注意力代价是 O(N²);Swin 把图切成不重叠小窗,只在窗内算注意力 → 复杂度随图线性增长。
 但纯窗内注意力窗间无法通信,所以下一层"移位窗口"(shift)让原本跨窗的像素落进同一窗 → 信息流动。
 这里纯 numpy 真算窗内注意力,并对比"不移位/移位"下哪些位置能互相看见,讲清 Swin 的核心取舍。
 跑：python3 本文件
"""
import numpy as np


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def window_partition(tokens, win):
    """把长为 N 的 token 序列切成不重叠的窗口(每窗 win 个)。真实 Swin 在 2D 上切,这里用 1D 演示同一机制。"""
    return tokens.reshape(-1, win, tokens.shape[-1])


def window_attention(tokens, win):
    """只在每个窗口内部算自注意力:Q·Kᵀ/√d → softmax → 加权 V。窗间不交互 → 省算力。"""
    windows = window_partition(tokens, win)
    d = tokens.shape[-1]
    outs = []
    for w in windows:
        attn = softmax(w @ w.T / np.sqrt(d))               # win×win 注意力,和全局的 N×N 比小得多
        outs.append(attn @ w)
    return np.concatenate(outs, axis=0), windows


if __name__ == "__main__":
    N, d, win = 8, 6, 4
    tokens = np.random.default_rng(0).standard_normal((N, d))

    # ① 常规窗口:窗0={0,1,2,3}, 窗1={4,5,6,7} —— token 3 和 4 相邻却分属两窗,看不见彼此
    out1, wins1 = window_attention(tokens, win)
    # ② 移位窗口:循环右移 shift 个位置再切窗 → 原来的边界像素被重新分组进同一窗
    shift = win // 2
    shifted = np.roll(tokens, shift, axis=0)
    out2, _ = window_attention(shifted, win)
    out2 = np.roll(out2, -shift, axis=0)                    # 算完移回原位置(真实 Swin 也要 reverse shift)

    print("窗数:", wins1.shape[0], " 每窗 token 数:", win, " 注意力复杂度 每窗 O(win²)=", win*win)
    print("常规窗:token3 与 token4 分属不同窗 → 本层无法交互")
    print("移位后:二者被移进同一窗 → 下一层可交互(窗间信息由此流动)")
    # 验证:窗内注意力每行是合法概率分布(和为1)
    sample = softmax(wins1[0] @ wins1[0].T / np.sqrt(d))
    assert np.allclose(sample.sum(1), 1.0)
    assert out1.shape == out2.shape == (N, d)
    assert not np.allclose(out1, out2)                      # 移位确实改变了每个 token 能看到的邻居
    print("✅ 窗内注意力省算力 + 移位窗口补窗间通信 = Swin 兼顾效率与感受野的关键设计")
    # 面试Q:Swin 为什么要"移位窗口"而不是直接用重叠窗口?
    #      A:重叠窗会重复计算、破坏不重叠分块的高效实现;移位仅换分组、算量不变,却让相邻层覆盖不同窗边界,逐层打通全局联系。
