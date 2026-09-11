"""
================================================================================
 CV Course · Chapter 3 · Swin Transformer 窗口注意力（学习笔记 · numpy 可跑）
================================================================================
 一句话：把 Transformer 用到视觉的高效做法——只在"局部窗口"内做自注意力,再"移窗"让窗口间通信。
 本章讲(纯 numpy 演示 窗口划分 + 窗口内自注意力,无需 torch)：
   ① 全局注意力对 N 个 patch 是 O(N²);Swin 只在小窗口内算 → 线性复杂度,适合高分辨率图。
   ② 窗口内自注意力：softmax(QKᵀ/√d)·V,只在窗口内的 token 间。
   ③ 移动窗口(Shifted Window)：下一层把窗口错位,让相邻窗口信息流动。
 要点：窗口注意力省算力,移窗补"跨窗口"通信;应用如超分 SwinIR。真实实现见 🔴torch。
 跑：python3 chapter3_Swin窗口注意力_学习笔记.py   （窗口注意力纯 numpy 真跑)
================================================================================
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def window_attention(tokens):
    """窗口内单头自注意力:tokens [win, d] → 同形。"""
    d = tokens.shape[-1]
    w = softmax(tokens @ tokens.T / np.sqrt(d))
    return w @ tokens


def main():
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((16, 8))          # 16 个 patch,8 维
    win = 4                                      # 窗口大小 4
    # 划分成 16/4=4 个窗口,各自做注意力(Swin 的核心:注意力只在窗口内)
    outs = [window_attention(seq[i:i + win]) for i in range(0, 16, win)]
    out = np.vstack(outs)
    assert out.shape == (16, 8)
    # 对比:全局注意力是 16×16 打分;窗口注意力是 4 个 4×4,省算力
    print(f"✅ Ch3 跑通：4 个窗口各做 4×4 注意力(而非全局 16×16) → 输出 {out.shape},省算力")
    print("   移窗:下一层窗口错位,让相邻窗口 token 也能交互(补跨窗口通信)。")
    # 面试：Q Swin 为何省算力? A 注意力只在局部窗口(线性 vs 平方); Q 移窗干嘛? A 让窗口间信息流动。


if __name__ == "__main__":
    main()
