"""
 CV Course · Ch3 · 基础案例：窗口内自注意力(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

win = np.random.default_rng(0).standard_normal((4, 6))   # 一个窗口:4 个 token
attn = softmax(win @ win.T / np.sqrt(6))                  # 窗口内注意力矩阵
out = attn @ win
print("窗口注意力矩阵(每行和=1):", attn.sum(1).round(2))
print("✅ 只在 4 个 token 的窗口内算注意力 → 输出", out.shape)
