"""
 ML for 3D · Ch3 · 基础案例：极简高斯泼溅(3D 点→2D 图,纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

size = 12
pts = np.array([[-0.5, -0.5, 0.1], [0.5, 0.5, 0.9], [0.0, 0.0, 0.5]])
cols = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
img = np.zeros((size, size, 3))
for i in np.argsort(-pts[:, 2]):                   # 远→近
    px = int((pts[i, 0] * 0.5 + 0.5) * (size - 1))
    py = int((pts[i, 1] * 0.5 + 0.5) * (size - 1))
    img[py, px] = cols[i]
print("✅ 3 个 3D 高斯泼溅到 12×12 图,非空像素:", int((img.sum(-1) > 0).sum()))
