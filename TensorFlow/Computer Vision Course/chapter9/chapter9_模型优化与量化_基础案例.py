"""
 CV Course · Ch9 · 基础案例：int8 量化/反量化(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

w = np.array([-1.0, -0.3, 0.0, 0.7, 1.0], np.float32)
lo, hi = w.min(), w.max(); scale = (hi - lo) / 255
q = np.round((w - lo) / scale).astype(int)
w_hat = q * scale + lo
print("原始:", w)
print("int8:", q, " (0~255)")
print("还原:", w_hat.round(3), " 误差:", np.abs(w - w_hat).max().round(4))
print("✅ float32→int8 体积缩 4 倍,误差极小")
