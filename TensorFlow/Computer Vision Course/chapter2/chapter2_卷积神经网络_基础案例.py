"""
 CV Course · Ch2 · 基础案例：手写卷积做边缘检测(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

img = np.zeros((5, 5)); img[:, 2:] = 1        # 中间一条竖边
kernel = np.array([[-1, 0, 1]] * 3)            # 竖直边缘核
out = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        out[i, j] = (img[i:i+3, j:j+3] * kernel).sum()
print("卷积响应(边缘处最大):\n", out)
print("✅ 卷积核滑窗'逐元素乘求和'即可提取边缘等局部特征")
