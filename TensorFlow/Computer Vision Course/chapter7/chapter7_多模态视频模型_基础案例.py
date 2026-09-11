"""
 CV Course · Ch7 · 基础案例：视频帧特征时序池化(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np

video = np.random.default_rng(0).standard_normal((5, 8))   # 5 帧 × 8 维
print("逐帧特征 shape:", video.shape)
print("mean 池化 → 视频向量:", video.mean(0).round(2))
print("✅ T 帧特征聚合成 1 个视频级向量(最简:时序 mean)")
