"""
 Deep RL Course · Ch6 · 基础案例：观测归一化(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np
data = np.random.default_rng(0).normal([10, 100], [2, 50], size=(1000, 2))  # 量纲悬殊
z = (data - data.mean(0)) / (data.std(0) + 1e-8)
print("归一化前 std:", data.std(0).round(1), " → 归一化后 std:", z.std(0).round(2))
print("✅ 机器人 RL 观测量纲差异大,归一化到~N(0,1)才好训(VecNormalize)")
