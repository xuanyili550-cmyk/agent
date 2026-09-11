"""
 Deep RL Course · Ch8 · 基础案例：PPO 裁剪目标(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np
adv, eps = 1.0, 0.2
for ratio in [0.5, 1.0, 1.5]:
    obj = min(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)
    print(f"ratio={ratio} (adv=1) → 裁剪目标 {obj:.2f}")
print("✅ ratio 偏离 1 太多就被 clip 截断 → 限制每步更新幅度,训练稳")
