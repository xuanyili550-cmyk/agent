"""
 Deep RL Course · Ch5 · 基础案例：策略平均回报评估(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np
rng = np.random.default_rng(0)
hit = {0: 0.3, 1: 0.7}
for name, act in [("总选动作1", 1), ("总选动作0", 0)]:
    avg = np.mean([1.0 if rng.random() < hit[act] else 0.0 for _ in range(200)])
    print(f"{name}: 平均回报 {avg:.2f}")
print("✅ 评估策略=跑多条回合取平均回报;RL 目标就是找平均回报最高的策略")
