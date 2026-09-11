"""
 Deep RL Course · Ch7 · 基础案例：协作博弈收益取决于双方(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np
payoff = np.array([[0, 0], [0, 2]])   # 都选1 才共赢
for aA in (0, 1):
    for aB in (0, 1):
        print(f"A={aA}, B={aB} → 回报 {payoff[aA, aB]}")
print("✅ 回报由'动作组合'决定 → 单 agent 看到的环境随对方策略而变(非平稳)")
