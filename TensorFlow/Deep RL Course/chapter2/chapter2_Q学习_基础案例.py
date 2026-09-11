"""
 Deep RL Course · Ch2 · 基础案例：Q 表 TD 更新一步(纯 numpy,可跑)
 Q(s,a) ← Q + α[r + γ·maxQ(s') − Q]。跑：python3 本文件
"""
import numpy as np

Q = np.zeros((3, 2))
s, a, r, s2 = 0, 1, 0.0, 1
alpha, gamma = 0.5, 0.9
Q[1] = [0.2, 0.8]                                  # 假设 s'=1 已有估值
Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
print("更新后 Q[0,1] =", round(Q[s, a], 3), "(= α·γ·maxQ(s') = 0.5·0.9·0.8)")
print("✅ TD 更新把'未来最优价值'折现回传到当前状态-动作")
