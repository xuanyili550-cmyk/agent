"""
 Deep RL Course · Ch3 · 基础案例：Bellman 目标计算(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np
r = np.array([1.0, 0.0, 0.5]); done = np.array([1, 0, 0]); gamma = 0.99
next_q = np.array([[0.2, 0.9], [1.0, 0.3], [0.4, 0.7]])   # Q_target(s')
y = r + gamma * next_q.max(1) * (1 - done)
print("Bellman 目标 y =", y.round(3), " (终止态那条 = r 本身)")
print("✅ y = r + γ·maxQ(s')·(1-done):终止态无未来,非终止态折现未来最优")
