"""
 Deep RL Course · Ch9 · 基础案例：GAE 优势估计(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np
def gae(r, v, gamma=0.99, lam=0.95):
    adv, g = np.zeros(len(r)), 0.0
    for t in reversed(range(len(r))):
        delta = r[t] + gamma * v[t+1] - v[t]
        g = delta + gamma * lam * g; adv[t] = g
    return adv
print("GAE 优势:", gae(np.array([1.0,1,1]), np.array([0.5,0.5,0.5,0])).round(3))
print("✅ GAE 用 λ 平滑多步 TD 误差,平衡优势估计的偏差与方差")
