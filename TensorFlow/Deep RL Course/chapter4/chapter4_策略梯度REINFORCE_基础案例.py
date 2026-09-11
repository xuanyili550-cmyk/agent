"""
 Deep RL Course · Ch4 · 基础案例：softmax 策略与 ∇logπ(纯 numpy,可跑)
 跑：python3 本文件
"""
import numpy as np
def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()
theta = np.array([0.0, 1.0]); pi = softmax(theta)
a = 1
grad_logpi = -pi.copy(); grad_logpi[a] += 1          # onehot(a) - π
print("策略 π =", pi.round(3), " 对动作", a, "的 ∇logπ =", grad_logpi.round(3))
print("✅ 回报高时 θ += α·G·∇logπ → 该动作概率上升")
