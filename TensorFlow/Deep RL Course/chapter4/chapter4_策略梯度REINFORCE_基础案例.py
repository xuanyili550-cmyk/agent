"""
 Deep RL Course · Ch4 · 基础案例：REINFORCE 在 softmax 多臂老虎机上真收敛(纯 numpy,机制真实,可跑)
 策略梯度 REINFORCE:参数化策略 π_θ(a),用采样回报 G 加权 ∇logπ 做梯度上升 → θ += α·G·∇logπ(a)。
 环境是多臂老虎机(每臂伯努利奖励),最优臂命中率最高。我们真的迭代若干轮,
 看 softmax 策略把概率质量挪到最优臂、平均回报上升;用基线(减移动均值)降方差。
 跑：python3 本文件
"""
import numpy as np


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = np.array([0.2, 0.5, 0.9, 0.3])        # 各臂真实命中率,臂 2 最优
    K = len(p)
    theta = np.zeros(K)                        # 策略参数(每臂一个偏好值)
    alpha = 0.1
    baseline = 0.0                             # 移动平均基线:降低梯度方差(不引入偏差)
    reward_hist = []

    for t in range(1, 3001):
        pi = softmax(theta)
        a = int(rng.choice(K, p=pi))           # 按当前策略采样动作
        r = 1.0 if rng.random() < p[a] else 0.0
        baseline += 0.01 * (r - baseline)      # 更新基线
        grad_logpi = -pi.copy()
        grad_logpi[a] += 1.0                   # ∇logπ(a) = onehot(a) − π
        theta += alpha * (r - baseline) * grad_logpi   # REINFORCE:优势加权梯度上升
        reward_hist.append(r)

    pi = softmax(theta)
    early, late = np.mean(reward_hist[:300]), np.mean(reward_hist[-300:])
    print("最终策略概率:", pi.round(3), " (真实最优臂 = 2)")
    print(f"平均回报:前300步 {early:.3f} → 后300步 {late:.3f}  (最优臂命中率 {p.max()})")
    assert pi.argmax() == 2 and pi[2] > 0.8, pi
    assert late > early, (early, late)
    print("✅ REINFORCE 收敛:概率质量集中到最优臂,平均回报升到接近最优臂命中率")
    print("面试Q:为何要基线? A:θ+=α(G−b)∇logπ,减去与动作无关的基线 b 不改变期望梯度但显著降方差,训练更稳。")
