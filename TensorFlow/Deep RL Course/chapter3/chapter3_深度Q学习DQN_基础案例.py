"""
 Deep RL Course · Ch3 · 基础案例：DQN 三大机制的 numpy 实现(半梯度TD+目标网络+经验回放,可跑)
 DQN = 用可微函数近似 Q + 三个稳定训练的关键机制。本例把神经网络换成线性近似器
 Q(s,a)=w_a·φ(s)(机制完全一样,只是更小更透明),在链式 MDP 上真的做经验回放采样、
 用"冻结的目标网络"算 Bellman 目标 y、按半梯度更新,看 TD 损失下降、贪婪策略变正确。
 跑：python3 本文件
"""
import numpy as np

N = 5
GOAL = N - 1
gamma = 0.9


def phi(s):
    """状态特征(近似器输入):归一化位置 + 偏置。DQN 里这一步换成 CNN/MLP,机制不变。"""
    return np.array([s / (N - 1), 1.0])


def step(s, a):
    s2 = max(0, s - 1) if a == 0 else min(GOAL, s + 1)
    return s2, (1.0 if s2 == GOAL else 0.0), s2 == GOAL


def q_all(W, s):
    """两个动作的 Q 值向量:Q(s,·)=W·φ(s),W 形状 (动作数, 特征数)。"""
    return W @ phi(s)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W = np.zeros((2, 2))                  # 在线网络参数
    W_target = W.copy()                   # 目标网络:冻结,定期同步(打断自举反馈回路)
    alpha = 0.1

    replay = []                           # 经验回放缓冲:打散样本相关性
    for _ in range(2000):                 # 先用随机策略填经验池
        s = int(rng.integers(N))
        a = int(rng.integers(2))
        s2, r, done = step(s, a)
        replay.append((s, a, r, s2, done))

    losses = []
    for it in range(4000):
        s, a, r, s2, done = replay[int(rng.integers(len(replay)))]   # 均匀采样一条经验
        y = r + gamma * q_all(W_target, s2).max() * (not done)       # 目标网络算 Bellman 目标
        td = y - q_all(W, s)[a]
        losses.append(td * td)
        W[a] += alpha * td * phi(s)       # 半梯度:只对预测 Q 求梯度,不动目标 y
        if it % 500 == 499:
            W_target = W.copy()           # 定期把在线网络拷进目标网络

    early, late = np.mean(losses[:200]), np.mean(losses[-200:])
    greedy = [int(q_all(W, s).argmax()) for s in range(GOAL)]
    print(f"TD 损失:前200步均值 {early:.4f} → 后200步均值 {late:.4f}")
    print("各状态贪婪动作(0左/1右):", greedy)
    assert late < early * 0.5, (early, late)
    assert all(g == 1 for g in greedy), greedy
    print("✅ 半梯度TD+目标网络+经验回放收敛:TD 损失显著下降、贪婪策略一路向右")
    print("面试Q:DQN 为何要目标网络? A:若目标 y 用不断变的在线网络算就是追自己尾巴、易发散;冻结目标网络让 y 短期稳定。")
