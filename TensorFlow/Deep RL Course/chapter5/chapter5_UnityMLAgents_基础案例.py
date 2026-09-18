"""
 Deep RL Course · Ch5 · 基础案例：Unity ML-Agents 交互范式的 numpy 概念替身(机制诚实标注,可跑)
 【诚实说明】真正的 Unity ML-Agents 依赖游戏引擎 + Python 侧经 gRPC 通信,本机无引擎跑不了。
 这里用纯 numpy 复刻其"环境-智能体"接口(reset()→观测;step(动作)→观测,奖励,是否终止)
 与训练闭环:一个 1D "走向目标"任务,策略是线性高斯,用策略梯度真的训练到能稳定到达目标。
 机制(观测/连续动作/奖励/回合/回报到达值/策略梯度)都真实,只是把 Unity 引擎换成几行 numpy 物理。
 跑：python3 本文件
"""
import numpy as np


class ReachEnv:
    """ML-Agents 风格环境:状态=当前位置,动作=位移,目标是原点。越靠近目标惩罚越小。"""

    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        self.x = self.rng.uniform(-1, 1)                  # 每回合随机初始位置
        self.t = 0
        return np.array([self.x])                         # 向量观测(ML-Agents 的 vector observation)

    def step(self, action):
        self.x += float(np.clip(action, -0.3, 0.3))       # 连续动作 = 位移(带上限)
        self.t += 1
        reward = -abs(self.x)                             # 离目标越远惩罚越大
        done = abs(self.x) < 0.05 or self.t >= 20         # 到达 or 超时终止
        return np.array([self.x]), reward, done


def policy(theta, obs, rng, sigma, explore=True):
    """线性高斯策略:均值 mean=θ·[obs,1],动作=mean+噪声。返回动作与均值(算 ∇logπ 用)。"""
    mean = theta[0] * obs[0] + theta[1]
    a = mean + (rng.normal(0, sigma) if explore else 0.0)
    return a, mean


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    env = ReachEnv(rng)
    theta = np.array([0.0, 0.0])
    alpha, sigma, gamma, baseline = 0.01, 0.2, 0.99, 0.0
    returns = []
    for ep in range(1500):
        obs, traj, rews, done = env.reset(), [], [], False
        while not done:
            a, mean = policy(theta, obs, rng, sigma)
            nobs, r, done = env.step(a)
            traj.append((obs[0], a, mean))
            rews.append(r)
            obs = nobs
        rtg, G = [], 0.0                                  # 计算每步的"回报到达值"(reward-to-go,降方差)
        for r in reversed(rews):
            G = r + gamma * G
            rtg.append(G)
        rtg = rtg[::-1]
        returns.append(sum(rews))
        baseline += 0.05 * (rtg[0] - baseline)            # 回报基线:优势=回报到达值−基线
        grad = np.zeros(2)                                # 攒整条回合的策略梯度,回合末更新一次
        for (x, a, mean), Gt in zip(traj, rtg):
            glog = (a - mean) / (sigma ** 2)              # 高斯策略 ∇logπ=(a−mean)/σ²·∂mean/∂θ
            grad += (Gt - baseline) * glog * np.array([x, 1.0])
        theta = theta + alpha * grad / len(traj)          # REINFORCE:优势加权梯度上升

    early, late = np.mean(returns[:100]), np.mean(returns[-100:])
    dists = []
    for _ in range(50):                                   # 确定性策略(无探索)评估终点距目标
        obs, done = env.reset(), False
        while not done:
            a, _ = policy(theta, obs, rng, sigma, explore=False)
            obs, _, done = env.step(a)
        dists.append(abs(obs[0]))
    print(f"回合回报:前100 {early:.3f} → 后100 {late:.3f}  (越接近0越好)")
    print(f"学到策略参数 θ={theta.round(3)}(θ0≈−1 表示按位置反向拉回),确定性评估平均终点距目标 {np.mean(dists):.3f}")
    assert late > early + 1.0, (early, late)
    assert np.mean(dists) < 0.1, np.mean(dists)
    print("✅ 用 numpy 复刻 ML-Agents 的交互接口+策略梯度,智能体学会稳定走向目标(真实需 Unity 引擎)")
    print("面试Q:ML-Agents 里 Python 与引擎怎么通信? A:引擎侧收集观测经 gRPC 发给 Python 训练器,训练器回传动作,解耦仿真与学习。")
