"""
 Deep RL Course · Ch1 · 基础案例：折扣累计回报 + 蒙特卡洛估值收敛(纯 numpy,机制真实,可跑)
 RL 要最大化的是"折扣累计回报"G_t = r_t + γ·r_{t+1} + γ²·r_{t+2}+…;它是随机的,
 我们真正关心的是它的"期望"(状态价值 V)。本例用一个可解析求解的小 MDP,让蒙特卡洛
 采样的平均回报真的收敛到解析期望,直观看到"回报是随机量、价值是它的期望"。
 跑：python3 本文件
"""
import numpy as np


def discounted_return(rewards, gamma):
    """从一条回合的奖励序列算逐步折扣回报:G_t = r_t + γ·G_{t+1}(倒着累加,O(n))。"""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]


def run_episode(rng, p_terminate):
    """一条回合:每步 +1 奖励,然后以 p 概率终止。回合长度随机 → 回报是随机量。"""
    rewards = []
    while True:
        rewards.append(1.0)
        if rng.random() < p_terminate:
            break
    return rewards


if __name__ == "__main__":
    gamma, p = 0.9, 0.2
    rng = np.random.default_rng(0)
    # 解析期望:存活到第 t 步概率 (1-p)^t,故 E[G]=Σ γ^t (1-p)^t = 1/(1-γ(1-p))
    analytic = 1.0 / (1.0 - gamma * (1.0 - p))

    demo = discounted_return([1, 0, 0, 1], gamma)
    print("逐步折扣回报 [1,0,0,1] (γ=0.9):", [round(x, 3) for x in demo])

    total = 0.0
    for n in range(1, 20001):
        G0 = discounted_return(run_episode(rng, p), gamma)[0]  # 该回合起点回报
        total += G0
        if n in (10, 100, 1000, 20000):
            print(f"  蒙特卡洛 {n:>5d} 回合平均回报 = {total / n:.3f}  (解析期望 {analytic:.3f})")

    mc = total / 20000
    assert abs(mc - analytic) < 0.05, (mc, analytic)
    print(f"✅ 平均回报 {mc:.3f} 收敛到解析期望 {analytic:.3f}:价值 V=E[G],RL 就是最大化它")
    print("面试Q:回报和价值区别? A:回报 G 是单条轨迹的随机折扣奖励和,价值是回报的期望(对策略与环境随机性求均)。")
