"""
================================================================================
 Deep RL Course · Chapter 1 · 深度强化学习入门（学习笔记 · 折扣回报 numpy 可跑）
================================================================================
 一句话：RL = 智能体在环境里试错、拿奖励、学策略最大化"折扣累计回报"。
 本章讲(纯 numpy 演示 MDP 基本量 + 随机策略 rollout,无需 gym)：
   ① 基本量:状态 s / 动作 a / 奖励 r / 策略 π / 折扣回报 G=Σγᵗrₜ。
   ② γ(折扣因子):越接近 1 越看重长远,越小越看重眼前。
   ③ rollout:按策略与环境交互一条轨迹,算它的回报。真实训练见 train_real(🔴 gym+SB3)。
 要点：RL 的目标是最大化"期望折扣回报",不是单步奖励。
 跑：python3 chapter1_深度强化学习入门_学习笔记.py   （折扣回报纯 numpy 真跑)
================================================================================
"""
import numpy as np


def discounted_return(rewards, gamma=0.99):
    """G = Σ γᵗ·rₜ。从后往前累加更稳更快。"""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]


def train_real():   # 🔴 需 gym + stable-baselines3,默认不调用
    import gymnasium as gym
    from stable_baselines3 import PPO
    env = gym.make("LunarLander-v2")
    return PPO("MlpPolicy", env).learn(total_timesteps=100_000)


def main():
    rewards = [0, 0, 0, 1]                          # 只有最后一步拿到奖励 1
    g99 = discounted_return(rewards, 0.99)
    g50 = discounted_return(rewards, 0.5)
    assert abs(g99[0] - 0.99 ** 3) < 1e-9           # 起点回报 = γ³·1
    assert g50[0] < g99[0]                          # γ 小 → 远期奖励被压得更狠
    print(f"✅ Ch1 跑通：折扣回报 γ=0.99 起点G={g99[0]:.3f} vs γ=0.5 起点G={g50[0]:.3f}(γ小更短视)")
    # 面试：Q RL 目标? A 最大化期望折扣回报; Q γ 作用? A 权衡眼前vs长远,保证无穷回报收敛。


if __name__ == "__main__":
    main()
