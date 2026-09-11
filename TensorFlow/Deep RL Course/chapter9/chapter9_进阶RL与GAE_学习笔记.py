"""
================================================================================
 Deep RL Course · Chapter 9 · 进阶 RL 与 GAE（学习笔记 · GAE numpy 可跑）
================================================================================
 一句话：大规模 RL 靠高吞吐采样框架(Sample Factory 训 ViZDoom);优势估计常用 GAE 平衡偏差/方差。
 本章讲(纯 numpy 实现 GAE 广义优势估计,无需框架)：
   ① 高吞吐:异步 + 向量化环境,把采样吞吐拉满(Sample Factory)。
   ② GAE:用 λ 平滑多步 TD 误差 δ_t=r_t+γV(s_{t+1})-V(s_t) → 优势估计(偏差/方差折中)。
   ③ λ=0 退化为单步 TD(低方差高偏差),λ=1 近似蒙特卡洛(高方差低偏差)。
 要点：GAE 是 PPO/A2C 的标配优势估计;真实训练见 train_real(🔴 需 sample-factory/vizdoom)。
 跑：python3 chapter9_进阶RL与GAE_学习笔记.py   （GAE 纯 numpy 真跑)
================================================================================
"""
import numpy as np


def gae(rewards, values, gamma=0.99, lam=0.95):
    """广义优势估计:A_t = Σ (γλ)^l · δ_{t+l}。values 比 rewards 多一个(末态)。"""
    adv = np.zeros(len(rewards))
    g = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        g = delta + gamma * lam * g
        adv[t] = g
    return adv


def main():
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.5, 0.5, 0.5, 0.0])           # V(s0..s3)
    a_lam0 = gae(rewards, values, lam=0.0)            # 单步 TD
    a_lam1 = gae(rewards, values, lam=1.0)            # 近蒙特卡洛
    assert a_lam1[0] > a_lam0[0]                      # λ=1 累积更多步,优势更大
    print(f"✅ Ch9 跑通：GAE λ=0→{a_lam0.round(2).tolist()} vs λ=1→{a_lam1.round(2).tolist()}(λ 调偏差/方差)")


if __name__ == "__main__":
    main()
