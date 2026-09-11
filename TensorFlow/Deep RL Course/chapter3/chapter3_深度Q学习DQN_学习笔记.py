"""
================================================================================
 Deep RL Course · Chapter 3 · 深度 Q 学习(DQN)（学习笔记 · Bellman目标+回放 numpy 可跑）
================================================================================
 一句话：状态太多(如像素)Q 表装不下 → 用神经网络逼近 Q;靠经验回放+目标网络稳住训练。
 本章讲(纯 numpy 演示 DQN 的两个关键机制,无需 torch)：
   ① 经验回放(replay buffer):存 (s,a,r,s')转移,随机采样打破样本相关性。
   ② Bellman 目标:y = r + γ·maxQ_target(s')(终止态只有 r)。用目标网络算,防自举震荡。
   ③ 损失 = (Q(s,a) − y)²;真实网络见 build_dqn(🔴 torch)。
 要点：回放破相关性、目标网络稳目标,是 DQN 能收敛的两大关键(面试高频)。
 跑：python3 chapter3_深度Q学习DQN_学习笔记.py   （回放+Bellman目标纯 numpy 真跑)
================================================================================
"""
import numpy as np
from collections import deque


def bellman_target(r, next_q, done, gamma=0.99):
    """y = r + γ·maxQ(s')·(1-done)。终止态没有未来。"""
    return r + gamma * next_q.max(axis=1) * (1 - done)


def main():
    buf = deque(maxlen=1000)                          # 经验回放缓冲
    rng = np.random.default_rng(0)
    for _ in range(200):                              # 塞入一些转移
        buf.append((rng.standard_normal(4), rng.integers(2), rng.random(), rng.standard_normal(4), rng.integers(2)))
    batch = [buf[i] for i in rng.integers(0, len(buf), 32)]   # 随机采样一批(破相关性)
    r = np.array([b[2] for b in batch])
    done = np.array([b[4] for b in batch])
    next_q = rng.standard_normal((32, 2))             # 目标网络对 s' 的 Q(此处随机代替)
    y = bellman_target(r, next_q, done)
    assert y.shape == (32,) and np.all(y[done == 1] == r[done == 1])   # 终止态 y==r
    print(f"✅ Ch3 跑通：回放缓冲采样 32 条 → Bellman 目标 y(终止态 y==r 校验通过)")
    # 面试：Q DQN 两大技巧? A 经验回放(破相关)+目标网络(稳目标); Q 目标怎么算? A r+γmaxQ_target(s')。


if __name__ == "__main__":
    main()
