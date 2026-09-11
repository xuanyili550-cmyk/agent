"""
================================================================================
 Deep RL Course · Chapter 2 · 表格 Q-Learning（学习笔记 · 完整训练 numpy 可跑）
================================================================================
 一句话：最经典 RL 算法——用一张 Q 表记"每个状态每个动作的价值",靠时序差分迭代到收敛。
 本章讲(纯 numpy 在一维走廊环境上完整训练 Q 表,真跑收敛)：
   ① 环境:状态 0..N-1,动作 左/右,走到最右端 goal 得 +1。
   ② ε-贪心 + 平局随机:探索 vs 利用;★Q 全相等(如初始全 0)时也随机,否则会"恒选左"陷在原地。
   ③ TD 更新:Q(s,a) ← Q + α[r + γ·maxQ(s',·) − Q(s,a)]。
 要点：Q-Learning 是"离策略 + 值迭代";训练后贪心策略应一路向右到达目标。
 跑：python3 chapter2_Q学习_学习笔记.py   （完整训练纯 numpy 真跑)
================================================================================
"""
import numpy as np


def train(N=5, episodes=3000, alpha=0.2, gamma=0.95, eps=0.2, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N, 2))                              # 动作 0=左 1=右
    for _ in range(episodes):
        s = 0
        for _ in range(50):
            if rng.random() < eps or Q[s, 0] == Q[s, 1]:   # ε-贪心;平局也随机(破"恒选左")
                a = int(rng.integers(2))
            else:
                a = int(np.argmax(Q[s]))
            s2 = max(0, s - 1) if a == 0 else min(N - 1, s + 1)
            r = 1.0 if s2 == N - 1 else 0.0
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])   # TD 更新
            s = s2
            if s == N - 1:
                break
    return Q


def main():
    Q = train()
    policy = Q.argmax(axis=1)                         # 每个状态的贪心动作
    assert (policy[:-1] == 1).all()                   # 除目标外都应学会"向右"
    print(f"✅ Ch2 跑通：Q-Learning 收敛,贪心策略={policy.tolist()}(1=右,一路向右到目标)")
    # 面试：Q Q-Learning 更新式? A Q←Q+α[r+γmaxQ(s')-Q]; Q 离策略? A 更新用 maxQ(贪心)而非实际所采动作。


if __name__ == "__main__":
    main()
