"""
 Deep RL Course · Ch7 · 基础案例：独立 Q-learning 玩协作博弈(纯 numpy,机制真实,可跑)
 多智能体核心难点=非平稳:每个 agent 眼里的"环境"含了对方策略,对方在学 → 环境在变。
 这里两个 agent 各自维护独立 Q 表(Independent Q-Learning),在协作博弈里反复对弈、各自 ε-greedy 学习,
 看它们能否从随机起步收敛到共赢均衡 (1,1)。回报由"动作组合"决定,正是 MARL 的联合动作耦合。
 跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(0)

# 纯协作博弈收益矩阵:只有双方都选动作1 才拿到奖励2,其余组合都是0 → (1,1) 是唯一有利均衡。
# (注:若把 (0,0) 也设成正收益,独立学习者常会卡在次优均衡=经典的 miscoordination 陷阱,见文末Q&A。)
PAYOFF = np.array([[0.0, 0.0],
                   [0.0, 2.0]])


def eps_greedy(q, eps):
    """ε-greedy:以 eps 概率探索,否则选当前 Q 最大的动作。探索是跳出次优均衡的关键。"""
    return rng.integers(2) if rng.random() < eps else int(np.argmax(q))


if __name__ == "__main__":
    qA, qB = np.zeros(2), np.zeros(2)          # 两个 agent 各自独立的 Q 值(无状态重复博弈,每动作一个 Q)
    alpha, eps = 0.1, 0.3
    joint_counts = np.zeros((2, 2))

    for it in range(3000):
        eps_t = max(0.02, eps * (1 - it / 3000))       # 探索率退火:先广探索后收敛
        aA, aB = eps_greedy(qA, eps_t), eps_greedy(qB, eps_t)
        r = PAYOFF[aA, aB]                              # 回报取决于联合动作 → 对 A 而言 B 的策略即环境的一部分
        qA[aA] += alpha * (r - qA[aA])                  # 各自只用自己观测到的回报更新(独立学习)
        qB[aB] += alpha * (r - qB[aB])
        if it >= 2800:                                  # 统计收敛后期的联合动作分布
            joint_counts[aA, aB] += 1

    greedy = (int(np.argmax(qA)), int(np.argmax(qB)))
    print("A 的 Q:", qA.round(2), " B 的 Q:", qB.round(2))
    print("收敛后期联合动作分布:\n", (joint_counts / joint_counts.sum()).round(2))
    print(f"两 agent 各自贪心选择 → {greedy}")
    assert greedy == (1, 1), "独立学习应收敛到共赢均衡 (1,1)"
    assert joint_counts[1, 1] == joint_counts.max(), "后期最常出现的联合动作应是 (1,1)"
    print("✅ 两个独立 Q 学习者从随机起步收敛到共赢均衡;回报由联合动作决定 → 各自面对的是非平稳环境")
    # 面试Q：多智能体 RL 为何比单智能体难? A：环境非平稳——每个 agent 的最优策略依赖他人策略,
    #        而他人也在学习,导致对单个 agent 而言转移/回报分布随时间漂移,单体收敛保证失效(需 CTDE、对手建模等)。
    #        实测:若存在竞争的次优均衡,独立学习者常因探索不足卡在次优均衡(miscoordination),这正是非平稳带来的坑。
