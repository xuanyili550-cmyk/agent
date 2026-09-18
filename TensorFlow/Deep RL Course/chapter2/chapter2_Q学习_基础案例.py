"""
 Deep RL Course · Ch2 · 基础案例：表格 Q 学习在走廊环境训练到收敛(纯 numpy,机制真实,可跑)
 Q学习 = 无模型、off-policy 的 TD 控制:Q(s,a) ← Q + α[r + γ·maxQ(s') − Q(s,a)]。
 环境是一维走廊:状态 0..N-1,最右格是目标(+1 并终止),动作只有左/右。我们真的跑很多
 回合、ε-贪婪探索,看 Q 表收敛、贪婪策略变成"一路向右",回合步数下降到最优。
 跑：python3 本文件
"""
import numpy as np

N = 6              # 走廊长度,状态 0..5,状态 5 是目标
GOAL = N - 1


def step(s, a):
    """环境转移:a=0 左,a=1 右。到达目标给 +1 并终止,其余每步 0(靠折扣促使尽快到达)。"""
    s2 = max(0, s - 1) if a == 0 else min(GOAL, s + 1)
    reward = 1.0 if s2 == GOAL else 0.0
    done = s2 == GOAL
    return s2, reward, done


def train(episodes=500, alpha=0.5, gamma=0.9, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N, 2))
    steps_hist = []
    for ep in range(episodes):
        s = 0
        eps = max(0.05, 1.0 - ep / 200)              # ε 逐步衰减:先探索后利用
        for t in range(100):
            a = int(rng.integers(2)) if rng.random() < eps else int(Q[s].argmax())
            s2, r, done = step(s, a)
            # off-policy TD:自举目标用 s' 的"最大" Q(贪婪/目标策略),与实际下一步动作无关
            Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s, a])
            s = s2
            if done:
                steps_hist.append(t + 1)
                break
    return Q, steps_hist


if __name__ == "__main__":
    Q, hist = train()
    greedy = Q.argmax(1)
    print("学到的 Q 表(行=状态,列=[左,右]):\n", Q.round(3))
    print("贪婪策略(0左/1右):", greedy[:GOAL].tolist(), " 首回合步数", hist[0])
    # 用纯贪婪策略(无探索)评估真实步数:应等于走廊长度=最优
    s, gsteps = 0, 0
    while s != GOAL and gsteps < 100:
        s, _, _ = step(s, int(Q[s].argmax()))
        gsteps += 1
    assert all(greedy[s] == 1 for s in range(GOAL)), greedy
    assert gsteps == GOAL, gsteps
    print(f"✅ Q 学习收敛:所有状态贪婪动作=向右,纯贪婪回合步数降到最优 {GOAL} 步")
    print("面试Q:Q学习为何是off-policy? A:更新目标用 maxQ(s')(目标=贪婪策略),行为却可用ε-贪婪探索,行为≠目标策略。")
