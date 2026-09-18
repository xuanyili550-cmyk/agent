"""
 Deep RL Course · Ch10 · 基础案例：Godot 风格 env-agent 循环 + 表格 Q-learning(纯 numpy,机制真实,可跑)
 Godot RL 里游戏引擎当环境:每帧把传感器观测发给 agent,agent 回一个动作,引擎推进物理并给奖励。
 引擎跑不动,这里用纯 numpy 造一个走廊 env(严格 Gym 接口 reset/step),真跑表格 Q-learning 若干 episode,
 看"到达目标所需步数"随训练下降、学到的贪心策略变成一路向右 —— 这就是 RL 训练内核。
 诚实标注:真实 GodotRL 用 TCP/共享内存把 Godot 引擎的观测流式传给 Python;此处 env 是简化替身。
 跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(0)


class Corridor:
    """1D 走廊:n 个格子,从 0 出发,到最右端 n-1 得奖励并结束。仿 Gym 的 reset/step 接口。"""

    def __init__(self, n=6):
        self.n = n

    def reset(self):
        self.pos = 0
        return self.pos                              # 观测=当前格子编号

    def step(self, action):                          # action: 0=左, 1=右
        self.pos = np.clip(self.pos + (1 if action == 1 else -1), 0, self.n - 1)
        done = self.pos == self.n - 1
        reward = 1.0 if done else -0.01              # 到达给 +1,每步小惩罚 → 鼓励尽快到达
        return self.pos, reward, done


if __name__ == "__main__":
    env = Corridor(n=6)
    Q = np.zeros((env.n, 2))                          # Q 表:每个状态×每个动作的价值
    alpha, gamma, eps = 0.5, 0.95, 0.2
    steps_per_ep = []

    for ep in range(200):
        s, done, steps = env.reset(), False, 0
        while not done and steps < 100:
            a = rng.integers(2) if rng.random() < eps else int(np.argmax(Q[s]))   # ε-greedy
            s2, r, done = env.step(a)
            target = r + (0 if done else gamma * Q[s2].max())    # 贝尔曼目标:即时奖励+折扣未来最优
            Q[s, a] += alpha * (target - Q[s, a])                # 时序差分更新
            s = s2; steps += 1
        steps_per_ep.append(steps)

    early = np.mean(steps_per_ep[:20])
    late = np.mean(steps_per_ep[-20:])
    greedy_policy = Q.argmax(1)[:-1]                  # 末状态是终点,不看
    print(f"前 20 个 episode 平均步数 {early:.1f} → 后 20 个 {late:.1f}(含探索噪声,故略高于最优)")
    print("学到的贪心策略(0左/1右):", greedy_policy.tolist())
    assert late < early, "训练后到达目标应更快"
    assert np.all(greedy_policy == 1), "最优策略应是每个非终点状态都向右"

    # 用纯贪心(关掉探索)评测一次,才能看出学到的策略本身是最优的 → 恰好用 n-1 步到达
    s, done, eval_steps = env.reset(), False, 0
    while not done and eval_steps < 100:
        s, _, done = env.step(int(np.argmax(Q[s]))); eval_steps += 1
    print(f"关掉探索后贪心评测:{eval_steps} 步到达(最短 {env.n - 1} 步)")
    assert eval_steps == env.n - 1, "学到的贪心策略应以最短步数到达"
    print("✅ Gym 风格 env-agent 循环 + Q-learning 跑通:步数下降、贪心策略最短到达 → RL 训练内核")
    # 面试Q：GodotRL 这类"引擎当环境"的框架,Python 侧要做什么? A：只需实现标准 RL 接口(reset/step 或向量化并行环境),
    #        通过桥接层拿引擎的观测/奖励;算法(Q-learning/PPO 等)与引擎解耦,换环境不用改学习代码。
