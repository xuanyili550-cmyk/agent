"""
================================================================================
 Deep RL Course · Chapter 10 · Godot RL Agents（学习笔记 · env-agent 循环 numpy 可跑）
================================================================================
 一句话：用开源 Godot 游戏引擎 + RL,给游戏 NPC 学复杂行为(2D/3D);Godot 侧建环境、Python 侧训练。
 本章讲(纯 numpy 模拟最小 env-agent 交互循环,无需 Godot)：
   ① Godot 引擎 ↔ Python RL 框架(SB3/CleanRL/Ray)的桥接;AI 传感器给观测。
   ② 交互循环:env.reset → while not done: action=policy(obs); obs,r,done=env.step(action)。
   ③ 支持记忆型 agent(LSTM/注意力)。真实见 train_real(🔴 需 godot-rl)。
 要点：所有 RL 训练内核都是这个"观测→动作→奖励→下一步"循环;Godot 只是提供环境。
 跑：python3 chapter10_GodotRL_学习笔记.py   （env-agent 循环纯 numpy 真跑)
================================================================================
"""
import numpy as np


class TinyEnv:
    """最小环境:到达 target 位置得 +1。模仿 Gym 的 reset/step 接口。"""
    def __init__(self, n=5):
        self.n = n

    def reset(self):
        self.pos = 0; return self.pos

    def step(self, action):                           # 0=左 1=右
        self.pos = max(0, self.pos - 1) if action == 0 else min(self.n - 1, self.pos + 1)
        done = self.pos == self.n - 1
        return self.pos, (1.0 if done else 0.0), done


def main():
    env = TinyEnv()
    obs = env.reset()
    total, done = 0.0, False
    for _ in range(20):                               # env-agent 交互循环
        action = 1                                    # 一个"总向右"的策略
        obs, r, done = env.step(action)
        total += r
        if done:
            break
    assert done and total == 1.0
    print(f"✅ Ch10 跑通：env-agent 循环(reset→step…)到达目标,回报={total}(Godot 只是换个更复杂的 env)")


if __name__ == "__main__":
    main()
