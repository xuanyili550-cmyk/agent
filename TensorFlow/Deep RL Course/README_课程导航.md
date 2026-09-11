# Deep RL Course · 课程导航

> 每个文件顶部有完整「学习笔记描述」。多为 HF 官方教程的 notebook 摘录/转写，
> 以**读懂原理**为主，不少需特定环境(GPU/gym/Unity/Blender 等)，不保证本机直接跑。

| 章 | 文件 | 一句话 |
|:--:|------|--------|
| 1 | `Deep Reinforcement Learning Agent.py` | RL = 智能体在环境里试错、拿奖励、学策略；本章训一个 LunarLander 并推到 HF Hub。 |
| 2 | `Q-Learning.py` | 最经典的 RL 算法——用一张 Q 表记录"每个状态下每个动作的价值"，迭代更新到收敛。 |
| 3 | `Deep Q-Learning.py` | 状态太多(如像素画面)Q 表装不下→用神经网络逼近 Q 函数，这就是 DQN。 |
| 4 | `Hands on.py` | 不学价值、直接学策略——策略梯度让"拿到高回报的动作"概率变大(REINFORCE)。 |
| 5 | `Unity ML-Agents.py` | 在 Unity 3D 游戏环境里训 RL 智能体(SnowballTarget/Pyramids)。 |
| 6 | `using Robotics Simulations with Panda-Gym.py` | 用 Panda-Gym 机械臂仿真训 RL——推/够/抓，向真实机器人过渡。 |
| 7 | `Multi-Agents systems.py` | 多个智能体在同一环境里协作/竞争——集中式 vs 分散式训练，及非平稳难题。 |
| 8 | `PPO from scratch.py` | PPO 是最常用的策略优化算法——用"裁剪(clip)"限制每步更新幅度，稳定又好用。 |
| 9 | `advanced Deep Reinforcement Learning.py` | 用高吞吐框架 Sample Factory 训复杂 3D 任务(ViZDoom)。 |
| 10 | `Godot RL Agents.py` | 开源 Godot 游戏引擎 + RL，给游戏 NPC 学复杂行为(2D/3D)。 |
