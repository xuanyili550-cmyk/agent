"""
================================================================================
 Deep RL Course · Chapter 5 · Unity ML-Agents（学习笔记 · 策略评估 numpy 可跑）
================================================================================
 一句话：在 Unity 3D 游戏环境里训 RL 智能体(SnowballTarget/Pyramids),工具链=Unity 环境 + Python 训练端。
 本章讲：
   ① ML-Agents:Unity 侧定义环境(观测/动作/奖励),Python 侧用 PPO 训练。
   ② 观测→策略→动作→奖励的闭环(本文件用 numpy 模拟"评估一个策略的平均回报")。
   ③ 训练结果推 Hub 可视化回放。真实训练见 train_real(🔴 需 Unity + mlagents)。
 要点：3D 仿真让 RL 逼近真实游戏/机器人;训练端算法仍是 PPO 那套。
 跑：python3 chapter5_UnityMLAgents_学习笔记.py   （策略评估纯 numpy 真跑)
================================================================================
"""
import numpy as np


def evaluate_policy(policy, n_episodes=100, seed=0):
    """模拟环境:动作对=更高命中率。评估策略平均回报(numpy 代替 Unity 环境)。"""
    rng = np.random.default_rng(seed)
    hit_rate = {0: 0.3, 1: 0.7}                      # 动作1更好
    returns = []
    for _ in range(n_episodes):
        a = policy(rng)
        returns.append(1.0 if rng.random() < hit_rate[a] else 0.0)
    return float(np.mean(returns))


def train_real():   # 🔴 需 Unity + mlagents,默认不调用
    # mlagents-learn config.yaml --run-id=SnowballTarget  (命令行训练)
    return "见 ML-Agents 官方:Unity 建环境 + mlagents-learn PPO 训练"


def main():
    good = evaluate_policy(lambda rng: 1)             # 总选好动作
    bad = evaluate_policy(lambda rng: 0)
    assert good > bad
    print(f"✅ Ch5 跑通：策略评估 好策略平均回报={good:.2f} > 差策略={bad:.2f}(Unity 里靠 PPO 学到好策略)")


if __name__ == "__main__":
    main()
