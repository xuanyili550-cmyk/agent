"""
================================================================================
 Deep RL Course · Chapter 4 · 策略梯度(REINFORCE)（学习笔记 · 策略更新 numpy 可跑）
================================================================================
 一句话：不学价值、直接学策略——让"拿到高回报的动作"概率变大(REINFORCE)。
 本章讲(纯 numpy 在一个双臂问题上训 softmax 策略,真跑能学会选更优臂)：
   ① 策略 π(a)=softmax(θ);按概率采样动作。
   ② REINFORCE 更新:θ ← θ + α·G·∇logπ(a)(回报高就强化该动作)。
   ③ softmax 策略的 ∇logπ 有简洁形式(one_hot(a) − π)。值方法 vs 策略方法的分野。
 要点：策略梯度直接优化策略,适合连续/随机动作;高方差,常配基线(如优势)降方差。
 跑：python3 chapter4_策略梯度REINFORCE_学习笔记.py   （策略训练纯 numpy 真跑)
================================================================================
"""
import numpy as np


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def main():
    rng = np.random.default_rng(0)
    theta = np.zeros(2)                               # 两个动作的偏好
    true_reward = np.array([0.2, 0.8])                # 动作1 更优(期望奖励高)
    alpha = 0.1
    for _ in range(2000):
        pi = softmax(theta)
        a = rng.choice(2, p=pi)                       # 按策略采样
        G = 1.0 if rng.random() < true_reward[a] else 0.0   # 伯努利奖励
        grad_logpi = -pi.copy(); grad_logpi[a] += 1   # ∇logπ(a) = onehot(a) − π
        theta += alpha * G * grad_logpi               # REINFORCE 更新
    pi = softmax(theta)
    assert pi[1] > 0.7                                # 学会更常选更优的动作1
    print(f"✅ Ch4 跑通：REINFORCE 训后策略 π={pi.round(2)}(学会偏向更优动作1)")
    # 面试：Q 值方法 vs 策略方法? A 前者学Q选动作(DQN),后者直接学π(PG); Q REINFORCE 高方差怎么办? A 减基线/用优势。


if __name__ == "__main__":
    main()
