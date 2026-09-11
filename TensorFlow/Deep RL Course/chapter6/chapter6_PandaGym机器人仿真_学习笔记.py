"""
================================================================================
 Deep RL Course · Chapter 6 · 机器人仿真 Panda-Gym（学习笔记 · VecNormalize numpy 可跑）
================================================================================
 一句话：用 Panda-Gym 机械臂仿真训 RL(推/够/抓);★机器人 RL 常靠 VecNormalize 归一化观测/奖励才稳。
 本章讲(纯 numpy 手写 VecNormalize 的"滑动均值/方差"归一化,无需 gym)：
   ① 环境:panda_gym 的 PandaReach 等,观测=关节/目标位置,动作=末端位移。
   ② VecNormalize:用滑动统计把观测归一化到均值0方差1 → 训练更稳更快。
   ③ 用 A2C/PPO 训练 + 评估 + 推 Hub。真实见 train_real(🔴 需 panda_gym+SB3)。
 要点：观测量纲差异大时不归一化会很难训;VecNormalize 是机器人 RL 的常用标配。
 跑：python3 chapter6_PandaGym机器人仿真_学习笔记.py   （滑动归一化纯 numpy 真跑)
================================================================================
"""
import numpy as np


class RunningNorm:
    """滑动均值/方差归一化(VecNormalize 的核心):在线更新统计,再归一化。"""
    def __init__(self, dim):
        self.mean = np.zeros(dim); self.var = np.ones(dim); self.count = 1e-4

    def update(self, x):                              # x: [batch, dim]
        bmean, bvar, bn = x.mean(0), x.var(0), x.shape[0]
        delta = bmean - self.mean
        tot = self.count + bn
        self.mean += delta * bn / tot
        self.var = (self.var * self.count + bvar * bn + delta ** 2 * self.count * bn / tot) / tot
        self.count = tot

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


def main():
    rng = np.random.default_rng(0)
    rn = RunningNorm(3)
    data = rng.normal(loc=[10, -5, 100], scale=[2, 1, 50], size=(2000, 3))   # 量纲差异大
    rn.update(data)
    z = rn.normalize(data)
    assert np.allclose(z.mean(0), 0, atol=0.1) and np.allclose(z.std(0), 1, atol=0.1)
    print(f"✅ Ch6 跑通：VecNormalize 后 均值≈{z.mean(0).round(2)} 方差≈{z.var(0).round(2)}(归一到~N(0,1),训练更稳)")


if __name__ == "__main__":
    main()
