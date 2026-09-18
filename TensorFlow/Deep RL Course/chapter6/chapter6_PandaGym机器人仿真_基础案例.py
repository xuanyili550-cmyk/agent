"""
 Deep RL Course · Ch6 · 基础案例：goal-conditioned 机器人仿真 + VecNormalize(纯 numpy,机制真实,可跑)
 panda-gym 用 pybullet 仿真机械臂,观测是 dict{观测, 已达目标 achieved, 期望目标 desired},奖励稀疏。
 引擎装不上,这里用纯 numpy 造一个 2D 末端执行器"够物"env:同样的 dict 观测 + 稀疏奖励 + 比例控制器,
 并真跑 VecNormalize 的在线均值/方差(Welford 增量法)—— 机器人各维量纲悬殊,不归一化极难训。
 诚实标注:这是概念替身,真实 panda-gym 靠 pybullet 物理引擎;此处运动学是简化的一阶积分。
 跑：python3 本文件
"""
import numpy as np

rng = np.random.default_rng(0)


class ReachEnv:
    """2D 够物任务:末端执行器 ee 要移动到随机目标 goal。仿 panda-gym 的 dict 观测 + 稀疏奖励。"""

    def __init__(self, threshold=0.05):
        self.threshold = threshold      # 到目标多近算成功

    def reset(self):
        self.ee = rng.uniform(-1, 1, size=2)
        self.goal = rng.uniform(-1, 1, size=2)
        return self._obs()

    def _obs(self):
        # panda-gym 风格:observation=末端状态, achieved_goal=当前已达位置, desired_goal=期望位置
        return {"observation": self.ee.copy(), "achieved_goal": self.ee.copy(), "desired_goal": self.goal.copy()}

    def step(self, action):
        self.ee = np.clip(self.ee + 0.1 * action, -1, 1)   # 一阶运动学:动作是速度增量
        dist = np.linalg.norm(self.ee - self.goal)
        reward = 0.0 if dist < self.threshold else -1.0    # 稀疏奖励:没到就 -1(panda-gym 默认)
        return self._obs(), reward, dist < self.threshold


class RunningNorm:
    """VecNormalize 的核心:在线统计流式观测的均值/方差(Welford),不用一次性存全部数据。"""

    def __init__(self, dim):
        self.mean, self.M2, self.count = np.zeros(dim), np.zeros(dim), 0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count                    # 增量更新均值
        self.M2 += delta * (x - self.mean)                 # 增量更新方差累积量

    def std(self):
        return np.sqrt(self.M2 / max(self.count, 1)) + 1e-8


if __name__ == "__main__":
    env, norm, all_obs = ReachEnv(), RunningNorm(2), []
    for ep in range(3):                                    # 跑几个 episode,比例控制器一定能够到
        obs, done, steps = env.reset(), False, 0
        while not done and steps < 200:
            action = obs["desired_goal"] - obs["achieved_goal"]   # 比例控制:朝目标方向走
            obs, r, done = env.step(action)
            norm.update(obs["observation"]); all_obs.append(obs["observation"]); steps += 1
        print(f"episode{ep}: {'成功够到' if done else '超时'},用 {steps} 步")
        assert done, "比例控制器应能够到目标"

    batch = np.array(all_obs)                              # 用离线批统计验证在线统计是否算对
    assert np.allclose(norm.mean, batch.mean(0), atol=1e-6), "Welford 均值应等于批均值"
    assert np.allclose(norm.std(), batch.std(0), atol=1e-6), "Welford 方差应等于批方差"
    print("在线均值:", norm.mean.round(3), " 批均值:", batch.mean(0).round(3), " → 一致")
    print("✅ dict 观测+稀疏奖励+比例控制跑通;VecNormalize 在线统计与批统计一致 → 机器人多维量纲可稳定归一化")
    # 面试Q：panda-gym 为何用稀疏奖励+VecNormalize? A：稀疏奖励更接近真实成败信号(常配 HER 回放),
    #        但各观测维度量纲差异大,VecNormalize 用运行均值/方差归一化才能让梯度尺度一致、训练不发散。
