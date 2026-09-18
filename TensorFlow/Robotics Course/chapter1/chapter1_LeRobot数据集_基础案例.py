"""
 Robotics Course · Ch1 · 基础案例：LeRobotDataset 的 delta_timestamps 时序窗口切分(纯 numpy,机制真实,可跑)
 机器人策略要"看历史、预测未来":一条训练样本不是单帧,而是围绕某个时刻按 delta_timestamps
 采样的一小段时间窗——过去几帧观测(给上下文) + 未来几帧动作(要预测的 action chunk)。
 这里从零实现这套"按秒偏移 → 折算成帧下标 → 越界补齐"的取样逻辑,和 LeRobotDataset 内部一致。
 跑：python3 本文件
"""
import numpy as np


class MiniLeRobotDataset:
    """极简数据集:一段 episode = 定长时序;__getitem__ 按 delta_timestamps 切出时间窗样本。"""

    def __init__(self, n_frames=15, fps=10):
        self.fps = fps                                   # 采样频率:秒偏移 × fps = 帧偏移
        t = np.arange(n_frames)
        # 真实里 obs 是图像/关节角、action 是电机指令;这里用可核对的数值序列代替
        self.obs = np.stack([np.sin(t / 3.0), np.cos(t / 3.0)], axis=1)   # (T,2) 观测
        self.action = (t * 0.1).reshape(-1, 1)                            # (T,1) 动作

    def _indices(self, idx, deltas):
        """秒偏移 → 帧下标,并 clip 到合法范围(边界处复制端点帧 = LeRobot 的 padding 策略)。"""
        raw = np.array([idx + round(d * self.fps) for d in deltas])       # 折算成帧下标
        clipped = np.clip(raw, 0, len(self.obs) - 1)                      # 越界→夹到端点
        pad_mask = raw != clipped                                         # 标记哪些是补出来的
        return clipped, pad_mask

    def __getitem__(self, idx, obs_deltas=(-0.2, -0.1, 0.0), act_deltas=(0.0, 0.1, 0.2)):
        """一条样本 = 过去 obs 窗口 + 未来 action 窗口(action chunking)。"""
        oi, o_pad = self._indices(idx, obs_deltas)
        ai, a_pad = self._indices(idx, act_deltas)
        return {
            "observation": self.obs[oi], "obs_is_pad": o_pad,            # (3,2) 含过去 200/100/0ms
            "action": self.action[ai], "action_is_pad": a_pad,          # (3,1) 未来 0/100/200ms
        }


if __name__ == "__main__":
    ds = MiniLeRobotDataset()
    mid = ds[8]                                          # 序列中部:窗口完整,无需 padding
    print("中部样本 obs 窗口(过去3帧):\n", mid["observation"].round(3))
    print("中部样本 action 窗口(未来3帧):", mid["action"].ravel().round(3), "补齐?", mid["action_is_pad"])
    assert not mid["obs_is_pad"].any() and not mid["action_is_pad"].any()

    edge = ds[14]                                        # 末帧:未来动作越界 → 触发端点补齐
    print("末帧样本 action 窗口:", edge["action"].ravel().round(3), "补齐?", edge["action_is_pad"])
    assert edge["action_is_pad"].tolist() == [False, True, True]        # 后两帧是补出来的
    assert edge["action"][1] == edge["action"][2]                      # 补齐=复制最后合法帧
    print("✅ delta_timestamps 把'秒偏移'切成过去观测+未来动作的时间窗,越界端点补齐并用 is_pad 标注,损失里可屏蔽")
    # 面试 Q&A：为什么要 action chunking(一次预测未来多帧动作)?
    #   A：减少策略被高频调用的次数、缓解误差累积/抖动,让机器人动作更平滑连贯——这也是 ACT/扩散策略的常见做法。
