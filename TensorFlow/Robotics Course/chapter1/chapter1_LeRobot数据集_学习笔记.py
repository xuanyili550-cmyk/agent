"""
================================================================================
 Robotics Course · Chapter 1 · LeRobot 数据集（学习笔记 · delta_timestamps 取帧 python 可跑）
================================================================================
 一句话：机器人学习的数据 = "观测(多路相机图) + 动作"的时间序列;delta_timestamps 决定每条样本取哪些时刻。
 本章讲(纯 python 模拟一段 episode,看清 delta_timestamps 怎么取帧,无需下数据)：
   ① 一段 episode = 按 fps 采样的一串帧(每帧有 obs 和 action)。
   ② delta_timestamps：相对"当前帧"取时刻。[0.0]=只当前;[-0.2,-0.1,0.0]=带 200ms 历史。
   ③ 为什么要历史帧：机器人策略常需"看一小段过去"才能决策(速度/趋势)。
 要点：观测历史(时间上下文)对策略很关键;真实用 LeRobotDataset(见 load_real 🟡需 lerobot)。
 跑：python3 chapter1_LeRobot数据集_学习笔记.py   （纯 python 真跑)
================================================================================
"""

def make_episode(n=20, fps=10):
    return [{"t": round(i / fps, 2), "obs": f"img{i}", "action": f"a{i}"} for i in range(n)]


def get_window(episode, idx, deltas, fps=10):
    """按 delta_timestamps 相对当前帧 idx 取帧:偏移 = round(delta*fps)。"""
    out = []
    for d in deltas:
        j = idx + round(d * fps)
        if 0 <= j < len(episode):
            out.append(episode[j])
    return out


def load_real():   # 🟡 需 pip install lerobot,默认不调用
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset("lerobot/svla_so101_pickplace",
                          delta_timestamps={"observation.images.up": [-0.2, -0.1, 0.0], "action": [0.0]})


def main():
    ep = make_episode()
    cur = get_window(ep, 5, [0.0])                     # 只当前帧
    hist = get_window(ep, 5, [-0.2, -0.1, 0.0])        # 带 200ms 历史(fps=10 → 偏移 -2,-1,0)
    assert len(cur) == 1 and len(hist) == 3
    assert hist[0]["obs"] == "img3" and hist[-1]["obs"] == "img5"
    print(f"✅ Ch1 跑通：当前帧={cur[0]['obs']};带历史={[f['obs'] for f in hist]}(200ms 上下文)")
    # 面试：Q 为什么要观测历史? A 单帧看不出速度/趋势,历史帧给时间上下文; Q delta_timestamps? A 相对当前帧取时刻。


if __name__ == "__main__":
    main()
