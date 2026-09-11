"""
 Robotics Course · Ch1 · 基础案例：delta_timestamps 取观测历史(纯 python,可跑)
 跑：python3 本文件
"""
fps = 10
episode = [{"obs": f"frame{i}", "action": f"act{i}"} for i in range(15)]

def window(idx, deltas):
    return [episode[idx + round(d * fps)] for d in deltas if 0 <= idx + round(d * fps) < len(episode)]

print("只当前帧 :", [f["obs"] for f in window(8, [0.0])])
print("带200ms历史:", [f["obs"] for f in window(8, [-0.2, -0.1, 0.0])])
print("✅ delta_timestamps 决定每条训练样本包含哪些时刻的观测/动作")
