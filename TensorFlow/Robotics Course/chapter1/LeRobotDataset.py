"""
================================================================================
 Robotics Course · Chapter 1 · LeRobot 数据集（学习笔记描述）
================================================================================
 一句话：机器人学习的数据长啥样——LeRobotDataset 用 delta_timestamps 组织"观测+动作"的时序。
 本章讲：
   ① 加载 LeRobotDataset(如 svla_so101_pickplace 抓放任务)。
   ② delta_timestamps：取当前帧,或取历史帧序列(如 -0.2,-0.1,0.0)做上下文。
   ③ 观测(多路相机图像)+ 动作 的对齐。
 要点：机器人策略常需"观测历史"才能决策,时序对齐是数据关键。
 说明：需 lerobot 库 + 联网下数据;参考为主。
================================================================================
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Simple: current observation → current action
delta_timestamps = {
    "observation.images.up": [0.0],  # Just current frame
    "action": [0.0]  # Just current action
}

dataset = LeRobotDataset(
    "lerobot/svla_so101_pickplace",
    delta_timestamps=delta_timestamps
)

# Use observation history for context
delta_timestamps = {
    "observation.images.up": [-0.2, -0.1, 0.0],  # 200ms history
    "action": [0.0]  # Current action
}

dataset = LeRobotDataset(
    "lerobot/svla_so101_pickplace",
    delta_timestamps=delta_timestamps
)

sample = dataset[100]
# Images shape: [3, C, H, W] - 3 historical frames
# Action shape: [action_dim] - single current action


# Predict multiple future actions at once
delta_timestamps = {
    "observation.images.up": [-0.1, 0.0],  # Recent + current
    "action": [0.0, 0.1, 0.2, 0.3]  # Current + 3 future actions
}

dataset = LeRobotDataset(
    "lerobot/svla_so101_pickplace",
    delta_timestamps=delta_timestamps
)

sample = dataset[100]
# Images shape: [2, C, H, W] - 2 observation frames
# Action shape: [4, action_dim] - 4 action predictions


# Downloads dataset to local cache
dataset = LeRobotDataset("lerobot/svla_so101_pickplace")

# Fastest access after download
sample = dataset[100]

from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

# Stream data without downloading
streaming_dataset = StreamingLeRobotDataset(
    "lerobot/svla_so101_pickplace",
    delta_timestamps=delta_timestamps
)

# Works exactly like regular dataset
sample = streaming_dataset[100]

import torch
from torch.utils.data import DataLoader

# Create DataLoader for training
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for batch in dataloader:
    # Move to device
    observations = batch["observation.state"].to(device)
    actions = batch["action"].to(device)
    images = batch["observation.images.up"].to(device)

    # Your model training here
    # loss = model(observations, images, actions)
    # loss.backward()
    # optimizer.step()

