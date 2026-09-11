"""
================================================================================
 CV Course · Chapter 8 · 3D 视觉的应用与数学基础（学习笔记描述）
================================================================================
 一句话：让机器"看懂三维世界"——应用遍及机器人/自动驾驶/医疗/AR/VR，底层是三维线性代数。
 本章讲：
   ① 应用：物体抓取、自主导航/深度感知、医学三维影像、AR/VR、动捕。
   ② 三维数据的线性代数基础(坐标变换/投影)。
   ③ 与后续 ML for 3D 的衔接。
 要点：2D→3D 关键在深度/多视角几何。
 说明：应用综述 + 数学基础(概念为主)，参考为主。
================================================================================
"""

# 3D视觉的应用
# 3D计算机视觉使机器能够感知和理解三维环境，从而在众多行业中催生出各种各样的应用。在本节中，您可以了解一些令人兴奋的3D计算机视觉应用。
#
# 机器人与自动化
# 物体操作： 3D视觉系统使机器人能够精确识别和抓取各种形状和尺寸的物体。这使它们能够执行诸如拾取和放置、组装和包装等任务。
# 质量控制： 3D视觉系统可用于检测制造的零件和产品是否存在缺陷，从而确保质量和一致性。
# 自主导航
# 自动驾驶汽车： 3D视觉摄像头和算法有助于深度感知——即感知与周围物体的距离信息。这使得自动驾驶汽车能够更好地感知周围环境，并在道路上安全行驶。
# 自主无人机和机器人：自主无人机和机器人利用 3D 计算机视觉技术来确定自身相对于周围环境的位置，绘制环境地图，并通过避开障碍物安全导航。
# 卫生保健
# 医学影像与诊断： 3D计算机视觉技术应用于CT扫描、MRI等医学影像技术。这些图像有助于可视化体内器官的3D结构，从而辅助诊断、治疗计划和外科手术。
# 手术机器人： 3D视觉系统辅助外科医生以更高的精度和控制力进行复杂的手术。
# 增强现实和虚拟现实
# 增强现实（AR）：像虚拟试穿衣服或在购买家具前将其可视化放置在您的生活空间中的 AR 应用，都是由 3D 计算机视觉实现的。
# 虚拟现实（VR）： VR体验通过3D计算机视觉技术，将用户沉浸在三维、交互式且通常逼真的环境中。
# 娱乐和游戏
# 动画和动作捕捉： 3D视觉系统可以跟踪和记录人体动作，从而在电影中创建逼真的角色动画。
# 游戏： 3D视觉技术使游戏开发者能够创建细节丰富、逼真的3D环境、景观、建筑和物体，并具有高质量的光照和阴影效果。

#三维数据的线性代数基础
# 不同的坐标系统对这种坐标系有不同的约定。最重要的区别在于惯用手，也就是X轴、Y轴和Z轴的相对方向。记住这种区别最简单的方法是将中指指向内侧，使拇指、食指和中指大致呈直角。左手的拇指（X轴）、食指（Y轴）和中指（Z轴）构成左手坐标系。同样，右手的手指构成右手坐标系。
#
# 在数学和物理学中，通常使用右手坐标系。然而，在计算机图形学中，不同的库和环境有不同的坐标约定。值得注意的是，Blender、PyTorch3D 和 OpenGL（大多）使用右手坐标系，而 DirectX 使用左手坐标系。本文将采用右手坐标系，遵循 Blender 和 NerfStudio 的约定。
#
# 转变
# 能够旋转、缩放和平移这些空间坐标非常有用。例如，如果物体在移动，或者如果我们想将这些坐标从相对于某个固定坐标系的世界坐标转换为相对于我们摄像机的坐标，就需要用到这些坐标。
#
# 这些变换可以用矩阵表示。这里我们用@表示矩阵乘法。为了能够以一致的方式表示平移、旋转和缩放，我们采用三维坐标。
# [
# x
# ，
# 是
# ，
# z
# ]
# [ x ,y ，z ]并添加一个额外的坐标
# 西
# =
# 1
# 西=1这些被称为齐次坐标——更一般地说，
# 西
# 西可以取任意值，并且所有点都在四维直线上
# [
# 西
# x
# ，
# 西
# 是
# ，
# 西
# z
# ，
# 西
# ]
# [ w x ,w y ，w z ，w ]对应于同一点
# [
# x
# ，
# 是
# ，
# z
# ]
# [ x ,y ，z ]在三维空间中。然而，在这里，
# 西
# 西始终为 1。
#
# Pytorch3d等库提供了一系列用于生成和操作变换的函数。
#
# 另一个需要注意的约定是：OpenGL 将位置视为列向量x（形状为 4x1），并M通过将向量左乘矩阵（M @ x）来应用变换；
# 而 DirectX 和 PyTorch3D 将位置视为行向量（形状为 1x4），并通过将向量右乘矩阵（）来应用变换x @ M。
# 要在两种约定之间进行转换，我们需要对矩阵进行转置M.T。我们将通过几个代码片段展示立方体在不同变换矩阵下的变换方式。
# 在这些代码片段中，我们将使用 OpenGL 的约定。

import numpy as np
import matplotlib.pyplot as plt


def plot_cube(ax, cube, label, color="black"):
    ax.scatter3D(cube[0, :], cube[1, :], cube[2, :], label=label, color=color)
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    for line in lines:
        ax.plot3D(cube[0, line], cube[1, line], cube[2, line], color=color)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

#规模化
# set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# plot original cube
plot_cube(ax, cube, label="Original", color="blue")

# scaling matrix (scale by 2 along x-axis and by 0.5 along y-axis)
scaling_matrix = np.array([[2, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


scaled_cube = scaling_matrix @ cube

plot_cube(ax, scaled_cube, label="Scaled", color="green")

#轮换
# set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# plot original cube
plot_cube(ax, cube, label="Original", color="blue")

# rotation matrix: +20 deg around x-axis
angle = 20 * np.pi / 180
rotation_matrix = np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1],
    ]
)


rotated_cube = rotation_matrix @ cube

plot_cube(ax, rotated_cube, label="Rotated", color="orange")

#组合变换
# set up figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# plot original cube
plot_cube(ax, cube, label="Original", color="blue")

# combination of transforms
combination_transform = rotation_matrix.dot(scaling_matrix.dot(translation_matrix))
final_result = combination_transform.dot(cube)
plot_cube(ax, final_result, label="Combined", color="violet")
#
# 微调入门
# 现在，让我们深入研究代码。如果您没有强大的GPU，我强烈建议您使用Kaggle而不是Colab。Kaggle具有以下几个优势：
#
# 每周最多可使用 30 小时 GPU
# 无连接中断
# 非常快速便捷地访问数据集
# 其中一种配置允许同时使用两个GPU，这将有助于您进行分布式训练。
# 你可以使用Kaggle 上的这个 notebook 直接进入代码部分。
#
# 我们将在这里详细讲解所有内容。首先，让我们从作者的代码库下载所有必要的模块以及带有 ViT-S 编码器的最小模型的检查点。
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import random
import h5py

import sys

sys.path.append("/kaggle/working/Depth-Anything-V2/metric_depth")

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import notebook_launcher
from accelerate import DistributedDataParallelKwargs

import transformers

import torch
import torchvision
from torchvision.transforms import v2
from torchvision.transforms import Compose
import torch.nn.functional as F
import albumentations as A

from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


train_paths = get_all_files("/kaggle/input/nyu-depth-dataset-v2/nyudepthv2/train")
val_paths = get_all_files("/kaggle/input/nyu-depth-dataset-v2/nyudepthv2/val")

# NYU Depth V2 40k. Original NYU is 400k
class NYU(torch.utils.data.Dataset):
    def __init__(self, paths, mode, size=(518, 518)):

        self.mode = mode  # train or val
        self.size = size
        self.paths = paths

        net_w, net_h = size
        # author's transforms
        self.transform = Compose(
            [
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True if mode == "train" else False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
            + ([Crop(size[0])] if self.mode == "train" else [])
        )

        # only horizontal flip in the paper
        self.augs = A.Compose(
            [
                A.HorizontalFlip(),
                A.ColorJitter(hue=0.1, contrast=0.1, brightness=0.1, saturation=0.1),
                A.GaussNoise(var_limit=25),
            ]
        )

    def __getitem__(self, item):
        path = self.paths[item]
        image, depth = self.h5_loader(path)

        if self.mode == "train":
            augmented = self.augs(image=image, mask=depth)
            image = augmented["image"] / 255.0
            depth = augmented["mask"]
        else:
            image = image / 255.0

        sample = self.transform({"image": image, "depth": depth})

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])

        # sometimes there are masks for valid depths in datasets because of noise e.t.c
        #         sample['valid_mask'] = ...

        return sample

    def __len__(self):
        return len(self.paths)

    def h5_loader(self, path):
        h5f = h5py.File(path, "r")
        rgb = np.array(h5f["rgb"])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f["depth"])
        return rgb, depth
# 原始的 NYU-D 数据集包含 40.7 万个样本，但我们只使用了其中的 4 万个样本子集。这会对最终模型的质量产生轻微影响。
# 论文作者仅使用水平翻转进行数据增强。
# 有时，深度图中的某些点可能无法正确处理，从而产生“坏像素”。
# 一些数据集除了图像和深度图之外，还包含一个用于区分有效像素和无效像素的掩码。
# 该掩码对于将坏像素从损失和指标计算中排除至关重要。

num_images = 5

fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

train_set = NYU(train_paths, mode="train")

for i in range(num_images):
    sample = train_set[i * 1000]
    img, depth = sample["image"].numpy(), sample["depth"].numpy()

    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = img * std + mean

    axes[i, 0].imshow(np.transpose(img, (1, 2, 0)))
    axes[i, 0].set_title("Image")
    axes[i, 0].axis("off")

    im1 = axes[i, 1].imshow(depth, cmap="viridis", vmin=0)
    axes[i, 1].set_title("True Depth")
    axes[i, 1].axis("off")
    fig.colorbar(im1, ax=axes[i, 1])

plt.tight_layout()

def get_dataloaders(batch_size):

    train_dataset = NYU(train_paths, mode="train")
    val_dataset = NYU(val_paths, mode="val")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # for dynamic resolution evaluations without padding
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    return train_dataloader, val_dataloader


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    mae = torch.mean(torch.abs(diff))

    silog = torch.sqrt(
        torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)
    )

    return {
        "d1": d1.detach(),
        "abs_rel": abs_rel.detach(),
        "rmse": rmse.detach(),
        "mae": mae.detach(),
        "silog": silog.detach(),
    }


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    mae = torch.mean(torch.abs(diff))

    silog = torch.sqrt(
        torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)
    )

    return {
        "d1": d1.detach(),
        "abs_rel": abs_rel.detach(),
        "rmse": rmse.detach(),
        "mae": mae.detach(),
        "silog": silog.detach(),
    }


def train_fn():

    set_seed(seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )

    # in the paper they initialize decoder randomly and use only encoder pretrained weights. Then full fine-tune
    # ViT-S encoder here
    model = DepthAnythingV2(**{**model_configs[model_encoder], "max_depth": max_depth})
    model.load_state_dict(
        {k: v for k, v in torch.load(model_weights_path).items() if "pretrained" in k},
        strict=False,
    )

    optim = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "pretrained" in name
                ],
                "lr": lr,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "pretrained" not in name
                ],
                "lr": lr * 10,
            },
        ],
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = SiLogLoss()  # author's loss

    train_dataloader, val_dataloader = get_dataloaders(batch_size)

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optim,
        len(train_dataloader) * warmup_epochs,
        num_epochs * scheduler_rate * len(train_dataloader),
    )

    model, optim, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optim, train_dataloader, val_dataloader, scheduler
    )

    if load_state:
        accelerator.wait_for_everyone()
        accelerator.load_state(state_path)

    best_val_absrel = 1000

    for epoch in range(1, num_epochs):

        model.train()
        train_loss = 0
        for sample in tqdm(
            train_dataloader, disable=not accelerator.is_local_main_process
        ):
            optim.zero_grad()

            img, depth = sample["image"], sample["depth"]

            pred = model(img)
            # mask
            loss = criterion(pred, depth, (depth <= max_depth) & (depth >= 0.001))

            accelerator.backward(loss)
            optim.step()
            scheduler.step()

            train_loss += loss.detach()

        train_loss /= len(train_dataloader)
        train_loss = accelerator.reduce(train_loss, reduction="mean").item()

        model.eval()
        results = {"d1": 0, "abs_rel": 0, "rmse": 0, "mae": 0, "silog": 0}
        for sample in tqdm(val_dataloader, disable=not accelerator.is_local_main_process):

            img, depth = sample["image"].float(), sample["depth"][0]

            with torch.no_grad():
                pred = model(img)
                # evaluate on the original resolution
                pred = F.interpolate(
                    pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True
                )[0, 0]

            valid_mask = (depth <= max_depth) & (depth >= 0.001)

            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

            for k in results.keys():
                results[k] += cur_results[k]

        for k in results.keys():
            results[k] = results[k] / len(val_dataloader)
            results[k] = round(accelerator.reduce(results[k], reduction="mean").item(), 3)

        accelerator.wait_for_everyone()
        accelerator.save_state(state_path, safe_serialization=False)

        if results["abs_rel"] < best_val_absrel:
            best_val_absrel = results["abs_rel"]
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_local_main_process:
                torch.save(unwrapped_model.state_dict(), save_model_path)

        accelerator.print(
            f"epoch_{epoch},  train_loss = {train_loss:.5f}, val_metrics = {results}"
        )


# P.S. While testing one configuration, I encountered an error in which the loss turned into nan.
# This is fixed by adding a small epsilon to the predictions to prevent division by 0

# You can run this code with 1 gpu. Just set num_processes=1
notebook_launcher(train_fn, num_processes=2)


model = DepthAnythingV2(**{**model_configs[model_encoder], "max_depth": max_depth}).to(
    "cuda"
)
model.load_state_dict(torch.load(save_model_path))

num_images = 10

fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

val_dataset = NYU(val_paths, mode="val")
model.eval()
for i in range(num_images):
    sample = val_dataset[i]
    img, depth = sample["image"], sample["depth"]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.inference_mode():
        pred = model(img.unsqueeze(0).to("cuda"))
        pred = F.interpolate(
            pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True
        )[0, 0]

    img = img * std + mean

    axes[i, 0].imshow(img.permute(1, 2, 0))
    axes[i, 0].set_title("Image")
    axes[i, 0].axis("off")

    max_depth = max(depth.max(), pred.cpu().max())

    im1 = axes[i, 1].imshow(depth, cmap="viridis", vmin=0, vmax=max_depth)
    axes[i, 1].set_title("True Depth")
    axes[i, 1].axis("off")
    fig.colorbar(im1, ax=axes[i, 1])

    im2 = axes[i, 2].imshow(pred.cpu(), cmap="viridis", vmin=0, vmax=max_depth)
    axes[i, 2].set_title("Predicted Depth")
    axes[i, 2].axis("off")
    fig.colorbar(im2, ax=axes[i, 2])

plt.tight_layout()


#神经辐射场（NeRFs）

import torch
import mediapy as media
import numpy as np


def positional_encoding(in_tensor, num_frequencies, min_freq_exp, max_freq_exp):
    """Function for positional encoding."""
    # Scale input tensor to [0, 2 * pi]
    scaled_in_tensor = 2 * np.pi * in_tensor
    # Generate frequency spectrum
    freqs = 2 ** torch.linspace(
        min_freq_exp, max_freq_exp, num_frequencies, device=in_tensor.device
    )
    # Generate encodings
    scaled_inputs = scaled_in_tensor.unsqueeze(-1) * freqs
    encoded_inputs = torch.cat(
        [torch.sin(scaled_inputs), torch.cos(scaled_inputs)], dim=-1
    )
    return encoded_inputs.view(*in_tensor.shape[:-1], -1)


def visualize_grid(grid, encoded_images, resolution):
    """Helper Function to visualize grid."""
    # Split the grid into separate channels for x and y
    x_channel, y_channel = grid[..., 0], grid[..., 1]
    # Show the original grid
    print("Input Values:")
    media.show_images([x_channel, y_channel], cmap="plasma", border=True)
    # Show the encoded grid
    print("Encoded Values:")
    num_channels_to_visualize = min(
        8, encoded_images.shape[-1]
    )  # Visualize up to 8 channels
    encoded_images_to_show = encoded_images.view(resolution, resolution, -1).permute(
        2, 0, 1
    )[:num_channels_to_visualize]
    media.show_images(encoded_images_to_show, vmin=-1, vmax=1, cmap="plasma", border=True)


# Parameters similar to your NeRFEncoding example
num_frequencies = 4
min_freq_exp = 0
max_freq_exp = 6
resolution = 128

# Generate a 2D grid of points in the range [0, 1]
x_samples = torch.linspace(0, 1, resolution)
y_samples = torch.linspace(0, 1, resolution)
grid = torch.stack(
    torch.meshgrid(x_samples, y_samples), dim=-1
)  # [resolution, resolution, 2]

# Apply positional encoding
encoded_grid = positional_encoding(grid, num_frequencies, min_freq_exp, max_freq_exp)

# Visualize result
visualize_grid(grid, encoded_grid, resolution)
