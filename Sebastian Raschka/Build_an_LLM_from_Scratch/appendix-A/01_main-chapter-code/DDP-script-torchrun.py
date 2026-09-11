# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Appendix A: Introduction to PyTorch (Part 3)

# ============================================================
# 中文说明（模块级文档）
# ============================================================
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
# 附录 A「PyTorch 入门」第三部分的配套代码。
#
# 它演示的核心主题是 **分布式数据并行（DistributedDataParallel, DDP）**：
# 如何用 PyTorch 的 DDP 在多张 GPU 上并行训练同一个（很小的）神经网络分类模型。
#
# 与本书正文中用来训练 GPT 大模型的代码不同，这里用的是一个极简的玩具例子
# （2 维输入、2 分类输出的小型全连接网络），目的是让读者先专注理解
# **DDP 的机制本身**（进程组初始化、模型包裹、数据切分、梯度同步、优雅退出），
# 而不必同时应对大模型训练的复杂性。这些机制之后可以直接套用到真正的
# GPT 预训练/微调脚本上，实现多卡加速训练。
#
# 本脚本的运行方式是通过 `torchrun` 启动多进程（每个 GPU 对应一个进程），
# 例如：
#     torchrun --nproc_per_node=2 DDP-script-torchrun.py
# torchrun 会自动为每个进程设置好环境变量（WORLD_SIZE、RANK、LOCAL_RANK 等），
# 脚本据此判断自己是第几个进程、总共有多少个进程，从而完成分布式初始化。
#
# DDP 训练的核心思想：
#   1. 每个 GPU 进程各自持有一份完整的模型副本；
#   2. 训练数据通过 DistributedSampler 切分成不重叠的子集，分发给各个进程；
#   3. 各进程各自做前向传播、反向传播，计算出本地梯度；
#   4. DDP 在反向传播时自动通过 All-Reduce 通信，将所有进程的梯度求平均，
#      从而保证每个进程上的模型参数更新完全一致（等价于用更大的 batch 训练）；
#   5. 训练结束后需要调用 destroy_process_group() 清理分布式环境。
# ============================================================

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NEW imports:
# 新增的导入（相较于附录 A 前两部分的单机训练代码）：
# os, platform 用于读取环境变量、判断操作系统类型；
# DistributedSampler 用于在多进程间无重叠地切分数据集；
# DistributedDataParallel（DDP）用于包裹模型，实现梯度自动同步；
# init_process_group / destroy_process_group 用于创建和销毁分布式进程组。
import os
import platform
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# NEW: function to initialize a distributed process group (1 process / GPU)
# this allows communication among processes
# 新增：初始化分布式进程组的函数（每张 GPU 对应一个进程）。
# 进程组建立之后，各进程之间才能互相通信（例如同步梯度）。
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group

    中文说明：
        初始化当前进程的分布式训练环境。

    参数含义：
        rank (int): 当前进程的唯一编号（从 0 开始）。在单机多卡场景下，
            rank 通常等价于该进程要使用的 GPU 编号（这里用 LOCAL_RANK）。
        world_size (int): 进程组中的进程总数，通常等于参与训练的 GPU 总数。

    该函数没有返回值；它的作用是设置环境变量、初始化底层通信后端（NCCL/Gloo），
    并把当前进程绑定到对应编号的 GPU 上。
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    # 只有在 torchrun 没有预先设置这两个环境变量时才手动设置。
    # MASTER_ADDR / MASTER_PORT 用于告诉所有进程去哪里做“集合点”握手通信，
    # 从而建立起分布式进程组（类似所有进程先在这个地址/端口碰头）。
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    # initialize process group
    # 初始化进程组：根据操作系统选择合适的通信后端（backend）。
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        # Windows 下的 PyTorch 未编译支持 libuv，需要显式关闭，否则初始化会报错。
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        # Windows 用户通常需要用 "gloo" 而不是 "nccl" 作为通信后端；
        # gloo 是 Facebook 提供的通用集合通信库，兼容性更好但性能不如 nccl。
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        # nccl 是 NVIDIA 提供的专为多 GPU 通信优化的集合通信库，
        # 在 Linux + NVIDIA GPU 场景下性能最好，是 DDP 训练的默认首选后端。
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 把当前进程绑定到编号为 rank 的 GPU 上，之后该进程的所有张量/模型都放在这张卡上。
    torch.cuda.set_device(rank)


class ToyDataset(Dataset):
    """
    中文说明：
        一个极简的自定义数据集，用于演示 DDP 训练流程本身，
        不涉及真实的文本/图像数据处理。

    它把已经准备好的特征张量 X 和标签张量 y 包装成标准的
    PyTorch Dataset 接口（__getitem__ / __len__），
    以便配合 DataLoader（以及下面的 DistributedSampler）使用。
    """
    def __init__(self, X, y):
        # X: 形状 (num_samples, num_features) 的特征张量
        # y: 形状 (num_samples,) 的标签张量
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        """
        中文说明：
            按索引取出单条样本。

        返回：
            one_x: 形状 (num_features,) 的特征向量
            one_y: 标量张量，对应的类别标签
        """
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        """中文说明：返回数据集的样本总数（标签张量第 0 维大小）。"""
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    """
    中文说明：
        一个非常简单的多层感知机（MLP）分类器，用于演示 DDP，
        而不是本书中真正的 GPT Transformer 模型。

    结构：输入层 -> 隐藏层1(30) -> ReLU -> 隐藏层2(20) -> ReLU -> 输出层
    """
    def __init__(self, num_inputs, num_outputs):
        """
        参数：
            num_inputs (int): 输入特征维度
            num_outputs (int): 输出类别数（分类头的 logits 维度）
        """
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            # 第一层隐藏层：num_inputs -> 30
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            # 第二层隐藏层：30 -> 20
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            # 输出层：20 -> num_outputs，输出未归一化的分类 logits
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        """
        中文说明：
            前向传播。

        参数：
            x: 形状 (batch_size, num_inputs) 的输入张量

        返回：
            logits: 形状 (batch_size, num_outputs) 的未归一化分类得分，
                后续会喂给 F.cross_entropy 计算损失。
        """
        logits = self.layers(x)
        return logits


def prepare_dataset():
    """
    中文说明：
        构造训练集和测试集，并返回对应的 DataLoader。

    返回：
        train_loader: 训练集 DataLoader（使用 DistributedSampler 做数据切分）
        test_loader: 测试集 DataLoader（未做分布式切分，各进程会各自跑一遍全部测试数据）
    """
    # 训练集：5 条样本，每条样本 2 维特征
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    # 测试集：2 条样本
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    # 取消下面几行的注释，可以把数据集放大（通过加噪声复制），
    # 从而在最多 8 张 GPU 上运行本脚本而不会因为样本太少导致某些进程分不到数据。
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # NEW: False because of DistributedSampler below
        # 新增：这里必须设为 False，因为下面的 DistributedSampler 自己负责打乱顺序；
        # 如果 DataLoader 和 Sampler 都做 shuffle 会产生冲突/报错。
        pin_memory=True,
        drop_last=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        # 新增：DistributedSampler 会把数据集按进程数 world_size 切分成不重叠的子集，
        # 每个 rank 的进程只会看到分配给自己的那一份数据，
        # 从而实现「数据并行」——不同 GPU 训练不同的数据子集，梯度再汇总平均。
        sampler=DistributedSampler(train_ds)  # NEW
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# NEW: wrapper
# 新增：训练主流程封装函数，每个分布式进程都会独立执行一遍这个函数。
def main(rank, world_size, num_epochs):
    """
    中文说明：
        DDP 训练的主入口，每个进程（对应一张 GPU）都会调用一次本函数。

    参数：
        rank (int): 当前进程编号 / GPU 编号
        world_size (int): 总进程数（总 GPU 数）
        num_epochs (int): 训练轮数

    流程：初始化分布式环境 -> 准备数据 -> 构建模型并用 DDP 包裹
         -> 训练循环（前向/反向/梯度同步/参数更新）-> 评估准确率 -> 销毁进程组
    """

    ddp_setup(rank, world_size)  # NEW: initialize process groups
    # 新增：为当前进程初始化分布式进程组，之后才能使用 DDP 和跨进程通信。

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)  # 把模型参数搬到当前进程对应的 GPU（rank 号）上
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # NEW: wrap model with DDP
    # 新增：用 DDP 包裹模型。DDP 会在每次 loss.backward() 时，
    # 自动在所有进程间对梯度做 All-Reduce（求和后取平均），
    # 从而保证每个进程更新后的模型参数保持一致，等价于用了 world_size 倍的 batch size。
    # the core model is now accessible as model.module
    # 包裹后，原始（未包裹）的模型可以通过 model.module 访问。

    for epoch in range(num_epochs):
        # NEW: Set sampler to ensure each epoch has a different shuffle order
        # 新增：每个 epoch 开始前都要调用 set_epoch，
        # 这样 DistributedSampler 内部用来打乱顺序的随机种子会随 epoch 变化，
        # 否则每个 epoch 各进程看到的数据切分顺序都完全一样，起不到打乱的效果。
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:
            # features: 形状 (batch_size, 2)；labels: 形状 (batch_size,)
            features, labels = features.to(rank), labels.to(rank)  # New: use rank
            # 新增：把数据搬到当前进程对应的 GPU（用 rank 而不是固定的 "cuda" ）
            logits = model(features)  # 前向传播，logits 形状 (batch_size, 2)
            loss = F.cross_entropy(logits, labels)  # Loss function
            # 交叉熵损失：衡量预测分类分布与真实标签之间的差异

            optimizer.zero_grad()  # 清空上一步残留的梯度
            loss.backward()  # 反向传播计算梯度；DDP 会在这一步自动触发跨进程梯度 All-Reduce
            optimizer.step()  # 用（已经跨进程平均过的）梯度更新参数

            # LOGGING
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    try:
        # 分别在训练集、测试集上计算准确率（每个进程各自计算自己看到的数据）
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)

    ####################################################
    # NEW (not in the book):
    # 新增（书中未包含）：如果 GPU 数量与数据量不匹配（例如样本太少、
    # 某个进程分不到任何数据），compute_accuracy 里的除法会触发 ZeroDivisionError，
    # 这里捕获后给出更友好的报错提示，指导用户如何正确运行本脚本。
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "torchrun --nproc_per_node=2 DDP-script-torchrun.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )
    ####################################################

    destroy_process_group()  # NEW: cleanly exit distributed mode
    # 新增：训练结束后销毁分布式进程组，释放通信资源，避免进程挂起或资源泄漏。


def compute_accuracy(model, dataloader, device):
    """
    中文说明：
        在给定的数据加载器上计算分类准确率。

    参数：
        model: 待评估的（DDP 包裹的）模型
        dataloader: 提供 (features, labels) 批次的 DataLoader
        device: 数据和模型所在的设备（这里传入的是 rank，即 GPU 编号）

    返回：
        float，正确预测样本数 / 总样本数
    """
    model = model.eval()  # 切换到评估模式（本模型中无 Dropout/BatchNorm，但仍是良好习惯）
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():  # 评估阶段不需要梯度，节省显存和计算
            logits = model(features)  # 形状 (batch_size, num_outputs)
        predictions = torch.argmax(logits, dim=1)  # 取每条样本得分最高的类别，形状 (batch_size,)
        compare = labels == predictions  # 逐元素比较，得到布尔张量
        correct += torch.sum(compare)  # 累加正确预测的数量
        total_examples += len(compare)  # 累加总样本数
    return (correct / total_examples).item()  # 返回标量准确率


if __name__ == "__main__":
    # NEW: Use environment variables set by torchrun if available, otherwise default to single-process.
    # 新增：优先使用 torchrun 启动时自动注入的环境变量来确定分布式配置；
    # 如果这些环境变量不存在（说明不是通过 torchrun 启动的），
    # 则退化为单进程运行（world_size=1, rank=0）。
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])  # 总进程数（= 参与训练的 GPU 总数）
    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])  # 本机内的局部进程编号（多机多卡场景下更常用这个）
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])  # 全局进程编号
    else:
        rank = 0

    # Only print on rank 0 to avoid duplicate prints from each GPU process
    # 只在 rank 0 上打印环境信息，避免多张卡各自的进程重复打印同样的内容刷屏。
    if rank == 0:
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("Number of GPUs available:", torch.cuda.device_count())

    torch.manual_seed(123)  # 固定随机种子，保证结果可复现
    num_epochs = 3
    main(rank, world_size, num_epochs)  # 启动当前进程的训练主流程
