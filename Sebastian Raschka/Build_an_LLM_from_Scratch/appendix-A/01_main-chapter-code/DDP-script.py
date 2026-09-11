# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Appendix A: Introduction to PyTorch (Part 3)

# ============================================================
# 中文说明（模块级 docstring）
# ============================================================
"""
本文件对应《从零构建大语言模型》(Build a Large Language Model From Scratch)
附录 A（PyTorch 入门，第三部分）中的示例代码。

用途：
- 演示如何使用 PyTorch 的 DistributedDataParallel（DDP，分布式数据并行）
  在多张 GPU 上并行训练同一个简单的神经网络分类模型。
- 本例中的模型和数据集都非常小（一个 2 层的玩具 MLP + 5 条训练样本），
  目的不是训练出有用的模型，而是让读者看清楚"多 GPU 分布式训练"这一套
  工程机制是如何搭建起来的：
    1) 如何初始化进程组（process group），让多个进程（每个进程绑定一张 GPU）
       之间可以互相通信；
    2) 如何用 DistributedSampler 让每个 GPU 只拿到训练集中不重叠的一部分数据；
    3) 如何用 DDP 包装模型，使得每个进程各自做前向/反向传播，
       然后在 backward 时自动对梯度做 all-reduce（跨进程梯度同步/平均）；
    4) 如何用 torch.multiprocessing.spawn 一次性拉起多个训练进程（每张卡一个）；
    5) 训练结束后如何优雅地销毁进程组，释放资源。

角色定位：
- 在全书的知识体系里，这是"从零训练 LLM"之外的一段工程基础课：
  当模型和数据量变大之后，单卡训练往往放不下、跑不动，
  这时就需要用 DDP 这类分布式训练技术，把训练任务切分到多张 GPU
  （甚至多台机器）上并行完成。本附录用一个极简的玩具例子，
  把 DDP 最核心的几个 API 讲清楚，方便读者以后把同样的思路套用到
  书中后续章节训练的 GPT 模型上。

阅读建议：
- 建议对照非分布式版本的训练脚本（不使用 DDP 的普通单卡训练循环）来看，
  文件中标了 "NEW" 注释的地方，就是从单卡训练改造成多卡分布式训练时
  新增/修改的部分。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NEW imports:
# 新增的导入：这些都是实现"多进程 + 分布式通信"所需要的模块
import os
import platform
import torch.multiprocessing as mp  # PyTorch 对标准 multiprocessing 的封装，用于启动多个训练进程
from torch.utils.data.distributed import DistributedSampler  # 让不同进程/GPU 拿到数据集中互不重叠的子集
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行：多进程各自训练、梯度自动同步
from torch.distributed import init_process_group, destroy_process_group  # 初始化/销毁进程组（进程间通信的基础设施）


# NEW: function to initialize a distributed process group (1 process / GPU)
# this allows communication among processes
# 新增：初始化"分布式进程组"的函数（约定一个进程对应一张 GPU）。
# 有了这个进程组，各个进程之间才能够互相通信（例如同步梯度）。
def ddp_setup(rank, world_size):
    """
    初始化 PyTorch 分布式训练所需的进程组。

    这是每个训练进程启动后要做的第一件事：告诉 PyTorch"大家（所有进程）
    要怎么找到彼此、用什么协议通信"。只有进程组初始化成功后，
    后面用 DDP 包装模型时，梯度的 all-reduce（跨进程求和/平均）通信
    才能正常工作。

    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
        rank: 当前进程的唯一编号（0, 1, 2, ...），通常也对应它使用的 GPU 编号
        world_size: 进程组中进程的总数（通常等于参与训练的 GPU 总数）
    """
    # rank of machine running rank:0 process
    # here, we assume all GPUs are on the same machine
    # rank 为 0 的进程所在机器的地址；这里假设所有 GPU 都在同一台机器（单机多卡）上，
    # 所以直接用 "localhost" 即可（多机多卡场景下这里需要填真实的主节点 IP）
    os.environ["MASTER_ADDR"] = "localhost"
    # any free port on the machine
    # 主节点用来监听通信的端口号，只要是机器上空闲的端口即可
    os.environ["MASTER_PORT"] = "12345"

    # initialize process group
    # 根据操作系统选择合适的通信后端（backend），并真正初始化进程组
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        # Windows 版 PyTorch 未编译支持 libuv，这里显式关闭，避免报错
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        # Windows 用户通常无法使用 NCCL（仅支持 Linux），需改用 gloo 后端
        # gloo：Facebook 的集合通信库，支持 CPU/GPU，但性能通常不如 NCCL
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        # nccl：NVIDIA 的集合通信库，专为多 GPU 通信优化（如 all-reduce 梯度同步），
        # 在 Linux + NVIDIA GPU 环境下是 DDP 训练的首选后端，速度最快
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 把当前进程绑定到编号为 rank 的 GPU 上，
    # 这样后续调用 .to(rank) / .cuda(rank) 时数据和模型都会放到这张卡上
    torch.cuda.set_device(rank)


class ToyDataset(Dataset):
    """
    一个极简的自定义 Dataset：把已经加载到内存中的特征张量 X 和标签张量 y
    包装成 PyTorch Dataset 接口，供 DataLoader 按索引取样本使用。

    参数：
        X: 特征张量，形状约定为 (num_samples, num_features)
        y: 标签张量，形状约定为 (num_samples,)
    """
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        # 按索引取出一条样本：
        # one_x 形状为 (num_features,)，one_y 是标量（该样本的类别标签）
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        # 数据集大小 = 标签张量第 0 维的长度，即样本总数
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    """
    一个极简的多层感知机（MLP）分类模型，用于在这个 DDP 示例中演示
    分布式训练流程，而非追求实际的分类效果。

    结构：输入层 -> 隐藏层1(30) -> ReLU -> 隐藏层2(20) -> ReLU -> 输出层

    参数：
        num_inputs: 输入特征维度（本例中为 2）
        num_outputs: 输出类别数 / logits 维度（本例中为 2，二分类）
    """
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            # 第 1 个隐藏层：(batch, num_inputs) -> (batch, 30)
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            # 第 2 个隐藏层：(batch, 30) -> (batch, 20)
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            # 输出层：(batch, 20) -> (batch, num_outputs)，输出未归一化的 logits
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        # x 形状: (batch, num_inputs)
        # logits 形状: (batch, num_outputs)，后续会喂给 cross_entropy 计算损失
        logits = self.layers(x)
        return logits


def prepare_dataset():
    """
    构造本示例使用的玩具训练集/测试集，并封装成支持分布式采样的 DataLoader。

    返回：
        train_loader: 训练集 DataLoader，内部使用 DistributedSampler
                       将数据切分给不同的 GPU/进程，各进程互不重叠
        test_loader: 测试集 DataLoader（未做分布式切分，供每个进程各自评估用）
    """
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    # 如果想在多于 2 张 GPU 上运行本脚本，需要更大的数据集（否则每张卡分不到数据），
    # 可以取消下面几行的注释，把数据复制并加噪声扩充成原来的 factor 倍
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
        # 新增说明：这里必须设为 False，因为下面的 DistributedSampler
        # 已经负责数据的打乱和切分逻辑了，二者的 shuffle 是互斥/冲突的
        pin_memory=True,  # 将数据锁页内存，加速 CPU -> GPU 的数据拷贝
        drop_last=True,  # 丢弃最后不满一个 batch 的样本，避免各进程 batch 数不一致
        # NEW: chunk batches across GPUs without overlapping samples:
        # 新增：DistributedSampler 会根据 world_size 和当前 rank，
        # 把训练集样本索引切分成互不重叠的若干份，每个进程（GPU）只拿自己那一份，
        # 这样多个 GPU 加起来正好覆盖整个训练集一次（一个 epoch），且不重复
        sampler=DistributedSampler(train_ds)  # NEW
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        # 测试集这里没有使用 DistributedSampler，
        # 所以每个进程都会用完整的测试集各自评估一遍（简单但有冗余计算）
    )
    return train_loader, test_loader


# NEW: wrapper
# 新增：每个子进程真正执行的训练入口函数。
# 使用 mp.spawn 启动多进程时，会为每个进程自动传入不同的 rank，
# 其余参数（world_size, num_epochs）由 args 统一传入。
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # NEW: initialize process groups
    # 新增：先完成分布式进程组初始化，之后才能安全地使用 DDP 包装模型

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)  # 把模型参数搬到编号为 rank 的这块 GPU 上
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # NEW: wrap model with DDP
    # the core model is now accessible as model.module
    # 新增：用 DistributedDataParallel 包装模型后：
    # - 每个进程仍然独立做前向传播、计算各自那份 mini-batch 的损失；
    # - 在 loss.backward() 时，DDP 会自动在各进程间对梯度做 all-reduce
    #   （跨 GPU 求和后取平均），保证每个进程更新后得到的模型参数完全一致；
    # - 原始（未包装）的模型可以通过 model.module 访问，
    #   例如保存 checkpoint 时通常要保存 model.module.state_dict()

    for epoch in range(num_epochs):
        # NEW: Set sampler to ensure each epoch has a different shuffle order
        # 新增：每个 epoch 开始前都要调用 set_epoch，
        # 它会用 epoch 号作为随机种子的一部分，让每一轮的数据打乱顺序都不同，
        # 否则所有 epoch 的数据切分/顺序都会完全一样，起不到打乱数据的效果
        train_loader.sampler.set_epoch(epoch)

        model.train()
        for features, labels in train_loader:
            # features 形状: (batch, 2)，labels 形状: (batch,)

            features, labels = features.to(rank), labels.to(rank)  # New: use rank
            # 新增：把当前 batch 的数据搬到本进程对应的 GPU（rank）上，
            # 保证数据和模型在同一块设备上，否则前向传播会报错
            logits = model(features)  # logits 形状: (batch, 2)
            loss = F.cross_entropy(logits, labels)  # Loss function
            # 交叉熵损失：内部会先对 logits 做 softmax，再与真实标签计算负对数似然

            optimizer.zero_grad()  # 清空上一步残留的梯度
            loss.backward()
            # 反向传播计算梯度；由于模型被 DDP 包装，这一步内部会自动触发
            # 跨进程的梯度 all-reduce（通信同步），使各进程梯度保持一致
            optimizer.step()  # 用同步后的梯度更新本进程（也是本地 GPU）上的模型参数

            # LOGGING
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()

    try:
        # 训练结束后分别在训练集/测试集上评估准确率；
        # 注意每个进程各自评估，会各自打印一份结果（GPU 编号不同）
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training accuracy", train_acc)
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test accuracy", test_acc)

    ####################################################
    # NEW (not in the book):
    # 新增（书中没有的部分）：给出更友好的报错提示。
    # 因为 train_loader 用了 DistributedSampler 切分数据，
    # 如果 GPU 数量过多而数据集又太小，可能导致某个进程分不到任何样本，
    # 从而在 compute_accuracy 里除以 0（ZeroDivisionError）
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )
    ####################################################

    destroy_process_group()  # NEW: cleanly exit distributed mode
    # 新增：训练/评估都结束后，销毁进程组，释放分布式通信相关的资源，
    # 避免进程退出时出现资源未清理的警告或残留


def compute_accuracy(model, dataloader, device):
    """
    在给定的 dataloader 上计算模型分类准确率。

    参数：
        model: 待评估的模型（可以是被 DDP 包装过的模型）
        dataloader: 提供 (features, labels) 批次数据的 DataLoader
        device: 数据要搬运到的设备（这里传入的是 rank，即某张 GPU 的编号）

    返回：
        准确率（0~1 之间的 float），即预测正确的样本数 / 总样本数
    """
    model = model.eval()  # 切换到评估模式（本例中无 Dropout/BatchNorm，实际影响不大，但是好习惯）
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():  # 评估阶段不需要构建计算图，节省显存和算力
            logits = model(features)  # logits 形状: (batch, num_outputs)
        predictions = torch.argmax(logits, dim=1)  # 取每个样本得分最高的类别，形状: (batch,)
        compare = labels == predictions  # 逐元素比较，得到布尔张量，形状: (batch,)
        correct += torch.sum(compare)  # 累加本批次预测正确的样本数
        total_examples += len(compare)  # 累加本批次的样本总数
    return (correct / total_examples).item()  # 正确率 = 累计正确数 / 累计总数，转成 Python float 返回


if __name__ == "__main__":
    # This script may not work for GPUs > 2 due to the small dataset
    # Run `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py` if you have GPUs > 2
    # 由于数据集太小（只有 5 条训练样本），当 GPU 数量超过 2 时，
    # DistributedSampler 切分后可能有些进程分不到数据，导致报错；
    # 如果机器上有超过 2 张 GPU，可以用 CUDA_VISIBLE_DEVICES=0,1 限定只用其中 2 张
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)  # 固定随机种子，保证结果可复现

    # NEW: spawn new processes
    # note that spawn will automatically pass the rank
    # 新增：用 torch.multiprocessing.spawn 一次性拉起 world_size 个子进程，
    # 每个子进程都会执行 main 函数，且 spawn 会自动把进程编号作为第一个参数（rank）
    # 传给 main（对应 main(rank, world_size, num_epochs) 的签名）
    num_epochs = 3
    world_size = torch.cuda.device_count()  # 进程数 = 可见 GPU 数量，一张卡对应一个进程
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size spawns one process per GPU
    # nprocs=world_size：为每一张 GPU 各自启动一个独立的训练进程，
    # 这些进程之间通过前面初始化的进程组（NCCL/gloo 后端）互相通信、同步梯度
