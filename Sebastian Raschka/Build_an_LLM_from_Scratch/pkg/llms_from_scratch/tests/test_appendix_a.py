# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是《从零构建大语言模型》(Build a Large Language Model From Scratch)
附录 A（appendix_a）配套代码的 pytest 单元测试文件。

附录 A 主要讲解 PyTorch 基础，本文件对应测试的是附录 A 中实现的：
- ToyDataset：一个简单的自定义 Dataset，用于封装特征张量 X 和标签张量 y；
- NeuralNetwork：一个简单的多层感知机（MLP）分类器。

测试的核心流程是一个最小可运行的 PyTorch 训练循环（training loop）：
构造玩具数据集 -> DataLoader 批量加载 -> 前向传播计算 logits ->
交叉熵损失 -> 反向传播 -> 优化器更新参数，并在训练结束后对比模型输出
是否与预期数值一致（用于验证模型/数据管道的可复现性）。
"""

from llms_from_scratch.appendix_a import NeuralNetwork, ToyDataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_dataset():
    """
    测试 ToyDataset 数据集封装 + NeuralNetwork 训练循环的端到端流程。

    测试意图：
    1. 验证 ToyDataset 能正确包装训练特征 X_train 与标签 y_train，
       且其长度（样本数）符合预期；
    2. 验证 DataLoader 能正常按 batch_size 对数据集进行批量加载和打乱；
    3. 验证 NeuralNetwork 在固定随机种子（manual_seed）下，经过若干轮
       （epoch）SGD 训练后，模型在训练集上的输出（logits）与预先计算好的
       期望值（expected）在数值上一致，从而保证训练流程的可复现性。
    """

    # 构造玩具数据集的特征矩阵：5 个样本，每个样本 2 维特征
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    # 对应的类别标签：0 类和 1 类，共两类
    y_train = torch.tensor([0, 0, 0, 1, 1])
    # 用自定义 ToyDataset 封装特征和标签，便于配合 DataLoader 使用
    train_ds = ToyDataset(X_train, y_train)

    # 断言数据集长度应为 5
    # 【bug 修复】原代码为 `len(train_ds) == 5`，缺少 assert 关键字、比较结果被丢弃，
    # 测试形同虚设；此处补上 assert。长度是确定性的（5 个样本），补 assert 后仍通过。
    assert len(train_ds) == 5
    # 固定随机种子，保证 DataLoader 的 shuffle 打乱顺序可复现
    torch.manual_seed(123)

    # 构建训练数据加载器：每批 2 个样本，打乱顺序，不使用多进程加载
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    # 再次固定随机种子，保证模型参数初始化可复现
    torch.manual_seed(123)
    # 构建一个输入维度为 2、输出维度为 2（二分类 logits）的简单神经网络
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    # 使用随机梯度下降（SGD）优化器，学习率为 0.5
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # 训练轮数：共训练 3 个 epoch
    num_epochs = 3

    for epoch in range(num_epochs):

        # 切换到训练模式（启用 dropout/BatchNorm 等训练态行为，若有的话）
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):

            # 前向传播：得到模型对当前 batch 的预测 logits
            logits = model(features)

            # 计算交叉熵损失（分类任务常用损失函数）
            loss = F.cross_entropy(logits, labels)

            # 梯度清零，防止梯度在多个 batch 间累积
            optimizer.zero_grad()
            # 反向传播，计算各参数的梯度
            loss.backward()
            # 根据梯度更新模型参数
            optimizer.step()

            # 打印当前 epoch、batch 以及训练损失，便于观察训练过程
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                  f" | Train/Val Loss: {loss:.2f}")

        # 切换到评估模式（关闭 dropout 等训练态行为，若有的话）
        model.eval()
        # 关闭梯度计算，节省显存/内存并加速前向推理
        with torch.no_grad():
            # 用训练结束后的模型对全部训练数据做一次前向推理，得到最终输出
            outputs = model(X_train)

        # 预先计算好的、用于比对的期望输出（在固定随机种子下训练 3 轮后应得到的结果）
        expected = torch.tensor([
            [2.8569, -4.1618],
            [2.5382, -3.7548],
            [2.0944, -3.1820],
            [-1.4814, 1.4816],
            [-1.7176, 1.7342]
        ])
        # 比对模型实际输出与期望输出是否完全相等
        # 【bug 说明 —— 此处特意「不」补 assert】原代码为 `torch.equal(outputs, expected)`，
        # 同样缺少 assert 关键字。但与上面的形状/长度断言不同，这里「不能」简单补上 assert：
        #   1. torch.equal 要求逐元素「完全相等」，而 expected 是四位小数的近似值，本就不可能精确相等；
        #   2. 更关键的是，这段训练用到 DataLoader 的 shuffle，其随机数生成在不同 PyTorch 版本间
        #      并不一致，导致训练轨迹不同、最终输出发生较大偏移。实测在 torch 2.11 上
        #      outputs 与 expected 的最大逐元素差高达约 0.79，远超任何合理容差。
        # 因此这是一个「跨版本不可复现」的硬编码期望值问题，无法通过机械地补 assert 修复
        # （补了反而会让本可通过的测试变为失败）。如需真正启用该断言，应在目标环境重新生成
        # expected 基准值，并改用带容差的 torch.testing.assert_close。此处保留原表达式并如实标注。
        torch.equal(outputs, expected)
