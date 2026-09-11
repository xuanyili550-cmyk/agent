# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文文档字符串（新增，不影响原有代码逻辑）。

本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 5 章（ch05）配套代码的 pytest 单元测试文件。

主要测试内容：
- 验证 `train_model_simple`（简单训练循环，定义在 llms_from_scratch.ch05 中）
  能够在极小规模的数据（仅取 1 个 batch 做训练与验证）上正常跑通训练流程；
- 同时对两种模型实现（标准版 GPTModel 与加速版 GPTModelFast）分别进行参数化测试，
  确保两者在相同配置和随机种子下训练行为一致（如初始 loss 数值、loss 是否下降等）。

测试会在需要时从 GitHub 下载示例文本《The Verdict》，并将其切分为训练集/验证集，
再通过 DataLoader 构造出仅含 1 个 batch 的迷你数据加载器，以加快测试速度。
"""

from llms_from_scratch.ch02 import create_dataloader_v1  # 第2章：构造用于GPT训练的滑动窗口式 DataLoader
from llms_from_scratch.ch04 import GPTModel, GPTModelFast  # 第4章：两种GPT模型实现（标准版与加速版）
from llms_from_scratch.ch05 import train_model_simple  # 第5章：简化版训练循环函数，本文件的主要测试对象

import os  # 用于判断文件是否已存在，避免重复下载

import requests  # 用于从远程URL下载示例文本数据
import pytest  # 测试框架，提供 @pytest.mark.parametrize 等能力
import tiktoken  # OpenAI 开源的 BPE 分词器，这里用于GPT-2编码
import torch  # PyTorch 深度学习框架
from torch.utils.data import Subset, DataLoader  # Subset 用于截取数据集前几条样本，DataLoader 用于批量加载


# GPT-124M 模型的配置字典（与书中标准配置一致，仅context_length为测试加速而缩短）
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # Shortened for test speed  # 原注释：为加快测试速度而缩短上下文长度
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# 训练过程中使用的其他超参数配置
OTHER_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 2,
    "batch_size": 1,
    "weight_decay": 0.1
}


# 使用 pytest 参数化装饰器，让同一个测试函数分别针对 GPTModel（标准实现）
# 和 GPTModelFast（加速实现）各运行一次，验证两种实现的训练行为是否一致
@pytest.mark.parametrize("ModelClass", [GPTModel, GPTModelFast])
def test_train_simple(tmp_path, ModelClass):
    """
    测试意图（中文说明）：

    验证在极小规模数据（仅1个训练batch + 1个验证batch）上，
    使用 `train_model_simple` 函数训练指定的 GPT 模型类（GPTModel 或 GPTModelFast）
    能够正常运行，且训练结果符合预期：
      1. 固定随机种子后，第一个 epoch 的训练/验证 loss 应为确定的数值（用于回归测试，
         检测代码改动是否意外改变了模型的前向/训练行为）；
      2. 训练过程中 loss 应该有所下降（最后一次训练loss小于第一次），
         证明优化器确实在更新模型参数、训练流程本身有效。

    参数:
        tmp_path: pytest 内置fixture，提供一个每次测试独立、测试结束后自动清理的临时目录，
                  用于存放下载的示例文本文件。
        ModelClass: 由 @pytest.mark.parametrize 注入，取值为 GPTModel 或 GPTModelFast，
                    用于测试两种模型实现在训练循环中的行为一致性。
    """
    torch.manual_seed(123)  # 固定随机种子，保证模型初始化权重、dropout等随机行为可复现
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用GPU，否则退回CPU

    ##############################
    # Download data if necessary
    ##############################
    # 若本地临时目录中不存在示例文本，则从GitHub下载；否则直接读取本地缓存文件，避免重复下载
    file_path = tmp_path / "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)  # 发起HTTP GET请求下载文本，设置30秒超时
        response.raise_for_status()  # 若响应状态码表示错误（如4xx/5xx），主动抛出异常，防止用错误内容继续测试
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)  # 将下载内容缓存到本地临时文件，便于同一测试会话内复用
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text_data = f.read()  # 已存在则直接读取本地缓存内容

    ##############################
    # Set up dataloaders
    ##############################
    # 按照90%训练 / 10%验证的比例切分文本数据
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    # 构造训练集DataLoader：滑动窗口的步长(stride)等于上下文长度，即窗口之间不重叠
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=OTHER_SETTINGS["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,   # 丢弃最后不足一个batch的数据，保证每个batch大小一致
        shuffle=True,     # 训练集需要打乱顺序，避免模型学到样本顺序带来的偏差
        num_workers=0
    )

    # 构造验证集DataLoader：不打乱顺序、不丢弃最后一个不完整batch，便于评估时覆盖全部验证数据
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=OTHER_SETTINGS["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # Limit to 1 batch for speed  # 原注释：为提升测试速度，仅取1个batch
    # 通过 Subset 截取数据集中的第0号样本（range(1)即索引[0]），从而将训练/验证都压缩到单个batch，
    # 大幅加快测试运行速度，同时仍能验证训练流程是否可正常运行。
    train_subset = Subset(train_loader.dataset, range(1))
    one_batch_train_loader = DataLoader(train_subset, batch_size=1)
    val_subset = Subset(val_loader.dataset, range(1))
    one_batch_val_loader = DataLoader(val_subset, batch_size=1)

    ##############################
    # Train model
    ##############################
    model = ModelClass(GPT_CONFIG_124M)  # 根据参数化传入的模型类（GPTModel或GPTModelFast）实例化模型
    model.to(device)  # 将模型迁移到目标计算设备（GPU或CPU）

    # 使用AdamW优化器，这是训练Transformer类模型的常用选择（对权重衰减做了解耦处理）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=OTHER_SETTINGS["learning_rate"],
        weight_decay=OTHER_SETTINGS["weight_decay"]
    )

    tokenizer = tiktoken.get_encoding("gpt2")  # 加载GPT-2使用的BPE分词器，用于文本编解码

    # 调用第5章实现的简化训练函数，执行训练+验证循环
    # eval_freq=1 表示每训练1个batch就做一次评估，eval_iter=1 表示评估时只用1个batch估算loss
    # start_context 用于在训练过程中生成样例文本，观察模型生成效果（不影响本测试的断言）
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, one_batch_train_loader, one_batch_val_loader, optimizer, device,
        num_epochs=OTHER_SETTINGS["num_epochs"], eval_freq=1, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # 断言1：固定随机种子下，第一次记录的训练loss应约为7.6（保留1位小数），
    # 用于回归检测——一旦模型结构、初始化方式或训练逻辑被意外改动，这里会失败提醒开发者
    assert round(train_losses[0], 1) == 7.6
    # 断言2：固定随机种子下，第一次记录的验证loss应约为10.1（保留1位小数），同样用于回归检测
    assert round(val_losses[0], 1) == 10.1
    # 断言3：验证训练确实在生效——最后一次记录的训练loss应低于第一次记录的训练loss，
    # 证明优化器在持续更新参数并降低了损失
    assert train_losses[-1] < train_losses[0]
