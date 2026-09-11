# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明：
本文件是针对《从零构建大语言模型》一书附录 D（Appendix D）中
`train_model` 训练函数（学习率预热 + 余弦退火调度、梯度裁剪等训练技巧）
的 pytest 单元测试文件。

测试整体思路：
1. 下载（或复用已下载的）《The Verdict》短篇小说文本作为训练语料；
2. 构建一个较小规模的 GPT 模型（GPTModel）；
3. 用 ch02 中的 `create_dataloader_v1` 构造训练/验证数据加载器；
4. 从中各取出一个 batch，跑若干个 epoch 的训练；
5. 断言初始训练/验证损失符合预期，并且训练损失在训练结束后确实下降，
   以此验证 `train_model` 的训练流程和学习率调度逻辑是可用、正确的。
"""

from llms_from_scratch.ch02 import create_dataloader_v1
from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.appendix_d import train_model

import os
import urllib

import tiktoken
import torch
from torch.utils.data import Subset, DataLoader


def test_train(tmp_path):
    """测试用例：验证附录 D 中 train_model 的训练流程能正常运行且损失下降。

    参数：
        tmp_path: pytest 内置夹具（fixture），提供一个仅本次测试可用的
            临时目录路径对象，用于存放下载下来的训练文本文件，
            测试结束后该临时目录会被 pytest 自动清理。

    验证点：
        - 第一个 batch 的训练损失、验证损失应接近书中给出的参考值；
        - 训练结束后的损失应低于训练开始时的损失（即模型确实在学习）。
    """

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 2,
        "batch_size": 1,
        "weight_decay": 0.1
    }

    # 固定随机种子，保证模型权重初始化、数据打乱等随机过程可复现，
    # 使得下面对损失值的断言（round(...) == 10.9 等）是稳定可靠的。
    torch.manual_seed(123)
    # 优先使用 GPU（CUDA），若不可用则回退到 CPU 运行。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Download data if necessary
    ##############################

    # 训练语料文件路径：位于 pytest 提供的临时目录下，命名为 the-verdict.txt
    file_path = tmp_path / "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        # 临时目录中还没有该文件，说明是首次运行，需要从远程 URL 下载文本
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        # 文件已存在（例如同一测试会话内被复用），直接从本地读取，避免重复下载
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    # 按照上面定义的（缩小版）配置初始化一个 GPT 模型
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    # 训练集占比 90%，剩余 10% 作为验证集
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    # 构造训练集 DataLoader：使用文本前 90% 部分，滑动窗口步幅等于上下文长度
    # （即窗口之间不重叠），drop_last=True 丢弃不足一个 batch 的尾部数据，
    # shuffle=True 打乱样本顺序以增强训练效果
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=OTHER_SETTINGS["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    # 构造验证集 DataLoader：使用文本后 10% 部分，不丢弃尾部数据、不打乱顺序，
    # 以便验证损失的评估结果稳定可复现
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=OTHER_SETTINGS["batch_size"],
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################

    # 加载 GPT-2 使用的 BPE 分词器，用于训练过程中生成示例文本以观察效果
    tokenizer = tiktoken.get_encoding("gpt2")

    # 为了让测试快速运行，这里只从训练/验证 DataLoader 的底层 Dataset 中
    # 各取出第 1 个样本（range(1)），重新包装成只含 1 个 batch 的 DataLoader，
    # 从而避免在测试中跑完整个数据集导致耗时过长
    train_subset = Subset(train_loader.dataset, range(1))
    one_batch_train_loader = DataLoader(train_subset, batch_size=1)
    val_subset = Subset(val_loader.dataset, range(1))
    one_batch_val_loader = DataLoader(val_subset, batch_size=1)

    peak_lr = 0.001  # this was originally set to 5e-4 in the book by mistake
    # 使用 AdamW 优化器，peak_lr 是学习率预热阶段结束后达到的峰值学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)  # the book accidentally omitted the lr assignment
    # 注：此处重复获取一次分词器（与上面的 tokenizer 变量重复赋值），
    # 保留原书/原代码写法，不做修改
    tokenizer = tiktoken.get_encoding("gpt2")

    n_epochs = 6
    warmup_steps = 1

    # 调用附录 D 中实现的 train_model 函数执行训练：
    # 内部包含学习率预热（warmup_steps 步内线性升到 peak_lr，再由 initial_lr/min_lr
    # 控制起止学习率并做余弦退火）、梯度裁剪等训练技巧；
    # eval_freq/eval_iter 控制评估频率与评估时使用的 batch 数量；
    # start_context 用于在训练过程中生成示例文本，直观展示模型效果。
    # 返回值分别为：每次记录的训练损失列表、验证损失列表、
    # 已见过的 token 数量列表、每一步的学习率列表。
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, one_batch_train_loader, one_batch_val_loader, optimizer, device, n_epochs=n_epochs,
        eval_freq=5, eval_iter=1, start_context="Every effort moves you",
        tokenizer=tokenizer, warmup_steps=warmup_steps,
        initial_lr=1e-5, min_lr=1e-5
    )

    # 断言：由于固定了随机种子，模型初始化和数据是确定的，
    # 因此第一次评估得到的训练损失应约等于 10.9（保留 1 位小数后相等）
    assert round(train_losses[0], 1) == 10.9
    # 断言：同理，第一次评估得到的验证损失应约等于 11.0
    assert round(val_losses[0], 1) == 11.0
    # 断言：经过若干个 epoch 的训练后，最终训练损失应低于初始训练损失，
    # 用以验证模型确实在学习（损失在下降）
    assert train_losses[-1] < train_losses[0]
