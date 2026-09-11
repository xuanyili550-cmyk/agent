# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是第 7 章（指令微调，Instruction Finetuning）配套代码的 pytest 单元测试。

测试目标：
    验证从「下载指令数据集 -> 构建 Dataset/DataLoader -> 加载一个小型 GPT 模型 ->
    使用 train_model_simple 进行指令微调」这一完整流程能够正常跑通，并且
    微调过程中的训练/验证损失符合预期的数值区间（用于回归测试，防止代码改动
    导致训练流程或数值结果出现意外偏差）。

依赖的被测对象：
    - llms_from_scratch.ch04.GPTModel：第 4 章实现的 GPT 模型结构。
    - llms_from_scratch.ch05.train_model_simple：第 5 章实现的简单训练循环。
    - llms_from_scratch.ch07 中的 download_and_load_file / InstructionDataset /
      format_input / custom_collate_fn：第 7 章实现的指令数据下载、数据集封装、
      指令格式化以及自定义 batch 拼接（collate）函数。
"""

from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import train_model_simple
from llms_from_scratch.ch07 import (
    download_and_load_file, InstructionDataset, format_input, custom_collate_fn
)

from functools import partial

import torch
from torch.utils.data import DataLoader
import tiktoken


def test_instruction_finetune(tmp_path):
    """
    端到端测试：指令数据集下载 + 构建 DataLoader + 小型 GPT 模型微调。

    参数：
        tmp_path: pytest 内置夹具（fixture），提供一个仅在本次测试中有效的
            临时目录路径，用于存放下载下来的指令数据集 JSON 文件，测试结束后
            会自动清理，避免污染文件系统。

    测试流程：
        1. 下载并加载指令微调数据集，按 85% / 10% / 5% 比例切分为
           训练集 / 测试集 / 验证集，并各自截取前 15 条样本以加快测试速度。
        2. 使用 tiktoken 的 gpt2 编码器，将数据封装为 InstructionDataset，
           并通过 custom_collate_fn 构建训练/验证用的 DataLoader。
        3. 构建一个参数规模很小的 GPTModel（仅用于测试，非真实预训练模型）。
        4. 使用 AdamW 优化器，调用 train_model_simple 对模型进行少量轮次的
           微调训练。
        5. 断言训练/验证过程中的损失值符合预期，用以验证训练流程的正确性
           及数值稳定性（回归测试）。
    """

    #######################################
    # Download and prepare dataset
    #######################################
    file_path = tmp_path / "instruction-data.json"  # 临时目录下的数据集文件保存路径
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)  # 下载（若已存在则直接读取）并解析 JSON 数据

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # Use very small subset for testing purposes
    # 为了让单元测试运行更快，这里只取每个子集的前 15 条样本，而不是完整数据集
    train_data = train_data[:15]
    val_data = val_data[:15]
    test_data = test_data[:15]

    tokenizer = tiktoken.get_encoding("gpt2")  # 使用 GPT-2 的 BPE 分词器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU，否则回退到 CPU

    # 使用 functools.partial 预先绑定 device 与 allowed_max_length 参数，
    # 生成一个可直接作为 DataLoader collate_fn 使用的函数
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=100)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)  # 固定随机种子，保证 DataLoader 的 shuffle 行为可复现

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # Load pretrained model
    #######################################

    # Small GPT model for testing purposes
    # 用于测试的极小规模 GPT 配置（真实预训练模型的超参数会大得多），
    # 目的只是验证微调流程能跑通，而不是追求模型效果
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 120,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "emb_dim": 12,
        "n_layers": 1,
        "n_heads": 2
    }
    model = GPTModel(BASE_CONFIG)
    model.eval()  # 先设为 eval 模式（后续训练循环内部会按需切换 train/eval）
    device = "cpu"  # 测试统一在 CPU 上运行，避免依赖 GPU 环境
    CHOOSE_MODEL = "Small test model"

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    #######################################
    # Finetuning the model
    #######################################

    num_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    torch.manual_seed(123)  # 再次固定随机种子，保证训练过程（如参数初始化后的更新）可复现
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    # 断言训练刚开始时的初始训练损失约为 10.9（保留一位小数），
    # 用于验证模型初始化与前向计算的数值是否符合预期
    assert round(train_losses[0], 1) == 10.9
    # 断言训练刚开始时的初始验证损失约为 10.9（保留一位小数）
    assert round(val_losses[0], 1) == 10.9
    # 断言经过若干轮训练后，最后一次记录的训练损失应低于初始训练损失，
    # 即模型确实在学习（损失在下降）
    assert train_losses[-1] < train_losses[0]
