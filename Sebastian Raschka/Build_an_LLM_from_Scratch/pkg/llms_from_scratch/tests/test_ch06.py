# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是第 6 章（文本分类/垃圾邮件分类微调）相关功能的 pytest 单元测试文件。

测试内容概述：
- 下载并解压垃圾短信分类数据集（SMS Spam Collection）；
- 构造类别均衡的数据集，并将其划分为训练集/验证集/测试集；
- 使用 tiktoken 的 GPT-2 分词器构造 SpamDataset 数据集与对应的 DataLoader；
- 加载一个用于测试的小型 GPTModel，并将其改造为二分类模型（替换输出头）；
- 冻结大部分参数，仅对最后一个 Transformer 块和最终归一化层解冻并进行微调；
- 调用 train_classifier_simple 在极小的数据子集上跑若干轮训练，
  并对训练/验证损失做基本的数值断言，验证训练流程能够正常收敛。
"""


from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch06 import (
    download_and_unzip_spam_data, create_balanced_dataset,
    random_split, SpamDataset, train_classifier_simple
)

from pathlib import Path

import requests
import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Subset


def test_train_classifier(tmp_path):
    """
    端到端集成测试：验证垃圾短信分类器的完整微调流程能够正常运行。

    该测试会：
    1. 下载（或从备用地址下载）并解压 SMS 垃圾短信数据集；
    2. 构造类别均衡的数据集并划分为训练/验证/测试集，写入临时目录；
    3. 基于 GPT-2 分词器构建 SpamDataset 及对应的 DataLoader；
    4. 加载一个参数量很小的 GPTModel（仅用于测试，非真实预训练权重）；
    5. 将模型改造为二分类模型，并只解冻最后一个 Transformer 块与最终归一化层；
    6. 在极小的数据子集（各 5 条样本）上运行若干个 epoch 的微调；
    7. 断言训练/验证损失的初始值与收敛趋势符合预期，以确认训练逻辑无误。

    参数：
        tmp_path: pytest 内置夹具，提供一个函数级别的临时目录路径，
                  用于存放下载的压缩包、解压后的数据及切分后的 csv 文件。
    """

    ########################################
    # Download and prepare dataset
    ########################################

    # 数据集的主下载地址（UCI 机器学习仓库）
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    # 压缩包在临时目录中的保存路径
    zip_path = tmp_path / "sms_spam_collection.zip"
    # 解压后数据所在的目录
    extracted_path = tmp_path / "sms_spam_collection"
    # 解压后具体的数据文件（tsv 格式：标签 + 文本）
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    try:
        # 优先尝试从主 URL 下载并解压数据集
        download_and_unzip_spam_data(
            url, zip_path, extracted_path, data_file_path
        )
    except (requests.exceptions.RequestException, TimeoutError) as e:
        # 主地址下载失败（网络异常或超时）时，打印提示并回退到备用地址重试
        print(f"Primary URL failed: {e}. Trying backup URL...")
        backup_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(
            backup_url, zip_path, extracted_path, data_file_path
        )

    # 读取 tsv 文件为 DataFrame，指定分隔符为制表符，并手动指定列名
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    # 对原始数据进行下采样，使 ham（正常短信）与 spam（垃圾短信）数量均衡
    balanced_df = create_balanced_dataset(df)
    # 将文字标签映射为数值标签：ham -> 0，spam -> 1，便于后续训练
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # 按 70% 训练 / 10% 验证 / 剩余 20% 测试的比例切分数据集
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # 将切分后的三个子集分别写入临时目录下的 csv 文件，供 SpamDataset 读取
    train_df.to_csv(tmp_path / "train.csv", index=None)
    validation_df.to_csv(tmp_path / "validation.csv", index=None)
    test_df.to_csv(tmp_path / "test.csv", index=None)

    ########################################
    # Create data loaders
    ########################################
    # 使用 GPT-2 的字节对编码（BPE）分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 构建训练集：max_length=None 表示由数据集内部自动计算最长序列长度并据此做 padding
    train_dataset = SpamDataset(
        csv_file=tmp_path / "train.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    # 构建验证集：复用训练集计算出的 max_length，保证输入维度一致
    val_dataset = SpamDataset(
        csv_file=tmp_path / "validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0  # 数据加载子进程数量，测试环境中设为 0 避免多进程开销
    batch_size = 8   # 每个批次的样本数量

    # 固定随机种子，保证 DataLoader 的 shuffle 行为可复现
    torch.manual_seed(123)

    # 训练集 DataLoader：开启随机打乱，并丢弃不满一个 batch 的尾部数据
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # 验证集 DataLoader：不打乱顺序，也不丢弃最后不满 batch 的数据
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ########################################
    # Load pretrained model
    ########################################

    # Small GPT model for testing purposes
    # 用于测试的小型 GPT 模型配置（并非真实预训练权重，仅验证训练流程正确性）
    BASE_CONFIG = {
        "vocab_size": 50257,     # 词表大小，与 GPT-2 分词器保持一致
        "context_length": 120,   # 支持的最大上下文长度
        "drop_rate": 0.0,        # dropout 比例，测试中关闭
        "qkv_bias": False,       # 注意力层的 Q/K/V 线性层是否使用偏置
        "emb_dim": 12,           # 极小的嵌入维度，加快测试速度
        "n_layers": 1,           # 仅使用 1 层 Transformer 块
        "n_heads": 2             # 注意力头数
    }
    # 实例化 GPT 模型
    model = GPTModel(BASE_CONFIG)
    # 切换为评估模式（此时尚未开始训练，先设为 eval 以关闭 dropout 等训练态行为）
    model.eval()
    device = "cpu"  # 测试环境使用 CPU 即可，无需 GPU

    ########################################
    # Modify and pretrained model
    ########################################

    # 先冻结模型全部参数，默认不参与梯度更新（即不会被微调）
    for param in model.parameters():
        param.requires_grad = False

    # 固定随机种子，保证新初始化的输出头权重可复现
    torch.manual_seed(123)

    num_classes = 2  # 二分类任务：ham / spam
    # 将原本用于语言建模的输出头替换为二分类线性层
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    # 将模型移动到指定设备（此处为 CPU）
    model.to(device)

    # 解冻最后一个 Transformer 块的参数，使其参与微调
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    # 解冻最终归一化层的参数，使其参与微调
    for param in model.final_norm.parameters():
        param.requires_grad = True

    ########################################
    # Finetune modified model
    ########################################

    # 固定随机种子，保证优化过程（如参数初始化相关的随机性）可复现
    torch.manual_seed(123)

    # 使用 AdamW 优化器，仅会更新 requires_grad=True 的参数（输出头 + 最后一个 block + 最终归一化层）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)

    # 为加快测试速度，仅取训练集的前 5 条样本构造一个极小的子集
    train_subset = Subset(train_loader.dataset, range(5))
    batch_train_loader = DataLoader(train_subset, batch_size=5)
    # 同样地，仅取验证集的前 5 条样本
    val_subset = Subset(val_loader.dataset, range(5))
    batch_val_loader = DataLoader(val_subset, batch_size=5)

    num_epochs = 5  # 在极小数据集上训练 5 个 epoch，用于验证训练循环是否收敛
    # 调用训练函数，返回训练/验证损失、训练/验证准确率的历史记录，以及累计已见样本数
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, batch_train_loader, batch_val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=1, eval_iter=1,
    )

    # 断言：第一次评估时的训练损失应约为 0.8（验证初始损失符合预期基准）
    assert round(train_losses[0], 1) == 0.8
    # 断言：第一次评估时的验证损失应约为 0.8（同上，验证初始状态一致）
    assert round(val_losses[0], 1) == 0.8
    # 断言：经过若干轮微调后，最终训练损失应低于初始训练损失，说明模型确实在学习/收敛
    assert train_losses[-1] < train_losses[0]
