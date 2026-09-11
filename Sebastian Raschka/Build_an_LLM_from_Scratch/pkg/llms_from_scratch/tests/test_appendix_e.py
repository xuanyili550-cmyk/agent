# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是附录 E（Appendix E：使用 LoRA 进行参数高效微调）对应的 pytest 单元测试。

测试目的：验证 `replace_linear_with_lora` 能够正确地把 GPT 模型中的
`torch.nn.Linear` 层替换为 LoRA（Low-Rank Adaptation，低秩自适应）线性层，
并且在这种“冻结原始权重 + 只训练 LoRA 低秩矩阵（以及部分解冻的最后一层
Transformer block 和 final_norm）”的设置下，模型仍然可以像第 6 章的
垃圾邮件分类微调任务那样正常训练、损失正常下降。

测试流程概览：
1. 下载并解压 SMS 垃圾邮件分类数据集（若主 URL 失败则回退到备份 URL）；
2. 构造平衡数据集，并划分为训练/验证/测试集；
3. 使用 GPT-2 分词器构建 SpamDataset 与 DataLoader；
4. 加载一个用于测试的小型 GPTModel，并冻结全部参数；
5. 替换分类头，并用 `replace_linear_with_lora` 把线性层替换为 LoRA 版本；
6. 解冻最后一个 Transformer block 和 final_norm 的参数，允许微调；
7. 使用极小的子集（5 条样本）跑若干个 epoch 的训练，验证初始损失和
   损失下降趋势是否符合预期。
"""


from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch06 import (
    download_and_unzip_spam_data, create_balanced_dataset,
    random_split, SpamDataset, train_classifier_simple
)
from llms_from_scratch.appendix_e import replace_linear_with_lora

from pathlib import Path

import pandas as pd
import requests
import tiktoken
import torch
from torch.utils.data import DataLoader, Subset


def test_train_classifier_lora(tmp_path):
    """
    测试要点：
    - 使用一个极小的 GPTModel（emb_dim=12, n_layers=1, n_heads=2）作为“预训练模型”的替身，
      以保证测试在 CPU 上快速运行；
    - 通过 `replace_linear_with_lora` 将模型中所有 nn.Linear 层替换为带有
      LoRA 低秩适配器的线性层（rank=16, alpha=16），验证 LoRA 微调流程可以跑通；
    - 仅解冻最后一个 Transformer block 和 final_norm 的参数（加上 LoRA 自身的参数），
      模拟“大部分参数冻结、仅小部分参数可训练”的高效微调场景；
    - 用极小的数据子集（5 条训练样本、5 条验证样本）训练 6 个 epoch，
      断言初始训练/验证损失约为 0.8，且训练损失在训练结束后低于初始值，
      从而验证训练确实在“学习”，而不是报错或者损失不降反升。

    参数：
        tmp_path: pytest 内置夹具，提供一个测试专用的临时目录，
                  用于存放下载的数据集压缩包、解压后的文件以及切分出的 csv 文件。
    """

    ########################################
    # Download and prepare dataset
    # 下载并准备数据集
    ########################################

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = tmp_path / "sms_spam_collection.zip"
    extracted_path = tmp_path / "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    try:
        # 优先尝试从主 URL（UCI 数据集仓库）下载并解压数据
        download_and_unzip_spam_data(
            url, zip_path, extracted_path, data_file_path
        )
    except (requests.exceptions.RequestException, TimeoutError) as e:
        # 若主 URL 下载失败（网络异常或超时），打印提示后改用备份 URL 重新下载
        print(f"Primary URL failed: {e}. Trying backup URL...")
        backup_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(
            backup_url, zip_path, extracted_path, data_file_path
        )

    # 读取制表符分隔的原始数据文件，并指定列名为 Label（标签）和 Text（短信文本）
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    # 对原始数据做类别平衡处理（使 ham/spam 样本数量一致，避免类别不平衡影响训练）
    balanced_df = create_balanced_dataset(df)
    # 将文本标签映射为数值标签：ham（正常邮件）-> 0, spam（垃圾邮件）-> 1
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # 按 70% / 10% / 20%（剩余部分）比例划分为训练集、验证集、测试集
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # 将划分好的数据集分别保存为 csv 文件，供后续 SpamDataset 读取
    train_df.to_csv(tmp_path / "train.csv", index=None)
    validation_df.to_csv(tmp_path / "validation.csv", index=None)
    test_df.to_csv(tmp_path / "test.csv", index=None)

    ########################################
    # Create data loaders
    # 创建数据加载器
    ########################################
    # 使用 GPT-2 的 BPE 分词器对文本进行编码
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(
        csv_file=tmp_path / "train.csv",
        max_length=None,  # 不限制最大长度，由数据集内部自动确定（并作为验证集的长度基准）
        tokenizer=tokenizer
    )

    val_dataset = SpamDataset(
        csv_file=tmp_path / "validation.csv",
        max_length=train_dataset.max_length,  # 与训练集保持一致的最大长度，便于批处理对齐
        tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8

    # 固定随机种子，保证 DataLoader 的 shuffle 行为可复现
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,  # 丢弃最后一个不满 batch_size 的批次，保证批次大小一致
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,  # 验证集不需要丢弃最后一个批次
    )

    ########################################
    # Load pretrained model
    # 加载“预训练”模型（此处为测试用的小模型）
    ########################################

    # Small GPT model for testing purposes
    # 用于测试目的的小型 GPT 模型配置（维度和层数都很小，以加快测试速度）
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
    model.eval()  # 切换为评估模式（本测试中主要影响 dropout 等，drop_rate 为 0 影响不大）
    device = "cpu"

    ########################################
    # Modify and pretrained model
    # 修改预训练模型（替换分类头、应用 LoRA）
    ########################################

    # 冻结模型所有原始参数，使其在微调阶段不被更新（体现参数高效微调思想）
    for param in model.parameters():
        param.requires_grad = False

    # 固定随机种子，保证分类头初始化及后续操作可复现
    torch.manual_seed(123)

    num_classes = 2
    # 将输出头替换为二分类线性层（垃圾邮件 vs 正常邮件）
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    # 核心测试对象：把模型中的所有 nn.Linear 替换为带 LoRA 低秩适配器的线性层
    # rank=16 表示低秩矩阵的秩，alpha=16 为 LoRA 缩放系数
    replace_linear_with_lora(model, rank=16, alpha=16)
    model.to(device)

    # 解冻最后一个 Transformer block 的参数，使其在微调阶段可训练
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    # 解冻最终层归一化（final_norm）的参数，使其在微调阶段可训练
    for param in model.final_norm.parameters():
        param.requires_grad = True

    ########################################
    # Finetune modified model
    # 微调修改后的模型
    ########################################

    # 固定随机种子，保证优化器及训练过程的可复现性
    torch.manual_seed(123)

    # 使用 AdamW 优化器，仅会更新 requires_grad=True 的参数（LoRA 参数 + 最后一层 + final_norm）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    # 为了让测试快速运行，仅取训练集/验证集的前 5 条样本构造极小规模的 DataLoader
    train_subset = Subset(train_loader.dataset, range(5))
    batch_train_loader = DataLoader(train_subset, batch_size=5)
    val_subset = Subset(val_loader.dataset, range(5))
    batch_val_loader = DataLoader(val_subset, batch_size=5)

    num_epochs = 6
    # 调用第 6 章实现的简易分类器训练函数，在小数据集上训练若干个 epoch，
    # 返回训练/验证损失、训练/验证准确率以及累计已见样本数
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, batch_train_loader, batch_val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=1, eval_iter=1,
    )

    # 断言：初始训练损失约为 0.8（保留一位小数比较），验证训练流程的数值符合预期基准
    assert round(train_losses[0], 1) == 0.8
    # 断言：初始验证损失约为 0.8，与训练损失基准一致
    assert round(val_losses[0], 1) == 0.8
    # 断言：训练结束时的损失应低于初始损失，证明模型（含 LoRA 参数）确实在学习
    assert train_losses[-1] < train_losses[0]
