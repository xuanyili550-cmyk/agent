# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块说明（中文）：
本文件对应《从零构建大语言模型》（Build a Large Language Model From Scratch）第 5 章
"Pretraining on Unlabeled Data"（在无标签数据上进行预训练）的主代码。

它演示了训练一个 GPT 风格语言模型所需的完整最小闭环，具体包括：
1. 文本与 token id 之间的相互转换（配合 tiktoken 的 GPT-2 分词器）；
2. 基于交叉熵损失的语言建模损失计算（单个 batch 与整个 DataLoader 上的平均损失）；
3. 训练过程中的周期性评估（在训练集/验证集上分别计算损失，用于监控过拟合）；
4. 一个简单但完整的训练循环 train_model_simple（前向传播 -> 反向传播 -> 参数更新）；
5. 训练过程中定期生成样例文本，直观感受模型能力的变化；
6. 训练完成后绘制训练/验证损失曲线，并保存/加载模型权重。

本文件中使用的 GPTModel、create_dataloader_v1、generate_text_simple 等均从
previous_chapters.py（对应前几章的代码）导入，是第 2~4 章内容的延续和整合，
第 5 章的重点是"训练"本身，而不是模型结构。
"""

import matplotlib.pyplot as plt
import os
import requests
import torch
import tiktoken


# Import from local files
# 从本地文件导入前几章已经实现好的组件：
# - GPTModel：第 4 章搭建的 GPT 模型结构（多层 Transformer Block + 输出头）
# - create_dataloader_v1：第 2 章实现的滑动窗口式数据加载器（把长文本切成 (输入, 目标) 样本对）
# - generate_text_simple：第 4 章实现的贪心解码文本生成函数
from previous_chapters import GPTModel, create_dataloader_v1, generate_text_simple


def text_to_token_ids(text, tokenizer):
    """将原始文本编码为模型可接受的 token id 张量。

    参数:
        text (str): 待编码的原始文本。
        tokenizer: 分词器对象（如 tiktoken 的 GPT-2 编码器），需实现 encode 方法。

    返回:
        torch.Tensor: 形状为 (1, seq_len) 的整型张量，其中 1 是 batch 维度，
        seq_len 是编码后 token 的数量。
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    # unsqueeze(0) 在最前面插入一个 batch 维度：(seq_len,) -> (1, seq_len)
    # 因为模型的前向传播默认接收 (batch_size, seq_len) 形状的输入
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将模型输出/输入的 token id 张量解码回可读文本。

    参数:
        token_ids (torch.Tensor): 形状为 (1, seq_len) 的 token id 张量（batch_size=1）。
        tokenizer: 分词器对象，需实现 decode 方法。

    返回:
        str: 解码后的文本字符串。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    # squeeze(0) 去掉 batch 维度：(1, seq_len) -> (seq_len,)，方便转成 list 后解码
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个 batch 上的语言建模（下一个 token 预测）交叉熵损失。

    参数:
        input_batch (torch.Tensor): 形状 (batch_size, seq_len) 的输入 token id。
        target_batch (torch.Tensor): 形状 (batch_size, seq_len) 的目标 token id
            （即 input_batch 整体右移一位，用于"预测下一个词"任务）。
        model (torch.nn.Module): GPT 模型。
        device (torch.device): 计算设备（"cpu" 或 "cuda"）。

    返回:
        torch.Tensor: 标量张量，表示该 batch 的平均交叉熵损失。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 将数据搬到与模型相同的设备上（CPU/GPU），否则会因设备不一致报错
    logits = model(input_batch)
    # logits 形状: (batch_size, seq_len, vocab_size)，即对每个位置预测词表上每个 token 的得分
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    # flatten(0, 1) 把 (batch_size, seq_len, vocab_size) 压平为 (batch_size*seq_len, vocab_size)
    # target_batch.flatten() 把 (batch_size, seq_len) 压平为 (batch_size*seq_len,)
    # 这样交叉熵就是把"每个位置的下一个 token 预测"当作一个独立的多分类问题来计算
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """在给定 DataLoader 上计算平均损失，可选只取前 num_batches 个 batch 以加速评估。

    参数:
        data_loader (DataLoader): 训练或验证用的数据加载器。
        model (torch.nn.Module): GPT 模型。
        device (torch.device): 计算设备。
        num_batches (int, 可选): 参与计算平均损失的 batch 数量；
            为 None 时使用整个 data_loader。

    返回:
        float: 平均损失；若 data_loader 为空则返回 float("nan")。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 取 num_batches 与实际可用 batch 数的较小值，避免越界
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # .item() 从标量张量中取出 Python float，避免累积计算图占用显存
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """在训练/验证集上分别评估模型当前的损失，用于训练过程中监控收敛与过拟合情况。

    参数:
        model (torch.nn.Module): GPT 模型。
        train_loader (DataLoader): 训练集数据加载器。
        val_loader (DataLoader): 验证集数据加载器。
        device (torch.device): 计算设备。
        eval_iter (int): 评估时各自采样的 batch 数量（而非遍历整个数据集，节省时间）。

    返回:
        tuple[float, float]: (train_loss, val_loss)
    """
    model.eval()
    # 切换到评估模式：关闭 Dropout、BatchNorm 等使用训练统计量的行为（本模型主要影响 Dropout 层）
    with torch.no_grad():
        # 评估阶段不需要反向传播，用 no_grad 关闭梯度追踪，节省显存和计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    # 评估结束后切回训练模式，确保后续训练时 Dropout 等重新生效
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """给定一段起始文本，让模型自回归生成后续内容并打印，用于直观检查训练效果。

    参数:
        model (torch.nn.Module): GPT 模型。
        tokenizer: 分词器对象。
        device (torch.device): 计算设备。
        start_context (str): 用于起始生成的提示文本（prompt）。

    返回:
        None（直接打印生成结果）。
    """
    model.eval()
    # 生成时同样要关闭 Dropout 等随机性，保证输出确定、稳定
    context_size = model.pos_emb.weight.shape[0]
    # model.pos_emb 是位置编码 Embedding 层，其权重形状为 (context_length, emb_dim)
    # 取 shape[0] 即拿到模型支持的最大上下文长度，防止生成时序列超出位置编码范围
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        # 生成过程只需要前向传播，无需梯度，节省显存
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        # generate_text_simple 内部会做自回归解码：
        # 每步用当前序列预测下一个 token（取 logits 最大值，即贪心解码），
        # 并将其拼接到序列末尾，重复 max_new_tokens 次
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
    # 生成结束后切回训练模式，避免影响后续的训练循环


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """最简单版本的 GPT 训练循环：标准的"前向 -> 反向 -> 更新参数"过程，
    并在训练中周期性地评估损失、打印生成样例。

    参数:
        model (torch.nn.Module): 待训练的 GPT 模型。
        train_loader (DataLoader): 训练集数据加载器。
        val_loader (DataLoader): 验证集数据加载器。
        optimizer (torch.optim.Optimizer): 优化器（如 AdamW）。
        device (torch.device): 计算设备。
        num_epochs (int): 训练的总轮数（epoch）。
        eval_freq (int): 每隔多少个训练 step 评估一次损失。
        eval_iter (int): 每次评估时使用的 batch 数量。
        start_context (str): 每轮训练结束后用于生成样例文本的提示语。
        tokenizer: 分词器对象。

    返回:
        tuple[list, list, list]: (train_losses, val_losses, track_tokens_seen)
            分别是各次评估时刻的训练损失、验证损失，以及累计已处理的 token 数量，
            可用于后续绘制损失曲线。
    """
    # Initialize lists to track losses and tokens seen
    # 用列表记录训练过程中的关键指标，便于训练结束后可视化分析
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0  # 累计已经"看过"（训练过）的 token 总数
    global_step = -1  # 全局训练步数计数器，从 -1 开始，第一次 +1 后即为 0

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # 训练模式下会启用 Dropout 等正则化机制

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            # 每个 batch 开始前清空上一步残留的梯度，否则梯度会在多个 batch 间累加
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            # 反向传播：根据损失对模型所有可训练参数计算梯度（存储在 param.grad 中）
            optimizer.step()  # Update model weights using loss gradients
            # 优化器根据梯度更新模型参数（AdamW 会同时应用权重衰减）
            tokens_seen += input_batch.numel()
            # numel() 返回 batch_size * seq_len，即该 batch 中总 token 数量，用于统计训练量
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                # 每训练 eval_freq 步评估一次，而不是每步都评估，减少额外开销
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        # 每个 epoch 结束后生成一段样例文本，直观感受模型生成能力随训练的变化
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制训练/验证损失曲线，横轴同时展示"训练轮数"和"已处理 token 数量"两种刻度。

    参数:
        epochs_seen (torch.Tensor 或 list): 与各次评估对应的 epoch 数（可为小数，插值得到）。
        tokens_seen (list): 与各次评估对应的累计 token 数量。
        train_losses (list): 各次评估时的训练损失。
        val_losses (list): 各次评估时的验证损失。

    返回:
        None（图像通过 fig.tight_layout() 整理布局，由调用方决定展示或保存）。
    """
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    # 主坐标轴：以 epoch 数为横轴，绘制训练损失与验证损失
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    # 创建共享 y 轴、独立 x 轴的第二个坐标轴，用于展示"已训练 token 数量"这一维度
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    # 用 alpha=0（完全透明）画一条不可见的曲线，目的仅仅是让 matplotlib
    # 根据 tokens_seen 的数值范围自动生成对齐的刻度，而不会在图上产生多余的可见线条
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings):
    """训练流程的总入口：下载/加载数据 -> 初始化模型与优化器 -> 构建数据加载器 -> 执行训练。

    参数:
        gpt_config (dict): GPT 模型结构超参数（词表大小、上下文长度、嵌入维度、
            注意力头数、层数、Dropout 比例、qkv 是否带 bias 等）。
        settings (dict): 训练相关超参数（学习率、训练轮数、batch 大小、权重衰减）。

    返回:
        tuple: (train_losses, val_losses, tokens_seen, model)
            分别是训练/验证损失记录列表、累计 token 数记录列表，以及训练完成的模型对象。
    """

    torch.manual_seed(123)
    # 固定随机种子，保证模型参数初始化、Dropout 掩码、数据打乱顺序等具有可复现性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 优先使用 GPU（如果可用），否则退回 CPU

    ##############################
    # Download data if necessary
    ##############################

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        # 本地不存在训练语料时，从远程仓库下载（该文件是本书第 2 章使用的示例短篇小说文本）
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        # 本地已存在则直接读取，避免重复下载
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    # 注意：nn.Module 的 .to(device) 是原地操作（in-place），不需要像张量那样重新赋值
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    # AdamW：在 Adam 基础上将权重衰减与梯度更新解耦，是训练 Transformer 类模型的常用优化器

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    # 按字符位置将文本切分为 90% 训练 / 10% 验证两部分（简单的顺序切分，非随机打乱）

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    # 训练集 DataLoader：
    # - max_length = context_length：每个样本序列长度等于模型上下文长度
    # - stride = context_length：滑动窗口步长等于窗口长度，即样本之间不重叠，充分利用数据且不重复
    # - drop_last=True：丢弃最后不满一个 batch 的数据，保证每个 batch 形状一致
    # - shuffle=True：训练时打乱样本顺序，有助于优化收敛

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    # 验证集 DataLoader：不丢弃末尾 batch、不打乱顺序，保证评估结果稳定可复现

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")
    # 使用与 GPT-2 完全一致的 BPE 分词器，保证词表大小（50257）与模型配置匹配

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
    # 调用前面定义的训练循环，实际执行多轮梯度下降训练

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        # 为了在本地/教学环境中更快训练和演示，这里把上下文长度从 GPT-2 原始的 1024 缩短为 256
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }
    # 这是 GPT-2 "124M"（约 1.24 亿参数）规模的标准配置

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }
    # 训练超参数：较小的 batch_size 是为了适应教学用的小规模文本数据和普通硬件

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)
    # 执行完整训练流程，得到训练好的模型以及训练/验证损失记录

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    # 用线性插值构造与 train_losses 等长的 epoch 坐标序列，
    # 因为评估是按 global_step 触发的，其对应的"epoch 小数值"并非整数
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")
    # 将损失曲线图保存为 PDF 文件，便于查看训练是否收敛、是否过拟合

    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    # 只保存模型参数（state_dict），而非整个模型对象，是 PyTorch 推荐的持久化方式
    model = GPTModel(GPT_CONFIG_124M)
    # 重新构建一个结构相同、参数随机初始化的模型实例
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    # 加载刚保存的参数，验证保存/加载流程可用；weights_only=True 出于安全考虑，
    # 只反序列化张量权重而不执行任意 Python 代码
