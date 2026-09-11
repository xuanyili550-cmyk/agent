# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明(docstring):

本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)一书
第 5 章附加内容(bonus)脚本:超参数网格搜索(hyperparameter grid search)。

功能概述:
- 定义一组待搜索的超参数网格(HPARAM_GRID),包括批大小、dropout 比例、
  学习率预热步数、权重衰减、峰值学习率、初始学习率、最小学习率、训练轮数等。
- 使用 itertools.product 生成所有超参数组合的笛卡尔积。
- 对每一组超参数组合:构建数据加载器、初始化 GPT 模型、使用 AdamW 优化器
  配合"线性预热 + 余弦退火"学习率调度策略进行训练,并在验证集上评估损失。
- 记录并输出验证损失最小的那组超参数配置(即最优超参数)。
- 支持通过 Ctrl+C(KeyboardInterrupt)提前终止搜索,并打印当前已知的最优结果。

原始英文注释均予以保留,新增内容为中文注释与中文 docstring。
"""

import itertools  # 用于生成超参数网格的笛卡尔积组合
import math  # 用于余弦退火学习率调度中的 cos/pi 计算
import os  # 用于处理文件路径(定位当前脚本所在目录、拼接数据文件路径)
import tiktoken  # OpenAI 的 BPE 分词器库,这里用于加载 GPT-2 编码器
import torch  # PyTorch 深度学习框架

# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from llms_from_scratch.ch02 import create_dataloader_v1  # 第2章实现的数据加载器构造函数 ｜ 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)
from llms_from_scratch.ch04 import GPTModel  # 第4章实现的 GPT 模型类 ｜ 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)


# Define a grid of hyperparameters to search over
# 定义待搜索的超参数网格:每个键对应一个候选值列表,后续会做笛卡尔积组合
HPARAM_GRID = {
    "batch_size": [2, 4, 8, 16],  # 批大小候选值
    "drop_rate": [0.0, 0.1, 0.2],  # dropout 比例候选值,用于模型正则化
    "warmup_iters": [10, 20, 30],  # 学习率线性预热(warmup)的迭代步数候选值
    "weight_decay": [0.1, 0.01, 0.0],  # AdamW 优化器的权重衰减系数候选值
    "peak_lr": [0.0001, 0.0005, 0.001, 0.005],  # 预热结束后达到的峰值学习率候选值
    "initial_lr": [0.00005, 0.0001],  # 预热阶段起始的学习率候选值
    "min_lr": [0.00005, 0.00001, 0.0001],  # 余弦退火阶段最终收敛到的最小学习率候选值
    "n_epochs": [5, 10, 15, 20, 25],  # 训练轮数(epoch 数)候选值
}


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算给定数据加载器(data_loader)上若干个批次的平均交叉熵损失。

    参数:
        data_loader: 数据加载器(DataLoader),迭代产出 (input_batch, target_batch) 张量对。
        model: GPT 模型实例,前向传播用于计算 logits。
        device: 运行设备("cpu" 或 "cuda"),用于将张量搬到对应设备上。
        num_batches (int, 可选): 需要计算损失的批次数量上限;为 None 时使用整个
            data_loader 的批次数;若指定值大于实际批次数,则取二者较小值。

    返回:
        float: 所取批次上的平均损失值;若 data_loader 为空,则返回 float("nan")。
    """
    total_loss = 0.  # 累加所有已处理批次的损失值
    if len(data_loader) == 0:
        return float("nan")  # 数据加载器为空时无法计算损失,返回 NaN 表示无效
    elif num_batches is None:
        num_batches = len(data_loader)  # 未指定时,使用全部批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 避免请求的批次数超过实际可用数量
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算单批次损失
            total_loss += loss.item()  # 累加标量损失值(item() 从张量取出 Python 数值)
        else:
            break  # 已达到所需批次数,提前退出循环以节省计算
    return total_loss / num_batches  # 返回平均损失


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算单个批次数据的交叉熵损失。

    参数:
        input_batch (torch.Tensor): 输入的 token id 序列,形状为 (batch_size, seq_len)。
        target_batch (torch.Tensor): 目标(下一个 token)id 序列,形状为 (batch_size, seq_len)。
        model: GPT 模型实例。
        device: 运行设备,用于将输入/目标张量搬移到对应设备。

    返回:
        torch.Tensor: 标量损失张量(交叉熵损失)。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据搬到指定设备(CPU/GPU)

    logits = model(input_batch)  # 前向传播,得到形状为 (batch_size, seq_len, vocab_size) 的 logits
    logits = logits.view(-1, logits.size(-1))  # 展平为 (batch_size*seq_len, vocab_size),便于逐 token 计算交叉熵
    loss = torch.nn.functional.cross_entropy(logits, target_batch.view(-1))  # target 同样展平为 (batch_size*seq_len,)
    return loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    在训练集和验证集上分别评估模型当前的损失,用于监控训练过程中的表现。

    参数:
        model: GPT 模型实例。
        train_loader: 训练集数据加载器。
        val_loader: 验证集数据加载器。
        device: 运行设备。
        eval_iter (int): 评估时使用的批次数量上限(避免评估整个数据集耗时过长)。

    返回:
        tuple(float, float): (train_loss, val_loss) 训练集与验证集上的平均损失。
    """
    model.eval()  # 切换到评估模式(关闭 dropout 等训练专用行为)
    with torch.no_grad():  # 评估阶段不需要计算梯度,节省显存/计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 评估结束后恢复训练模式,以便继续训练
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_iter, warmup_iters=10,
                initial_lr=3e-05, min_lr=1e-6):
    """
    训练 GPT 模型,采用"线性预热 + 余弦退火"的学习率调度策略。

    参数:
        model: 待训练的 GPT 模型实例。
        train_loader: 训练集数据加载器。
        val_loader: 验证集数据加载器。
        optimizer: 优化器实例(如 AdamW),其初始学习率被视为调度中的峰值学习率(max_lr)。
        device: 运行设备。
        n_epochs (int): 训练轮数。
        eval_iter (int): 训练结束后评估时使用的批次数量上限。
        warmup_iters (int, 默认 10): 学习率线性预热阶段的迭代步数。
        initial_lr (float, 默认 3e-05): 预热阶段起始学习率。
        min_lr (float, 默认 1e-6): 余弦退火阶段最终收敛到的最小学习率。

    返回:
        tuple(float, float): 训练结束后在训练集与验证集上评估得到的 (train_loss, val_loss)。
    """
    global_step = 0  # 全局迭代步数计数器,贯穿所有 epoch

    max_lr = optimizer.param_groups[0]["lr"]  # 将优化器初始设置的学习率作为预热阶段结束后的峰值学习率

    # Calculate total number of iterations
    # 计算总迭代步数 = 每个 epoch 的批次数 * 总 epoch 数,用于余弦退火进度计算
    total_training_iters = len(train_loader) * n_epochs

    # Calculate the learning rate increment at each step during warmup
    # 计算预热阶段每一步学习率的线性增量:从 initial_lr 线性增长到 max_lr,共 warmup_iters 步
    lr_increment = (optimizer.param_groups[0]["lr"] - initial_lr) / warmup_iters

    for epoch in range(n_epochs):
        model.train()  # 确保处于训练模式(启用 dropout 等)
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清空上一步累积的梯度

            # Increment the global step at the beginning of the iteration
            # 在本次迭代开始时递增全局步数(注意:先自增,故第一次迭代 global_step 从 1 开始)
            global_step += 1

            # Warmup: adjust learning rate linearly
            # 预热阶段:学习率随步数线性增长
            if global_step <= warmup_iters:
                lr = initial_lr + global_step * lr_increment
            # Cosine annealing phase
            # 余弦退火阶段:学习率按余弦曲线从 max_lr 平滑衰减到 min_lr
            else:
                progress = (global_step - warmup_iters) / (total_training_iters - warmup_iters)  # 退火阶段的进度比例,范围 [0,1]
                lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))  # 余弦退火公式

            # Apply the calculated learning rate
            # 将计算出的学习率应用到优化器的每一个参数组
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次的损失
            loss.backward()  # 反向传播,计算梯度

            # Apply gradient clipping
            # 预热结束后才进行梯度裁剪,避免预热阶段学习率较小时裁剪反而带来不必要的限制
            if global_step >= warmup_iters:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 将梯度范数裁剪到最大 1.0,防止梯度爆炸

            optimizer.step()  # 根据梯度更新模型参数

    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)  # 训练结束后做一次评估

    return train_loss, val_loss


if __name__ == "__main__":

    # Generate all combinations of hyperparameters
    # 生成 HPARAM_GRID 中所有超参数的笛卡尔积组合列表,每个元素是一个元组(按字典中键的顺序取值)
    hyperparameter_combinations = list(itertools.product(*HPARAM_GRID.values()))
    total_combinations = len(hyperparameter_combinations)  # 组合总数,用于打印搜索进度
    print(f"Total hyperparameter configurations: {total_combinations}")

    # Placeholder for the best loss and best hyperparameters
    # 用于记录当前搜索到的最优验证损失及对应超参数配置的占位变量
    best_val_loss = float("inf")  # 初始化为正无穷,保证第一次比较时必然被更新
    best_hparams = {}  # 保存最优超参数字典

    script_path = os.path.abspath(__file__)  # 获取当前脚本文件的绝对路径
    script_dir = os.path.dirname(script_path)  # 提取脚本所在目录,便于定位同目录下的数据文件
    with open(os.path.join(script_dir, "the-verdict.txt"), "r", encoding="utf-8") as file:
        text_data = file.read()  # 读取训练用的原始文本语料(小说《The Verdict》)

    tokenizer = tiktoken.get_encoding("gpt2")  # 加载与 GPT-2 兼容的 BPE 分词器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU,否则退回 CPU

    train_ratio = 0.95  # 训练集占比 95%,其余 5% 作为验证集
    split_idx = int(train_ratio * len(text_data))  # 计算训练/验证集划分的字符索引位置

    torch.manual_seed(123)  # 固定随机种子,保证整体流程可复现

    interrupted = False  # 标记搜索过程是否被用户手动中断(Ctrl+C)
    current_config = 0  # 当前正在评估的配置序号(从 1 开始计数,用于打印进度)
    for combination in hyperparameter_combinations:

        try:
            current_config += 1
            print(f"Evaluating configuration {current_config} of {total_combinations}")

            # Unpack the current combination of hyperparameters
            # 将当前元组组合与 HPARAM_GRID 的键一一对应,还原为可读的超参数字典
            HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(), combination))

            GPT_CONFIG_124M = {
                "vocab_size": 50257,    # Vocabulary size  # 词表大小,对应 GPT-2 的 BPE 词表
                "context_length": 256,  # Context length -- shortened from original 1024 tokens  # 上下文长度(相比原始 1024 做了缩短,便于快速搜索)
                "emb_dim": 768,         # Embedding dimension  # 词嵌入维度
                "n_heads": 12,          # Number of attention heads  # 多头注意力的头数
                "n_layers": 12,         # Number of layers  # Transformer 层数
                "drop_rate": HPARAM_CONFIG["drop_rate"],  # 使用当前超参数组合中的 dropout 比例
                "qkv_bias": False,     # Query-Key-Value bias  # 是否在 Q/K/V 线性层中使用偏置项
            }

            torch.manual_seed(123)  # 每个超参数组合训练前重新固定种子,保证不同配置间数据划分/初始化等的公平对比
            train_loader = create_dataloader_v1(
                text_data[:split_idx],  # 使用训练集部分的文本
                batch_size=HPARAM_CONFIG["batch_size"],  # 当前组合的批大小
                max_length=GPT_CONFIG_124M["context_length"],  # 每个样本序列的最大长度
                stride=GPT_CONFIG_124M["context_length"],  # 滑动窗口步幅,等于上下文长度即不重叠采样
                drop_last=True,  # 训练集丢弃不足一个批次的尾部数据,保证批大小一致
                shuffle=True,  # 训练集需要打乱顺序
                num_workers=0  # 不使用多进程数据加载
            )

            val_loader = create_dataloader_v1(
                text_data[split_idx:],  # 使用验证集部分的文本
                batch_size=HPARAM_CONFIG["batch_size"],
                max_length=GPT_CONFIG_124M["context_length"],
                stride=GPT_CONFIG_124M["context_length"],
                drop_last=False,  # 验证集保留全部数据,不丢弃尾部不足批次的部分
                shuffle=False,  # 验证集无需打乱,保持顺序便于评估的一致性
                num_workers=0
            )

            model = GPTModel(GPT_CONFIG_124M)  # 根据当前配置实例化一个全新的 GPT 模型
            model.to(device)  # 将模型参数搬移到目标设备

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=HPARAM_CONFIG["peak_lr"],  # 优化器初始学习率即预热结束后的峰值学习率
                weight_decay=HPARAM_CONFIG["weight_decay"]  # 当前组合的权重衰减系数
            )

            encoded_start_context = tokenizer.encode("Nevertheless")  # 将起始文本编码为 token id 列表(此处未在后续代码中实际使用,仅编码)
            encoded_tensor = torch.tensor(encoded_start_context).unsqueeze(0)  # 转为张量并增加 batch 维度,形状为 (1, seq_len)

            train_loss, val_loss = train_model(
                model, train_loader, val_loader, optimizer, device,
                n_epochs=HPARAM_CONFIG["n_epochs"],
                eval_iter=1,  # 训练后评估仅使用 1 个批次,加快超参数搜索速度
                warmup_iters=HPARAM_CONFIG["warmup_iters"],
                initial_lr=HPARAM_CONFIG["initial_lr"],
                min_lr=HPARAM_CONFIG["min_lr"]
            )

            # Log the best hyperparameters based on validation loss
            # 若当前配置的验证损失优于历史最优,则更新最优记录
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_hparams = HPARAM_CONFIG

        except KeyboardInterrupt:
            # 捕获用户手动中断(Ctrl+C),打印当前已找到的最优结果后跳出搜索循环
            print("Hyperparameter search completed.")
            print(f"Best hyperparameters: {best_hparams}")
            print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
            interrupted = True
            break

    if not interrupted:
        # 正常遍历完所有组合(未被中断)时,打印最终的最优超参数与对应损失
        print("Hyperparameter search completed.")
        print(f"Best hyperparameters: {best_hparams}")
        print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
