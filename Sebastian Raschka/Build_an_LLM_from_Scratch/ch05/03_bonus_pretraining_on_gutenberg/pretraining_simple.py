# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script for pretraining a small GPT-2 124M parameter model
on books from Project Gutenberg.

Before running this script, make sure you downloaded and
processed the dataset as described in the README.md.

模块级中文说明：
本脚本是在 Project Gutenberg 语料库上，对一个小型 GPT-2（1.24 亿参数）模型
进行预训练的主程序。整体流程包括：
    1. 解析命令行参数，构建模型配置（正常模式 / debug 调试模式）；
    2. 递归遍历数据目录，收集所有 .txt 训练文件；
    3. 逐本书（逐个 txt 文件）构建训练/验证 DataLoader，并在其上做若干轮
       梯度更新（train_model_simple 核心训练循环）；
    4. 定期做验证集评估、打印生成样例、保存 checkpoint；
    5. 训练结束后绘制损失曲线并保存最终模型权重。
该脚本设计上偏"简单/演示"用途（文件名含 simple），因此没有学习率调度、
梯度裁剪、混合精度等工程优化，重点在于把预训练主循环讲清楚。
"""

import argparse
import os
from pathlib import Path
import time
import tiktoken
import torch

# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# 从项目自身的包中导入前几章已经实现好的组件：
#   - create_dataloader_v1：第2章实现的滑动窗口式 DataLoader 构造函数
#   - GPTModel：第4章实现的 GPT 模型结构
#   - calc_loss_batch / evaluate_model / plot_losses / generate_and_print_sample：
#     第5章实现的训练辅助函数（计算单个 batch 的损失、评估、画图、生成样例文本）
from llms_from_scratch.ch02 import create_dataloader_v1  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)
from llms_from_scratch.ch04 import GPTModel  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)
from llms_from_scratch.ch05 import calc_loss_batch, evaluate_model, plot_losses, generate_and_print_sample  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)


def read_text_file(file_path):
    """读取指定路径的文本文件并返回其全部内容（字符串）。

    参数:
        file_path: 文本文件路径（str 或 Path）。

    返回:
        str，文件的完整文本内容（以 UTF-8 编码读取）。
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data


def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0):
    """将一段长文本按比例切分为训练集/验证集，并分别构建 DataLoader。

    参数:
        text_data: str，待切分的完整文本（通常是一本书的内容）。
        train_ratio: float，训练集所占比例（0~1），例如 0.9 表示前 90% 用于训练。
        batch_size: int，每个 batch 包含的样本数。
        max_length: int，每个训练样本的 token 序列长度（即模型的上下文长度）。
        stride: int，滑动窗口步长，用于控制样本之间的重叠程度。
        num_workers: int，DataLoader 使用的子进程数，默认 0（主进程加载）。

    返回:
        (train_loader, val_loader) 二元组，均为 torch DataLoader 对象。
        每个 batch 的张量形状约为 (batch_size, max_length)。
    """
    # 按 train_ratio 计算切分点索引，前半部分作训练集，后半部分作验证集
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,   # 训练集丢弃不满一个 batch 的尾部数据，保证 batch 大小一致
        shuffle=True,      # 训练集需要打乱顺序，避免模型学到样本顺序带来的偏差
        num_workers=num_workers
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,  # 验证集保留全部数据，不因不满一个 batch 而丢弃
        shuffle=False,     # 验证集不需要打乱，保证评估结果可复现、可比较
        num_workers=num_workers
    )
    return train_loader, val_loader


def convert_time(seconds):
    """将秒数转换为 (小时, 分钟, 秒) 的整数三元组，便于人类阅读的时间格式化输出。

    参数:
        seconds: float 或 int，总秒数。

    返回:
        (hours, minutes, seconds) 三个 int 组成的元组。
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, book_start_time, index, total_files):
    """打印当前处理进度的耗时统计与预计剩余时间（ETA）。

    参数:
        start_time: float，整个训练开始的时间戳（time.time() 的返回值）。
        book_start_time: float，当前这本书开始处理的时间戳。
        index: int，当前是第几本书（从 1 开始计数）。
        total_files: int，语料库中书籍（文件）总数。

    返回:
        None，仅打印信息到标准输出。
    """
    book_end_time = time.time()  # End time of processing this book
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    books_remaining = total_files - index
    # 用"已处理的书籍平均耗时"估算剩余书籍所需时间，属于简单线性外推
    average_time_per_book = total_elapsed_time / index
    eta = average_time_per_book * books_remaining

    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"Book processed {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining books: {eta_h}h {eta_m}m {eta_s}s")


def train_model_simple(model, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter, start_context,
                       output_dir, save_ckpt_freq, tokenizer,
                       batch_size=1024, train_ratio=0.90):
    """在 Gutenberg 语料（多个 txt 书籍文件）上执行简化版的预训练主循环。

    与常规做法不同的是：本函数不是把所有文本一次性加载为一个大数据集，
    而是"逐本书"读取文本、"逐本书"新建 DataLoader 并训练，这样可以避免
    一次性把全部语料加载进内存。

    参数:
        model: GPTModel 实例，待训练的语言模型。
        optimizer: torch.optim.Optimizer，例如 AdamW，用于更新模型参数。
        device: torch.device，模型和数据所在的计算设备（"cuda" 或 "cpu"）。
        n_epochs: int，训练的总轮数（每轮会遍历一次全部书籍）。
        eval_freq: int，每隔多少个 global_step 做一次训练/验证损失评估。
        eval_iter: int，评估时每个 DataLoader 上采样的 batch 数（用于加速评估）。
        print_sample_iter: int，每隔多少个 global_step 生成一次文本样例并打印。
        start_context: str，用于生成样例文本的起始提示词。
        output_dir: pathlib.Path，checkpoint 文件的保存目录。
        save_ckpt_freq: int，每隔多少个 global_step 保存一次 checkpoint。
        tokenizer: tiktoken 编码器实例，用于生成样例文本时的编码/解码。
        batch_size: int，训练/验证 DataLoader 的批大小，默认为 1024。
        train_ratio: float，每本书内部训练集所占比例，默认 0.90。

    返回:
        (train_losses, val_losses, track_tokens_seen) 三个 list：
            - train_losses: 各次评估时记录的训练集损失（float 列表）。
            - val_losses: 各次评估时记录的验证集损失（float 列表）。
            - track_tokens_seen: 各次评估时累计已训练的 token 数（int 列表）。

    注意（保留原实现细节，不做修改）：
        函数体内使用的 all_files 与 total_files 并非本函数的参数，而是
        依赖 Python 的作用域规则，在调用发生时从模块级全局变量中读取
        （即 __main__ 代码块中定义的 all_files / total_files）。
    """

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    start_time = time.time()

    try:
        for epoch in range(n_epochs):

            # Iterate over the books in the training corpus
            # 遍历语料库中的每一本书（每个 txt 文件），index 从 1 开始计数
            for index, file_path in enumerate(all_files, 1):
                book_start_time = time.time()
                # 读取整本书的文本，并在末尾拼接 <|endoftext|> 分隔符，
                # 用于告诉模型"这本书到此结束"，避免不同书籍内容被误认为连续文本
                text_data = read_text_file(file_path) + " <|endoftext|> "
                print(f"Tokenizing file {index} of {total_files}: {file_path}")

                # Initialize new data loaders for each book
                # 为当前这本书单独构建训练/验证 DataLoader；
                # max_length 和 stride 都设为 context_length，
                # 表示样本之间不重叠（步长等于窗口长度）
                train_loader, val_loader = create_dataloaders(
                    text_data,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    max_length=GPT_CONFIG_124M["context_length"],
                    stride=GPT_CONFIG_124M["context_length"],
                    num_workers=0
                )
                print("Training ...")
                model.train()  # 切换为训练模式（启用 dropout 等）
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()  # 清空上一步遗留的梯度
                    # 计算当前 batch 的交叉熵损失；
                    # input_batch/target_batch 形状均为 (batch_size, context_length)
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()   # 反向传播，计算各参数梯度
                    optimizer.step()  # 根据梯度更新模型参数
                    tokens_seen += input_batch.numel()  # 累计已训练过的 token 总数
                    global_step += 1  # 全局训练步数自增

                    # Optional evaluation step
                    # 每隔 eval_freq 步做一次训练/验证集损失评估，用于监控训练是否正常收敛
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        print(f"Ep {epoch+1} (Step {global_step}): "
                              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    # Generate text passage
                    # 每隔 print_sample_iter 步，用当前模型生成一段样例文本并打印，
                    # 便于直观观察模型生成能力的变化
                    if global_step % print_sample_iter == 0:
                        generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )

                # 注意：这里判断的是 `global_step % save_ckpt_freq` 本身的真假值
                # （非零即真），即当取余结果不为 0 时保存 checkpoint；
                # 这是原始实现逻辑，保持不变。
                if global_step % save_ckpt_freq:
                    file_name = output_dir / f"model_pg_{global_step}.pth"
                    torch.save(model.state_dict(), file_name)  # 仅保存模型参数（state_dict）
                    print(f"Saved {file_name}")

                # 打印本书处理耗时、累计耗时以及剩余书籍的预计完成时间
                print_eta(start_time, book_start_time, index, total_files)

    except KeyboardInterrupt:
        # 捕获用户手动中断（Ctrl+C），在退出前保存一份"中断快照"checkpoint，
        # 避免长时间训练因误操作中断而丢失全部进度
        file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
        torch.save(model.state_dict(), file_name)
        print(f"Saved {file_name}")

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":

    # 构建命令行参数解析器；formatter_class 会在 --help 中自动展示每个参数的默认值
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="GPT Model Training Configuration")

    parser.add_argument("--data_dir", type=str, default="gutenberg/data",
                        help="Directory containing the training data")
    parser.add_argument("--output_dir", type=str, default="model_checkpoints",
                        help="Directory where the model checkpoints will be saved")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="Number of epochs to train the model")
    parser.add_argument("--print_sample_iter", type=int, default=1000,
                        help="Iterations between printing sample outputs")
    parser.add_argument("--eval_freq", type=int, default=100,
                        help="Frequency of evaluations during training")
    parser.add_argument("--save_ckpt_freq", type=int, default=100_000,
                        help="Frequency of saving model checkpoints during training")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--debug", type=bool, default=False,
                        help="Uses a very small model for debugging purposes")

    args = parser.parse_args()

    if args.debug:
        # debug 模式下使用极小的模型配置，方便快速跑通整条训练流程、排查代码问题，
        # 而不必等待真实规模模型的训练耗时
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 10,    # Context length
            "emb_dim": 12,           # Embedding dimension
            "n_heads": 2,            # Number of attention heads
            "n_layers": 2,           # Number of layers
            "drop_rate": 0.0,        # Dropout rate, deactivated via 0.0 as dropout in LLMs is not recommended anymore
            "qkv_bias": False        # Query-key-value bias
        }

    else:
        # 正式训练模式下使用与 GPT-2 124M 一致的标准配置
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "emb_dim": 768,          # Embedding dimension
            "n_heads": 12,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "drop_rate": 0.1,        # Dropout rate
            "qkv_bias": False        # Query-key-value bias
        }

    # 优先使用 GPU（CUDA）加速训练，若不可用则退回 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)  # 固定随机种子，保证模型初始化等随机过程可复现
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    # 使用 AdamW 优化器；weight_decay=0.1 对权重做 L2 正则化，缓解过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")  # 使用与 GPT-2 一致的 BPE 分词器

    data_dir = args.data_dir
    # 递归遍历 data_dir 下的所有子目录，收集所有以 .txt 结尾的文件路径，
    # 这些文件即为 Gutenberg 语料库中的各本书
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)

    if total_files == 0:
        # 没找到任何训练文本文件时提前退出，避免后续训练循环报错
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保 checkpoint 输出目录存在

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, optimizer, device,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        print_sample_iter=args.print_sample_iter,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    # 生成与 train_losses 等长的、均匀分布在 [0, n_epochs] 区间的"epoch 坐标轴"，
    # 用于绘图时的横坐标（按 epoch 维度展示损失曲线）
    epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # 训练全部完成后，保存最终模型权重（与训练中途的 checkpoint 区分开）
    torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
