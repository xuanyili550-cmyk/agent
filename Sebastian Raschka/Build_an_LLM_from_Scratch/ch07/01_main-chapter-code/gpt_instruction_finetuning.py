# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# A minimal instruction finetuning file based on the code in chapter 7

"""
模块中文说明
============
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
第 7 章的“极简版指令微调（Instruction Finetuning）”独立脚本。

它在全书中的角色：
- 第 5 章介绍了如何用无监督方式（下一词预测）预训练 GPT 模型；
- 第 6 章介绍了如何用有监督方式对分类任务做微调；
- 本章（第 7 章）在此基础上，展示如何用“指令-输入-输出”三元组数据，
  对预训练好的 GPT-2 模型做“指令微调”，使其能够按照自然语言指令完成任务
  （即构建一个简化版的 InstructGPT / ChatGPT 式对话能力的前身）。

整体流程：
1. 下载/加载指令微调数据集（JSON 格式，每条包含 instruction/input/output）；
2. 将每条数据格式化为统一的 Alpaca 风格提示模板（format_input）；
3. 用自定义 Dataset（InstructionDataset）把文本预先分词；
4. 用自定义 collate 函数（custom_collate_fn）在 batch 内做动态填充，
   并把输入右移一位得到目标序列，同时用 ignore_index 屏蔽多余的填充 token，
   使其不参与损失计算；
5. 加载预训练的 GPT-2 权重，构建模型；
6. 用标准的训练循环（train_model_simple，定义在 previous_chapters.py 中）
   对模型做指令微调，并记录/绘制训练与验证损失曲线；
7. 用微调后的模型对测试集生成回答，保存结果与模型权重。

本文件是脚本化、可独立运行的版本，对应正文 Jupyter Notebook 中的核心代码，
方便读者直接用 `python gpt_instruction_finetuning.py` 命令行运行整个流程。
"""

from functools import partial
from importlib.metadata import version
import json
import os
import re
import time

import matplotlib.pyplot as plt
import requests
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import from local files in this folder
# 从当前目录下的本地模块导入：
# - download_and_load_gpt2: 下载并加载 OpenAI 发布的 GPT-2 预训练权重
# - previous_chapters 中的工具函数/类均来自前几章已经实现好的代码，
#   这里直接复用，避免重复造轮子
from gpt_download import download_and_load_gpt2
from previous_chapters import (
    calc_loss_loader,      # 计算某个 DataLoader 上的平均损失
    generate,               # 自回归生成函数（贪心/多样化采样均可，这里用于推理）
    GPTModel,               # GPT 模型主体结构（第4章实现）
    load_weights_into_gpt,  # 把下载好的 GPT-2 权重加载进自定义的 GPTModel
    text_to_token_ids,      # 文本 -> token id 张量
    train_model_simple,     # 简化版训练循环（前向、反向、优化器更新、周期性评估）
    token_ids_to_text       # token id 张量 -> 文本
)


class InstructionDataset(Dataset):
    """
    指令微调数据集类（PyTorch Dataset 子类）。

    作用：
        将原始的 JSON 格式指令数据（每条含 instruction/input/output 字段）
        转换为统一的提示文本，并在构造函数中“预先分词”（pre-tokenize），
        以避免在训练时每个 epoch 都重复调用分词器，提升训练效率。

    数据格式（每条 entry）：
        instruction_plus_input = format_input(entry)   # 指令+输入部分
        response_text = "\\n\\n### Response:\\n" + entry['output']  # 期望回答部分
        full_text = instruction_plus_input + response_text          # 拼接成完整训练文本

    属性：
        self.data: 原始数据列表（未处理）
        self.encoded_texts: 每条样本对应的 token id 列表（List[List[int]]）
    """
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        # 预先分词：提前把所有样本的文本转换成 token id 列表并缓存，
        # 这样 __getitem__ 只需要做简单的列表索引，避免重复分词开销
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        # 返回第 index 条样本对应的 token id 列表（尚未填充，长度可变）
        return self.encoded_texts[index]

    def __len__(self):
        # 数据集样本总数
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    自定义的 DataLoader collate 函数，用于把一个 batch 内长度不一的 token
    序列，转换成等长的输入张量 inputs 与目标张量 targets。

    参数：
        batch: List[List[int]]，一个 batch 内多条样本的 token id 列表（长度不一）
        pad_token_id: 用于填充的 token id，这里用 GPT-2 的 <|endoftext|> (50256)
        ignore_index: 交叉熵损失中要忽略的标签值（PyTorch 默认约定 -100），
                      用于屏蔽“多余”的填充位置，使其不参与损失计算
        allowed_max_length: 若设置，则将 inputs/targets 截断到该最大长度，
                            防止序列超出模型的上下文窗口
        device: 目标计算设备（"cpu" 或 "cuda"）

    返回：
        inputs_tensor:  形状 (batch_size, seq_len) 的输入 token id 张量
        targets_tensor: 形状 (batch_size, seq_len) 的目标 token id 张量
                        （相对 inputs 整体右移一位，用于“预测下一个 token”）
    """
    # Find the longest sequence in the batch
    # 找到 batch 内最长序列的长度（+1 是为后面追加的 <|endoftext|> 结束符预留位置）
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        # 在每条样本末尾追加一个 <|endoftext|> token，显式标记回答结束
        new_item += [pad_token_id]
        # Pad sequences to max_length
        # 将序列填充到本 batch 内的最大长度，保证可以堆叠成规整张量
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        # inputs: 去掉最后一个 token（因为它没有“下一个token”可预测）
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        # targets: 整体右移一位，即 targets[t] = inputs[t+1]，
        # 这正是“用当前及之前所有 token 预测下一个 token”的自回归训练目标

        # New: Replace all but the first padding tokens in targets by ignore_index
        # 关键点：一条样本的填充 token 中，只保留“第一个”pad_token_id 作为
        # 有效的结束符标签（让模型学会何时停止生成），
        # 其余多余的填充位置全部替换为 ignore_index(-100)，
        # 这样在计算交叉熵损失时 PyTorch 会自动跳过这些位置，不产生梯度贡献
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        # 可选：若设置了最大长度限制（如 1024，对应 GPT-2 的上下文窗口），
        # 则将过长序列截断，避免超出模型支持的位置编码范围
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    # 将列表堆叠为二维张量 (batch_size, seq_len)，并搬运到目标设备（如 GPU）
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    """
    下载（若本地不存在）并加载 JSON 格式的指令微调数据集文件。

    参数：
        file_path: 本地缓存文件路径，若已存在则跳过下载
        url: 数据集的远程下载地址

    返回：
        data: 解析后的 Python 对象（通常是 List[dict]，
              每个 dict 含 instruction/input/output 字段）
    """
    if not os.path.exists(file_path):
        # 本地不存在缓存文件时才发起网络请求，避免重复下载
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def format_input(entry):
    """
    将单条原始数据（含 instruction、input、output 字段）格式化为
    统一的 Alpaca 风格提示词模板（不含 output 部分）。

    参数：
        entry: dict，至少包含 "instruction" 和 "input" 键
               （"input" 可以为空字符串，表示该指令不需要额外输入）

    返回：
        str: 拼接好的提示文本，形如：
             "Below is an instruction ... \\n\\n### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}"
             若 input 为空，则不包含 "### Input:" 部分。

    说明：
        这种“指令/输入/回答”三段式模板是指令微调数据的常见组织方式，
        让模型学会区分任务描述、任务输入与期望输出三部分的边界。
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 只有当 input 字段非空时才拼接 "### Input:" 段落，
    # 这样纯指令类任务（无需额外输入）的模板会更简洁
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    绘制训练/验证损失曲线，并保存为 PDF 文件。

    参数：
        epochs_seen: 与 train_losses/val_losses 对齐的“已训练轮数”坐标（用于横轴1）
        tokens_seen: 与 train_losses 对齐的“已处理 token 数”坐标（用于横轴2）
        train_losses: 训练损失列表
        val_losses: 验证损失列表

    效果：
        在同一张图上用两个 x 轴分别展示“按 epoch”和“按已见 token 数”的损失变化，
        便于直观比较训练进度与数据吞吐量的关系；图像保存为
        "loss-plot-standalone.pdf"。
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training and validation loss against epochs
    # 主坐标轴：以“训练轮数”为横轴，绘制训练损失与验证损失
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    # 共享同一个 y 轴，新增一条不可见曲线，仅用于对齐第二条横轴的刻度
    # （即让“已见 token 数”这一横轴与上面的 epoch 横轴刻度对应起来）
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plot_name = "loss-plot-standalone.pdf"
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


def main(test_mode=False):
    """
    指令微调主流程入口函数。

    参数：
        test_mode: bool，为 True 时使用极小规模的数据子集与极小模型，
                   仅用于快速跑通流程做内部测试（例如 CI），
                   不代表书中推荐的真实训练配置。

    主要步骤：
        1. 打印关键依赖库版本，便于复现环境；
        2. 下载并划分指令数据集为 训练/验证/测试 三部分；
        3. 构建 Dataset 与 DataLoader（含动态 padding 的 collate 函数）；
        4. 加载 GPT-2 预训练权重（或测试用的小模型）；
        5. 计算微调前的初始损失作为基线；
        6. 执行指令微调训练循环，并记录/绘制损失曲线；
        7. 用微调后的模型对测试集生成回答，保存结果 JSON 与模型权重文件。

    返回：
        无返回值；结果以副作用形式写入磁盘文件
        （instruction-data-with-response-standalone.json、*.pth、损失曲线 PDF）。
    """
    #######################################
    # Print package versions
    #######################################
    print()
    pkgs = [
        "matplotlib",  # Plotting library
        "tiktoken",    # Tokenizer
        "torch",       # Deep learning library
        "tqdm",        # Progress bar
        "tensorflow",  # For OpenAI's pretrained weights
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")

    #######################################
    # Download and prepare dataset
    #######################################
    file_path = "instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    # 按 85% / 10% / 5% 的比例划分 训练集 / 测试集 / 验证集
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]  # 剩余约 5% 作为验证集

    # Use very small subset for testing purposes
    # 测试模式下只取前 10 条样本，快速验证代码流程是否能跑通
    if test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    # 使用与 GPT-2 一致的 BPE 分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    # 用 partial 预先绑定 device 与 allowed_max_length 参数，
    # 得到一个符合 DataLoader collate_fn 接口（仅需传入 batch）的可调用对象；
    # allowed_max_length=1024 对应 GPT-2 的最大上下文长度，防止越界
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)  # 固定随机种子，保证 DataLoader shuffle 等操作可复现

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,       # 训练集需要打乱顺序，避免模型学到样本顺序的偏差
        drop_last=True,     # 丢弃最后不足一个 batch 的样本，保证每个 batch 大小一致
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,      # 验证集不需要打乱，保证评估结果可复现、可比较
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # Load pretrained model
    #######################################

    # Small GPT model for testing purposes
    # 测试模式下使用一个参数量极小的 GPT 配置，
    # 目的仅是验证整个训练/推理流程能否正确运行，而非追求效果
    if args.test_mode:
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
        model.eval()
        device = "cpu"
        CHOOSE_MODEL = "Small test model"

    # Code as it is used in the main chapter
    # 正式流程：加载真实的 GPT-2 预训练权重进行指令微调（书中推荐路径）
    else:
        BASE_CONFIG = {
            "vocab_size": 50257,     # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,        # Dropout rate
            "qkv_bias": True         # Query-key-value bias
        }

        # 不同规模 GPT-2 模型对应的结构超参数（嵌入维度、层数、注意力头数）
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        CHOOSE_MODEL = "gpt2-medium (355M)"

        # 用所选模型规模的专属超参数更新基础配置
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        # 从模型名中提取规模标识（如 "355M"），用于下载对应的预训练权重
        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)  # 将 OpenAI 原始权重映射并加载进自定义结构
        model.eval()
        model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    #######################################
    # Finetuning the model
    #######################################
    print("Initial losses")
    # 微调前先在训练集/验证集各抽取 5 个 batch，计算平均损失作为基线，
    # 便于后续与微调后的损失做对比，评估微调效果
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    start_time = time.time()
    # AdamW 优化器：常用于 Transformer 微调，weight_decay 提供 L2 正则化，
    # 学习率 5e-5 是指令微调阶段常见的较小学习率（相较预训练更保守）
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2

    torch.manual_seed(123)  # 固定种子，保证训练过程（如批次顺序）可复现
    # 调用前几章实现的通用训练循环：
    # 内部会执行 前向传播 -> 计算交叉熵损失（自动忽略 ignore_index 标签）
    # -> 反向传播 -> 优化器更新，并周期性在训练/验证集上评估、
    # 用 start_context 提示词生成样例文本以直观查看训练效果
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 生成与 train_losses 等长的“已训练轮数”坐标，用于绘图横轴对齐
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    #######################################
    # Saving results
    #######################################
    print("Generating responses")
    # 用微调后的模型，对测试集中的每条指令生成回答，
    # 并将生成结果写回每条数据的 "model_response" 字段，便于后续人工/自动评估
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,                      # 最多生成 256 个新 token
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256                              # 遇到 <|endoftext|> 即提前停止生成
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        # 生成结果中包含了原始输入提示词，这里截取掉输入部分，
        # 并去除模板中的 "### Response:" 标记，只保留模型真正生成的回答内容
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = "instruction-data-with-response-standalone.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Responses saved as {test_data_path}")

    # 根据所选模型名生成权重保存文件名（去掉空格和括号，避免文件名非法字符）
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft-standalone.pth"
    torch.save(model.state_dict(), file_name)  # 仅保存模型参数（state_dict），而非整个模型对象
    print(f"Model saved as {file_name}")


if __name__ == "__main__":

    import argparse

    # 命令行参数解析：目前仅支持一个 --test_mode 开关，
    # 用于在小规模/小模型上快速验证脚本是否能正常运行（如 CI 场景）
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Finetune a GPT model for classification"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the chapter (recommended).")
    )
    args = parser.parse_args()

    main(args.test_mode)
