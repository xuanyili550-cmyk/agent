# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# Code to run the exercises; see exercise-solutions.ipynb for more information

"""
中文模块说明
============
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
第 7 章「指令微调（Instruction Finetuning）」配套的**课后练习实验脚本**。

它把第 7 章主代码中最基础的指令微调流程（加载 GPT-2、构造指令数据集、
训练、生成、保存结果）封装进 `main()` 函数，并通过若干布尔开关（flag）
组合出书中提出的几种练习变体，方便一次性运行/对比：

1. baseline           —— 原始指令微调流程（无任何改动）。
2. mask_instructions   —— 在计算损失时，将「指令 + 输入」部分的 token
                          从损失中屏蔽（mask），只对「回复」部分反传梯度，
                          对应 `InstructionDatasetWithMasking` +
                          `custom_collate_with_masking_fn`。
3. alpaca_52k          —— 使用规模更大的 Stanford Alpaca 5.2 万条指令数据集
                          替换书中自带的小数据集，验证数据规模对效果的影响。
4. phi3_prompt         —— 把 Alpaca 风格的提示模板替换为 Phi-3 风格的
                          `<|user|>` / `<|assistant|>` 模板，对应
                          `InstructionDatasetPhi` + `format_input_phi`。
5. lora                —— 使用 LoRA（Low-Rank Adaptation，低秩适配）技术，
                          冻结原始权重、只训练新增的低秩矩阵 A、B，
                          从而大幅减少可训练参数量，对应
                          `LoRALayer` / `LinearWithLoRA` / `replace_linear_with_lora`。

这些变体通过命令行参数 `--exercise_solution` 选择，最终都会：
  - 下载/加载预训练 GPT-2 权重；
  - 在指令数据集上做若干轮微调；
  - 绘制训练/验证损失曲线；
  - 对测试集生成模型回复并保存为 JSON；
  - 保存微调后的模型权重（.pth）。

【重要】本文件按用户要求，仅在原始英文代码基础上**新增中文注释**，
不改变任何变量名、函数签名、逻辑、缩进或字符串字面量。
"""

from functools import partial
from importlib.metadata import version
import json
import math
import os
import re
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import requests
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import from local files in this folder
# 从同目录下的本地模块导入依赖：
#   - gpt_download: 负责下载并加载 OpenAI 官方发布的 GPT-2 预训练权重（TensorFlow 格式）
#   - previous_chapters: 汇总前几章已经实现好的 GPTModel、训练/生成/损失计算等工具函数
from gpt_download import download_and_load_gpt2
from previous_chapters import (
    calc_loss_loader,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text
)


class InstructionDataset(Dataset):
    """
    基础版指令微调数据集（Alpaca 风格）。

    对每条形如 {"instruction":..., "input":..., "output":...} 的样本，
    拼接成「指令+输入」提示文本 + 「### Response:\n回复」，
    再整体用 tokenizer 编码为一个 token id 序列，预先缓存在
    self.encoded_texts 中，避免训练时重复分词，提高效率。

    __getitem__ 返回的是单条未经填充（padding）的变长 token 序列，
    真正的填充与构造 (inputs, targets) 张量在 collate_fn 中完成。
    """
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        # 预先对所有样本分词，避免 DataLoader 每个 epoch 重复调用 tokenizer.encode，节省训练时间
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        # 返回第 index 条样本的 token id 列表（未 padding，长度各不相同）
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionDatasetWithMasking(Dataset):
    """
    支持「指令掩码（instruction masking）」的指令数据集，对应练习变体 mask_instructions。

    与 InstructionDataset 的区别：额外记录每条样本中
    「指令+输入」部分编码后的 token 长度（instruction_lengths），
    这样在 custom_collate_with_masking_fn 中就可以把 targets 里
    属于指令/输入部分的位置设为 ignore_index（-100），
    使得损失函数只在「模型生成的回复」部分计算梯度，
    从而让模型专注于学习「如何回答」而不是「记住指令怎么写」。
    """
    def __init__(self, data, tokenizer):
        self.data = data

        # New: Separate list for instruction lengths
        # 新增：单独维护一个列表，记录每条样本「指令+输入」部分的 token 长度，
        # 供 collate 阶段做掩码时使用
        self.instruction_lengths = []
        self.encoded_texts = []

        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

            # New: collect instruction lengths
            # 新增：单独对「指令+输入」部分再编码一次，只为获取其 token 长度，
            # 后续训练时用这个长度去屏蔽 targets 中对应位置的损失
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)

    def __getitem__(self, index):
        # New: return both instruction lengths and texts separately
        # 新增：同时返回该样本的指令长度和完整编码文本，
        # 二者会作为元组传给 custom_collate_with_masking_fn
        return self.instruction_lengths[index], self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class InstructionDatasetPhi(Dataset):
    """
    使用 Phi-3 风格提示模板（<|user|> / <|assistant|>）的指令数据集，
    对应练习变体 phi3_prompt，用于对比不同提示模板格式对微调效果的影响。
    """
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:

            ###################################################################
            # NEW: Use `format_input_phi` and adjust the response text template
            # 新增：改用 Phi-3 风格的 format_input_phi 构造提示，
            # 并把回复分隔符从 "### Response:" 换成 "<|assistant|>:"
            instruction_plus_input = format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            ###################################################################
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


class LinearWithLoRA(torch.nn.Module):
    """
    LoRA 包装层：把一个已有的 nn.Linear 层包装成
    「原始线性层（冻结） + 低秩适配分支（可训练）」的组合。

    前向计算为：output = linear(x) + lora(x)
    其中 linear 是原始预训练权重（一般会被冻结，requires_grad=False），
    lora 是新增的低秩矩阵分支，只训练这一小部分参数即可实现高效微调，
    这正是 LoRA（Low-Rank Adaptation）技术的核心思想。

    参数：
        linear: 待包装的原始 nn.Linear 层
        rank:   低秩矩阵的秩 r（越小，新增参数越少）
        alpha:  缩放系数，控制 LoRA 分支输出的幅度
    """
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # x 形状: (batch, seq_len, in_features)
        # 输出 = 原始线性变换 + LoRA 低秩分支的增量，二者形状相同 (batch, seq_len, out_features)
        return self.linear(x) + self.lora(x)


class LoRALayer(torch.nn.Module):
    """
    LoRA 低秩分支本体：用两个小矩阵 A (in_dim x rank) 和 B (rank x out_dim)
    的乘积 A@B 来近似模拟一个满秩的 (in_dim x out_dim) 权重更新量 ΔW，
    从而将需要训练的参数量从 in_dim*out_dim 降低到 (in_dim+out_dim)*rank，
    在 rank << min(in_dim, out_dim) 时可大幅节省显存和计算量。

    初始化技巧：
        - A 使用 kaiming_uniform_ 初始化（有非零随机值）；
        - B 初始化为全零；
      这样在训练刚开始时 A@B = 0，LoRA 分支不会改变原模型的输出，
      保证微调是从预训练模型的行为「平滑地」开始学习的。
    """
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        # x 形状: (..., in_dim) -> x@A 形状: (..., rank) -> @B 形状: (..., out_dim)
        # alpha 作为缩放系数，控制低秩增量对最终输出的影响强度
        x = self.alpha * (x @ self.A @ self.B)
        return x


def replace_linear_with_lora(model, rank, alpha):
    """
    递归遍历模型的所有子模块，把每一个 nn.Linear 层原地替换为
    LinearWithLoRA 包装层，从而给整个模型的所有线性层都注入 LoRA 分支。

    参数：
        model: 待改造的 nn.Module（一般是 GPTModel 或其子模块）
        rank:  LoRA 低秩矩阵的秩
        alpha: LoRA 缩放系数
    无返回值：直接原地（in-place）修改 model 的子模块。
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            # 命中 nn.Linear 层：用 setattr 直接替换成 LinearWithLoRA 包装后的模块
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            # 非 Linear 层（如 LayerNorm、Embedding、容器模块等）：递归深入其子模块继续查找
            replace_linear_with_lora(module, rank, alpha)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    DataLoader 的 collate_fn：把一个 batch 内长度不一的 token 序列
    填充（pad）到统一长度，并构造出用于「下一个 token 预测」训练的
    (inputs, targets) 张量对。

    核心步骤：
        1. 每条样本末尾先补一个 <|endoftext|> token（pad_token_id），作为结束标记；
        2. 用 pad_token_id 把所有样本填充到 batch 内最长序列的长度；
        3. inputs = 序列去掉最后一个 token；targets = 序列整体右移一位（即预测下一个 token）；
        4. 除每条样本第一个 padding token 外，其余 padding 位置的 targets
           都设为 ignore_index(-100)，这样交叉熵损失会自动忽略这些位置，
           不让模型在纯 padding 上学到无意义的梯度；
        5. 可选按 allowed_max_length 截断，避免超出模型上下文长度限制。

    返回：
        inputs_tensor, targets_tensor，形状均为 (batch_size, batch_max_length-1)，
        已经搬运到指定 device 上。
    """
    # Find the longest sequence in the batch
    # 找到该 batch 内（含新增的 <|endoftext|> 后）最长的序列长度，作为统一填充长度
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        # 在序列末尾追加一个 <|endoftext|> token，标记回复的结束
        new_item += [pad_token_id]
        # Pad sequences to max_length
        # 用 pad_token_id 填充到本 batch 的统一长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        # 新增：targets 中除了第一个 padding token（即真正的结束符，保留参与损失）外，
        # 其余全部替换为 ignore_index，避免模型被大量 padding 干扰训练
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        # 新增：如指定了最大长度上限，则对 inputs/targets 做截断，防止超出模型上下文窗口
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    # 把 list 堆叠成 batch 张量，并搬运到目标 device（CPU/GPU）
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def custom_collate_with_masking_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    带「指令掩码」的 collate_fn，配合 InstructionDatasetWithMasking 使用，
    对应练习变体 mask_instructions。

    与 custom_collate_fn 的唯一关键区别：
        除了屏蔽 padding 部分的损失外，还会额外把 targets 中属于
        「指令+输入」部分（长度为 instruction_length）的位置也设为
        ignore_index(-100)，使得模型训练时只在「回复」token 上计算损失，
        不会因为要「重新生成一遍指令文本」而分散学习信号。

    参数中 batch 的每个元素是 (instruction_length, token_ids) 元组，
    而不是纯 token_ids 列表——这是与普通 collate_fn 的输入形状差异。
    """
    # Find the longest sequence in the batch
    # 找到该 batch 内最长序列长度；注意这里 batch 元素是 (instruction_length, item) 元组
    batch_max_length = max(len(item)+1 for instruction_length, item in batch)   # New: batch is now a tuple

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:  # New: batch is now a tuple
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replace all but the first padding tokens in targets by ignore_index
        # 与基础版一致：屏蔽多余的 padding token 对应的损失
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Mask all input and instruction tokens in the targets
        # 新增：把 targets 中属于「指令+输入」部分（前 instruction_length-1 个位置，
        # 因为 targets 相对 inputs 整体右移了一位）全部设为 -100，
        # 使损失函数只在「回复」部分的 token 上计算梯度
        targets[:instruction_length-1] = -100

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    """
    下载并加载 JSON 格式的指令数据集。

    若本地 file_path 不存在，则从 url 下载文本内容并写入本地文件（做本地缓存，
    避免重复下载）；若已存在则直接读取本地文件。最终统一以 JSON 解析并返回
    Python 对象（通常是一个 list[dict]，每个 dict 含 instruction/input/output 字段）。

    参数：
        file_path: 本地缓存文件路径
        url:       远程数据集下载地址（baseline 用书中自带小数据集，
                   alpaca52k 用 Stanford Alpaca 的 5.2 万条数据集）
    返回：
        解析后的 JSON 数据（list[dict]）
    """
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def format_input_phi(entry):
    """
    按 Phi-3 提示模板格式化单条样本的「指令+输入」部分。

    格式为：
        <|user|>
        {instruction}
        {input}（如果 input 非空）

    与 format_input（Alpaca 风格）相比，模板更简洁，
    没有大段的任务说明文字，直接用 <|user|> 标签标识角色，
    用于对比不同提示词模板对指令微调效果的影响。
    """
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )

    input_text = f"\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def format_input(entry):
    """
    按 Alpaca 风格提示模板格式化单条样本的「指令+输入」部分，
    这是第 7 章正文以及本文件 baseline 变体默认使用的模板。

    格式为固定的任务说明文字 + "### Instruction:" 段落 +
    （如果有）"### Input:" 段落。
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, plot_name):
    """
    绘制训练/验证损失随「训练轮数（epoch）」和「已见 token 数」变化的曲线图，
    并保存为 PDF 文件。

    图中使用双 x 轴：
        - 下方 x 轴（ax1）：以 epoch 为单位；
        - 上方 x 轴（ax2）：以已处理 token 数为单位，与下方共享同一 y 轴（loss），
          便于直观对比「训练进度」与「实际计算量」两个维度。

    参数：
        epochs_seen:  与 loss 记录点一一对应的 epoch 数（可为小数，如 0.5 epoch）
        tokens_seen:  与 loss 记录点一一对应的累计已处理 token 数
        train_losses: 训练损失序列
        val_losses:   验证损失序列
        plot_name:    保存的文件名（.pdf）
    无返回值，直接把图保存到磁盘。
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training and validation loss against epochs
    # 绘制训练/验证损失随 epoch 变化的曲线（下方 x 轴）
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    # 创建共享同一 y 轴的第二条 x 轴，用于展示「已处理 token 数」这一视角
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


def main(mask_instructions=False, alpaca52k=False, phi3_prompt=False, lora=False):
    """
    指令微调实验的主流程，四个布尔开关分别对应四种练习变体，
    可以互相组合（除 mask_instructions 与 phi3_prompt 互斥外）：

        mask_instructions: 是否在损失计算中屏蔽指令部分（见上方数据集/collate 说明）
        alpaca52k:         是否使用更大规模的 Stanford Alpaca 5.2 万条数据集
        phi3_prompt:       是否使用 Phi-3 风格的提示模板
        lora:               是否用 LoRA 做参数高效微调（冻结原权重，只训练低秩分支）

    主要步骤：
        1. 打印依赖库版本，便于复现实验环境；
        2. 下载/加载指令数据集，按 85%/10%/5% 切分为 train/test/val；
        3. 根据开关选择对应的 Dataset 类和 collate_fn，构建 DataLoader；
        4. 下载并加载 GPT-2（默认 medium 355M）预训练权重到 GPTModel；
        5. 若启用 lora，则冻结全部原始参数，再用 LoRA 层替换所有 Linear 层；
        6. 微调前先评估初始损失作为基线；
        7. 用 AdamW 优化器训练 num_epochs 轮，并周期性在验证集上评估、生成样例文本；
        8. 绘制并保存损失曲线（文件名按启用的开关拼接后缀）；
        9. 对测试集逐条生成模型回复，保存为 JSON；
        10. 保存微调后的模型权重（.pth，文件名同样按开关拼接后缀）。

    无返回值：所有产出（图片、JSON、模型权重）均以文件形式落盘。
    """
    #######################################
    # Print package versions
    # 打印关键依赖库版本，便于复现实验结果时核对环境
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
    # 下载并准备指令数据集，按开关决定使用书中自带的小数据集还是 Alpaca 5.2 万条数据集
    #######################################
    file_path = "instruction-data.json"

    if alpaca52k:
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    else:
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing
    # 剩余约 5% 作为验证集（val_data），用于训练过程中周期性评估

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    if alpaca52k:
        allowed_max_length = 512
    else:
        allowed_max_length = 1024

    if mask_instructions and phi3_prompt:
        raise ValueError("Simultaneous support for instruction masking and the Phi-3 prompt template has not been implemented, yet.")

    # 根据开关组合选择对应的 Dataset 类与 collate_fn：
    #   mask_instructions -> 指令掩码版
    #   phi3_prompt       -> Phi-3 提示模板版
    #   否则              -> 基础 Alpaca 风格版
    if mask_instructions:
        customized_collate_fn = partial(custom_collate_with_masking_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetWithMasking
    elif phi3_prompt:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetPhi
    else:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDataset

    num_workers = 0

    if alpaca52k:
        batch_size = 4
    else:
        batch_size = 8

    torch.manual_seed(123)

    train_dataset = CustomDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = CustomDataset(val_data, tokenizer)
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
    # 加载 OpenAI 官方发布的 GPT-2 预训练权重，构建 GPTModel 并注入权重
    #######################################
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    if lora:
        # LoRA 微调流程：
        #   1) 先统计原本可训练参数量（此时应等于全部参数量）；
        #   2) 冻结所有原始参数（requires_grad=False），使其在训练中不再更新；
        #   3) 用 replace_linear_with_lora 给所有 Linear 层注入 LoRA 低秩分支
        #      （LoRA 分支默认 requires_grad=True，是唯一会被训练的部分）；
        #   4) 统计最终可训练参数量，应远小于原模型总参数量，体现「参数高效微调」的优势。
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters before: {total_params:,}")

        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters after: {total_params:,}")
        replace_linear_with_lora(model, rank=16, alpha=16)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LoRA parameters: {total_params:,}")
        model.to(device)

    #######################################
    # Finetuning the model
    # 微调模型：先评估微调前的初始损失作为基线，再进行若干轮训练
    #######################################
    print("Initial losses")
    with torch.no_grad():
        # 微调前先在少量 batch（num_batches=5）上快速评估训练/验证损失，作为对照基线
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    start_time = time.time()

    num_epochs = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    torch.manual_seed(123)

    # 训练过程中用验证集第一条样本的「指令+输入」作为示例上下文，
    # 每隔一定步数打印模型生成的样例文本，直观观察训练效果的变化；
    # 根据 phi3_prompt 开关选择对应的提示模板格式化函数
    start_context = format_input_phi(val_data[0]) if phi3_prompt else format_input(val_data[0])

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

    # 根据启用的开关组合，给损失曲线图文件名依次追加对应后缀，
    # 便于区分不同实验变体产出的结果文件
    plot_name = "loss-plot.pdf"
    if mask_instructions:
        plot_name = plot_name.replace(".pdf", "-mask-instructions.pdf")
    if alpaca52k:
        plot_name = plot_name.replace(".pdf", "-alpaca52k.pdf")
    if phi3_prompt:
        plot_name = plot_name.replace(".pdf", "-phi3-prompt.pdf")
    if lora:
        plot_name = plot_name.replace(".pdf", "-lora.pdf")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        plot_name = plot_name.replace(".pdf", "-baseline.pdf")

    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_name)
    print(50*"-")

    #######################################
    # Saving results
    # 保存实验结果：对测试集逐条生成模型回复，并保存回复结果与微调后的模型权重
    #######################################
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input_phi(entry) if phi3_prompt else format_input(entry)

        # 用贪心/采样生成（由 generate 内部策略决定）从提示文本继续生成，
        # 最多生成 256 个新 token，遇到 eos_id（<|endoftext|>=50256）提前停止
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        # 生成文本中包含原始的提示部分，这里截去提示前缀，
        # 并去掉模板里的分隔符标记（"### Response:" 或 "<|assistant|>:"），
        # 得到纯粹的模型回复文本
        if phi3_prompt:
            response_text = generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
        else:
            response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = "instruction-data-with-response.json"
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"

    # 与损失图文件名类似，按启用的开关组合为结果 JSON 和模型权重文件名追加对应后缀
    if mask_instructions:
        test_data_path = test_data_path.replace(".json", "-mask-instructions.json")
        file_name = file_name.replace(".pth", "-mask-instructions.pth")
    if alpaca52k:
        test_data_path = test_data_path.replace(".json", "-alpaca52k.json")
        file_name = file_name.replace(".pth", "-alpaca52k.pth")
    if phi3_prompt:
        test_data_path = test_data_path.replace(".json", "-phi3-prompt.json")
        file_name = file_name.replace(".pth", "-phi3-prompt.pth")
    if lora:
        test_data_path = test_data_path.replace(".json", "-lora.json")
        file_name = file_name.replace(".pth", "-lora.pth")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        test_data_path = test_data_path.replace(".json", "-baseline.json")
        file_name = file_name.replace(".pth", "-baseline.pth")

    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    print(f"Responses saved as {test_data_path}")

    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")


if __name__ == "__main__":

    import argparse

    # 命令行入口：通过 --exercise_solution 参数选择要运行的练习变体，
    # 五种取值分别对应 main() 的五种开关组合
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Instruction finetune a GPT model"
    )
    options = {"baseline", "mask_instructions", "alpaca_52k", "phi3_prompt", "lora"}
    parser.add_argument(
        "--exercise_solution",
        type=str,
        default="baseline",
        help=(
            f"Which experiment to run. Options: {options}."
        )
    )
    args = parser.parse_args()

    if args.exercise_solution == "baseline":
        main()
    elif args.exercise_solution == "mask_instructions":
        main(mask_instructions=True)
    elif args.exercise_solution == "alpaca_52k":
        main(alpaca52k=True)
    elif args.exercise_solution == "phi3_prompt":
        main(phi3_prompt=True)
    elif args.exercise_solution == "lora":
        main(lora=True)
    else:
        raise ValueError(f"{args.exercise_solution} is not a valid --args.exercise_solution option. Options: {options}")
