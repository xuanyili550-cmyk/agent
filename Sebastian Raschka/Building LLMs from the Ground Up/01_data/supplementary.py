# ============================================================================
# 本模块：为 GPT 预训练准备「数据管道」。
# 核心任务：把一大段原始文本 -> BPE 词元 id 序列 -> 用滑动窗口切成
#           (input, target) 训练样本对 -> 用 DataLoader 打包成 batch。
# 训练目标是「预测下一个 token」，因此 target 恰好是 input 右移一位。
# ============================================================================

import torch                         # PyTorch：提供 tensor 与自动微分
import tiktoken                      # OpenAI 的 BPE 分词器（GPT-2/3/4 同款）
# ✅ 已修复：原代码有一行错误的 `from PIL.TiffImagePlugin import idx`（idx 并不来自 PIL），已删除。
#           真正需要的 idx 是下方 __getitem__ 的参数名。
# Dataset：数据集抽象基类，需实现 __len__ 与 __getitem__；
# DataLoader：负责批处理(batch)、打乱(shuffle)、并行加载(num_workers)。
from torch.utils.data import Dataset, DataLoader
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# ----------------------------------------------------------------------------
# GPTDatasetV1：把整本书变成一批固定长度的 (input, target) 序列对。
# 继承 torch 的 Dataset，配合 DataLoader 使用。
# ----------------------------------------------------------------------------
class GPTDatasetV1(Dataset):
    # max_length：每个训练样本的 token 长度（即模型一次看到的上下文窗口）。
    # stride：滑动窗口每次向右移动的步长；stride==max_length 时样本不重叠，
    #         stride<max_length 时相邻样本有重叠（可增加样本数、更充分利用数据）。
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids=[]            # 存放所有 input 序列（每个是一个 tensor）
        self.target_ids=[]           # 存放所有 target 序列（input 右移一位）
        # Tokenize the entire text
        # 先把整段文本编码成一维的 token id 序列。allowed_special 允许把
        # "<|endoftext|>" 当作单个特殊 token 而非拆成普通字符。
        token_ids=tokenizer.encode(txt,allowed_special={"<|endoftext|>"})
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 滑动窗口：从位置 0 起，每隔 stride 取一段长度 max_length 的窗口。
        # 循环上界 len-max_length 保证 target 右移一位后仍不越界。
        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i +max_length]        # 输入：token[i : i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]    # 目标：整体右移一位，即「下一个 token」
            self.input_ids.append(torch.tensor(input_chunk))   # 转成 tensor 存入
            self.target_ids.append(torch.tensor(target_chunk))
    # 样本总数 = 切出的窗口个数
    def __len__(self):
        return len(self.input_ids)
    # 按索引取第 idx 个 (input, target) 对；DataLoader 会反复调用它来组 batch。
    # ✅ 已修复：原代码参数名为 ids 却在函数体误用 idx（来自已删除的错误 import），
    #           idx/ids 混用会导致索引错乱；现统一为同一个参数名 idx。
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# ----------------------------------------------------------------------------
# create_dataloader_v1：一站式工厂函数，输入原始文本，返回可迭代的 DataLoader。
# ✅ 已修复：原代码把该函数错误地缩进在了 GPTDatasetV1 类内部（只能靠「类名.方法」
#           歪打正着调用）；现移到类外、作为模块级函数（与后续章节的 supplementary.py
#           保持一致），调用方也相应改为 create_dataloader_v1(...)。
# ----------------------------------------------------------------------------
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")     # 用 GPT-2 的 BPE 分词器
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # 构造上面的数据集
    # DataLoader：把单条样本自动堆叠成 (batch_size, max_length) 的张量批次
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,      # 每批样本数
        shuffle=shuffle,            # 是否打乱顺序（训练常设 True，防止顺序偏置）
        drop_last=drop_last,        # 丢弃最后不足一个 batch 的残余，保证形状一致
        num_workers=num_workers,    # 子进程数，用于并行加载数据（0 表主进程加载）
    )
    return dataloader