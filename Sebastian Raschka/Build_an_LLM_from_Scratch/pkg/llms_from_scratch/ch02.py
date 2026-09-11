# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块对应《从零构建大语言模型》第 2 章：文本数据处理。

核心内容：
1. 使用 tiktoken 的 GPT-2 BPE（Byte Pair Encoding，字节对编码）分词器，
   将原始文本转换为 token id 序列；
2. 用滑动窗口（sliding window）的方式，把很长的一段 token id 序列切分成
   多个长度固定为 max_length 的重叠子序列，分别作为模型的输入（input）
   和目标（target，即输入整体右移一位后的序列，用于"预测下一个词"任务）；
3. 将上述逻辑封装成 PyTorch 的 Dataset 与 DataLoader，供后续章节训练
   GPT 模型时按 batch 读取数据。
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    """
    自定义 PyTorch Dataset：把一整段文本切分成用于"下一个词预测"任务的
    (输入, 目标) 样本对。

    做法：
        - 先用分词器把整段文本编码成一个很长的 token id 列表；
        - 用一个长度为 max_length 的窗口，以步长 stride 在这条 token id
          列表上滑动，每次滑动截取一段作为输入 input_chunk；
        - 目标 target_chunk 是 input_chunk 整体向右平移一个 token 的结果，
          即 target_chunk[j] = input_chunk[j + 1] 对应位置的下一个 token，
          这正是语言模型"给定前文预测下一个 token"的训练目标。

    参数：
        txt (str): 原始训练文本（整本书/整段语料）。
        tokenizer: 具备 .encode() 方法的分词器对象（本章使用 tiktoken 的
            GPT-2 编码器）。
        max_length (int): 每个训练样本的 token 序列长度（即上下文窗口大小）。
        stride (int): 滑动窗口每次移动的步长（token 数）。
            stride < max_length 时，相邻样本之间会有重叠；
            stride == max_length 时，样本之间不重叠；
            stride 越小，生成的样本数越多，训练数据的"密度"越高，
            但也会带来更多相似样本、增加过拟合与训练成本。
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 对整段文本做一次性分词，得到一个很长的 token id 列表（一维）。
        # allowed_special={"<|endoftext|>"} 表示允许该特殊分隔符作为
        # "一个 token" 直接编码，而不是被拆成普通字符/子词，
        # 这个特殊符号常用于拼接多篇文档时标记文档边界。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 滑动窗口切分：
        #   - range 的终点是 len(token_ids) - max_length，
        #     确保每次截取 target_chunk 时 i + max_length + 1 不会越界；
        #   - 步长为 stride，因此当 stride < max_length 时，相邻窗口之间
        #     会有 (max_length - stride) 个 token 重叠。
        for i in range(0, len(token_ids) - max_length, stride):
            # 输入序列：从位置 i 到 i + max_length（不含），长度为 max_length
            input_chunk = token_ids[i:i + max_length]
            # 目标序列：整体右移一位，即从 i+1 到 i+max_length+1（不含），
            # 长度同样为 max_length；target_chunk 的第 j 个元素正是
            # input_chunk 第 j 个元素的"下一个 token"，
            # 用于训练模型做逐位置的下一词预测。
            target_chunk = token_ids[i + 1: i + max_length + 1]
            # 转成一维 LongTensor，形状均为 (max_length,)
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集中样本（即滑动窗口切出的片段）总数。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        按索引取出一条样本。

        返回：
            (input_ids, target_ids)：两个形状均为 (max_length,) 的一维
            张量，分别是模型的输入 token 序列和对应的下一词预测目标。
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    便捷函数：根据原始文本构建 GPT-2 分词 + 滑动窗口切分的 DataLoader。

    参数：
        txt (str): 原始训练文本。
        batch_size (int): 每个 batch 的样本数，决定 DataLoader 产出张量
            的第 0 维大小，最终每个 batch 的 input/target 形状均为
            (batch_size, max_length)。
        max_length (int): 每条样本的 token 序列长度（上下文窗口大小）。
        stride (int): 滑动窗口步长，控制样本之间的重叠程度（见
            GPTDatasetV1 的说明）。
        shuffle (bool): 是否在每个 epoch 打乱样本顺序。
        drop_last (bool): 若最后一个 batch 的样本数不足 batch_size，
            是否丢弃该 batch（训练时通常设为 True，以保证每个 batch
            形状一致，避免因最后一批样本过少导致的梯度不稳定）。
        num_workers (int): 数据加载所用的子进程数（0 表示在主进程加载）。

    返回：
        torch.utils.data.DataLoader：每次迭代产出一个
        (input_ids, target_ids) 元组，二者形状均为 (batch_size, max_length)。
    """

    # Initialize the tokenizer
    # 初始化分词器：使用 tiktoken 提供的 GPT-2 编码方案（BPE 词表）。
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    # 用上面定义的 GPTDatasetV1 完成"分词 + 滑动窗口切分"的全部预处理。
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # DataLoader 负责把 Dataset 中的多个 (max_length,) 样本按 batch_size
    # 打包成 (batch_size, max_length) 的批量张量，并按需打乱、丢弃尾批。
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
