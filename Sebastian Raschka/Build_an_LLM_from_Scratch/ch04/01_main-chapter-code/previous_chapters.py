# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
中文说明（模块级 docstring）：

本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 4 章 (ch04) 的"前置代码"汇总文件 previous_chapters.py。

按照本书的写作惯例，每一章都会把前面章节已经讲解、实现并测试过的代码，
原样收集到一个 previous_chapters.py 文件中，方便本章后续内容直接 import 使用，
而不必在每一章重复粘贴之前的实现。本文件汇总了第 2 章（数据加载/分词）与
第 3 章（自注意力机制）的核心代码，具体包括：

1. GPTDatasetV1：把原始文本切分成用于"预测下一个词"训练任务的
   (输入序列, 目标序列) 样本对，采用滑动窗口的方式构造训练数据。
2. create_dataloader_v1：基于上面的 Dataset，构造 PyTorch 的 DataLoader，
   用于批量、乱序地喂给模型训练。
3. MultiHeadAttention：多头因果自注意力（Multi-Head Causal Self-Attention）
   模块，是 Transformer/GPT 架构中最核心的组件之一，用于让模型在预测当前
   位置的词时，只能"看到"它自己以及之前的词（即因果掩码，causal mask），
   并且通过多个"头"（head）从不同的子空间并行学习不同的注意力模式。

本章 (ch04) 会在此基础上，继续搭建完整的 GPT 模型结构（如前馈网络、
层归一化、残差连接、Transformer Block 等），所以这些"前置代码"是构建
完整 GPT 模型的地基。
"""

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """
    中文说明：
    自定义的 PyTorch Dataset，用于把一整段原始文本转换成"预测下一个词"
    (next-token prediction) 任务所需的训练样本。

    做法是：先用 tokenizer 把整段文本编码成一串 token id 序列，然后用一个
    长度为 max_length 的滑动窗口，以步长 stride 在这串 token id 上滑动，
    每滑动一次就截取出一段 (输入, 目标) 序列对：
        - 输入序列 input_chunk：窗口内的 token
        - 目标序列 target_chunk：把窗口整体向右移动一位后的 token
          （即每个位置上，目标就是"输入的下一个词"）

    当 stride < max_length 时，相邻样本之间会有重叠部分；
    当 stride == max_length 时，样本之间不重叠（书中默认演示的是有重叠的情况，
    可以让模型看到更多不同的上下文组合，起到类似数据增强的效果）。

    参数：
        txt (str): 原始的整段训练文本。
        tokenizer: 分词器对象（这里配合 tiktoken 的 GPT-2 编码器使用），
            需要有 .encode() 方法。
        max_length (int): 每个训练样本（即每个"上下文窗口"）包含的 token 数，
            对应模型的上下文长度 context_length。
        stride (int): 滑动窗口每次移动的步长（单位：token 个数）。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 中文：先把整段文本一次性编码成 token id 列表；
        # allowed_special 允许文本中出现的 "<|endoftext|>" 特殊标记被正常编码
        # （而不是被当作普通文本报错或拆开）。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 中文：用滑动窗口把整本书切成多个长度为 max_length、可能相互重叠的序列。
        # range 的终点是 len(token_ids) - max_length，这样保证每次截取
        # input_chunk 和 target_chunk（多取一位）时都不会越界。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            # 中文：目标序列相对输入序列整体右移一位，即 target[t] = input[t+1]，
            # 这正是语言模型"预测下一个 token"训练目标的构造方式。
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """中文：返回数据集中样本（即滑动窗口切出的序列对）的总数，供 DataLoader 使用。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        中文：根据索引 idx 取出第 idx 个训练样本。
        返回值：一个二元组 (input_ids, target_ids)，
        两者形状均为一维张量 (max_length,)，分别是输入 token 序列和目标 token 序列。
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    中文说明：
    便捷的工厂函数，把"创建 tokenizer -> 创建 GPTDatasetV1 -> 包装成 DataLoader"
    这三步封装成一次调用，方便训练脚本直接拿到可迭代的批量数据加载器。

    参数：
        txt (str): 原始训练文本。
        batch_size (int): 每个训练批次包含多少个样本（即多少个 max_length 长度的序列）。
        max_length (int): 每个样本的 token 序列长度（上下文窗口大小）。
        stride (int): 滑动窗口步长，控制样本之间的重叠程度。
        shuffle (bool): 是否在每个 epoch 打乱样本顺序（训练时通常为 True）。
        drop_last (bool): 若最后一个 batch 数量不足 batch_size 时是否丢弃，
            设为 True 可以避免最后一个 batch 形状不一致导致的问题。
        num_workers (int): 用于数据加载的子进程数量，0 表示在主进程中加载。

    返回值：
        torch.utils.data.DataLoader：可迭代对象，每次迭代产出一个 batch，
        包含 (input_ids, target_ids)，形状均为 (batch_size, max_length)。
    """
    # Initialize the tokenizer
    # 中文：使用 tiktoken 提供的 GPT-2 官方 BPE 分词器/编码器。
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    # 中文：用上面定义的 GPTDatasetV1，把文本切分成 (输入, 目标) 序列对组成的数据集。
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # 中文：DataLoader 负责按 batch_size 打包样本、可选打乱顺序 shuffle，
    # 并按需丢弃不满一个 batch 的尾部数据 drop_last。
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


class MultiHeadAttention(nn.Module):
    """
    中文说明：
    多头因果自注意力（Multi-Head Causal Self-Attention）模块，是 GPT 这类
    仅解码器 (decoder-only) Transformer 的核心构件。

    核心思想：
        1. 用三个独立的线性层，把输入向量分别投影成 Query（查询）、Key（键）、
           Value（值）三组向量。
        2. 把 Q/K/V 沿特征维度切分成 num_heads 个"头"，每个头独立地在
           更低维的子空间里计算注意力，最后再把各头的结果拼接起来。
           这样模型可以同时从多个不同的表示子空间里捕捉信息（例如某些头
           关注语法关系，某些头关注语义关系）。
        3. 使用"因果掩码"(causal mask)：对角线以上的位置（代表"未来"的
           token）会被置为 -inf，从而在做完 softmax 之后其注意力权重为 0，
           保证每个位置只能"看到"自己以及之前的 token，这正是自回归语言
           模型能够逐词生成文本的关键约束。

    参数：
        d_in (int): 输入向量的维度（embedding 维度）。
        d_out (int): 输出向量的维度，同时也是 Q/K/V 投影后的总维度，
            会被平均分配给每个注意力头，因此必须能被 num_heads 整除。
        context_length (int): 支持的最大上下文长度，用于预先构造好因果掩码矩阵。
        dropout (float): 注意力权重上使用的 dropout 概率，用于正则化、防止过拟合。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool): 是否给 Q/K/V 的线性变换加上偏置项 bias，默认为 False
            （与原始 GPT-2 论文/实现保持一致的一种可配置项）。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 中文：要求 d_out 能被 num_heads 整除，这样才能把总维度平均切分给每个头，
        # 保证每个头的维度 head_dim 是整数。
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：head_dim 就是每个注意力头内部 Q/K/V 向量的维度。

        # 中文：Q/K/V 三个独立的线性投影层，把输入 (…, d_in) 映射到 (…, d_out)。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # 中文：out_proj 是多头注意力最后的输出投影层，用于把拼接后的多头
        # 结果再做一次线性混合（融合不同头学到的信息）。
        self.dropout = nn.Dropout(dropout)
        # 中文：register_buffer 把因果掩码注册为模型的"缓冲区"（非可训练参数），
        # 这样它会随模型一起被 .to(device) 移动到 GPU/CPU，但不会被优化器更新。
        # torch.triu(..., diagonal=1) 生成一个上三角矩阵（不含主对角线），
        # 上三角部分为 1，代表"未来"位置，后面会用它来屏蔽注意力分数。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        中文说明：
        前向传播，计算多头因果自注意力的输出。

        参数：
            x (Tensor): 输入张量，形状为 (b, num_tokens, d_in)，
                b 是 batch size，num_tokens 是当前序列长度，d_in 是输入特征维度。

        返回值：
            Tensor: 注意力模块的输出，形状为 (b, num_tokens, d_out)，
                与输入的 batch 和序列长度保持一致，特征维度变为 d_out。
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)
        # 中文：此时 keys/queries/values 的形状都是 (b, num_tokens, d_out)，
        # 还没有区分"多头"，d_out 里其实隐含了 num_heads * head_dim。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆成 (num_heads, head_dim) 两维，
        # 也就是把一个大的投影结果，"隐式地"切分成多个头各自的向量。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度换到 num_tokens 前面，这样后续矩阵乘法就是
        # 对每个 (batch, head) 独立地在 (num_tokens, head_dim) 上做注意力计算，
        # 即把 num_heads 当作一个"批量"维度参与并行计算。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 中文：计算注意力分数（打分矩阵），即每个 query 位置对每个 key 位置的
        # 相似度。queries @ keys.transpose(2,3) 的形状为
        # (b, num_heads, num_tokens, num_tokens)，第 i 行第 j 列表示
        # "位置 i 的 query 与位置 j 的 key 的点积相似度"。
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        # 中文：因为 mask 是按 context_length（最大长度）预先构造好的，这里
        # 根据当前实际的 num_tokens 截取出对应大小的子矩阵，并转换成布尔型
        # （True 表示需要屏蔽的位置，即"未来"位置）。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 中文：把 mask 中为 True（即上三角、代表未来位置）的注意力分数
        # 原地替换为 -inf，这样经过 softmax 之后这些位置的权重会变成 0，
        # 从而实现"看不到未来 token"的因果约束。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：对分数除以 sqrt(head_dim) 做缩放（即"scaled" dot-product attention），
        # 这是为了防止点积数值过大导致 softmax 梯度消失，然后在最后一维
        # （key/token 维度）上做 softmax，得到归一化的注意力权重。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 中文：对注意力权重做 dropout，随机丢弃一部分注意力连接，起正则化作用。

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：用注意力权重对 values 做加权求和，得到每个 query 位置的
        # "上下文向量"(context vector)。attn_weights @ values 的形状是
        # (b, num_heads, num_tokens, head_dim)，再 transpose(1, 2) 换回
        # (b, num_tokens, num_heads, head_dim)，方便下一步拼接多头。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多个头的输出重新拼接（flatten）回单一的特征维度 d_out，
        # .contiguous() 保证张量在内存中连续存储，这样 .view() 才能正确重塑形状。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：最后再做一次线性投影，融合各个头的信息，得到最终输出。

        return context_vec