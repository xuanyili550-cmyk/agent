# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-5.
# This file can be run as a standalone script.

"""
中文模块说明
============
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 第 6 章
(ch06) 示例代码所依赖的"前置代码合集"。它把第 2~5 章中逐步搭建起来的核心组件
汇总在一起，方便第 6 章（微调 GPT 做文本分类）直接复用，而不用重复粘贴：

    - 第 2 章：GPT 风格的数据集/数据加载器 (GPTDatasetV1, create_dataloader_v1)，
      负责把原始文本切分成滑动窗口式的 (输入, 目标) token 序列对。
    - 第 3 章：多头自注意力机制 (MultiHeadAttention)，包含因果掩码
      (causal mask) 与缩放点积注意力 (scaled dot-product attention)。
    - 第 4 章：层归一化 (LayerNorm)、GELU 激活函数、前馈网络 (FeedForward)、
      Transformer 块 (TransformerBlock) 以及完整的 GPT 模型 (GPTModel)，
      还有最简单的贪心解码文本生成函数 (generate_text_simple)。
    - 第 5 章：将 OpenAI 官方发布的 GPT-2 预训练权重加载进我们自己实现的
      GPTModel 中的工具函数 (assign, load_weights_into_gpt)，以及文本与
      token id 之间相互转换的辅助函数 (text_to_token_ids, token_ids_to_text)。

第 6 章会在此基础上，把 GPTModel 的输出头替换成分类头，从而将一个预训练好的
语言模型微调为垃圾邮件分类器。因此本文件本身不引入新概念，而是作为
"承上启下"的工具箱使用。
"""

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    """GPT 训练用的滑动窗口数据集（第 2 章）。

    把一整段文本先用 BPE 分词器编码成 token id 序列，然后用一个长度为
    `max_length`、步长为 `stride` 的滑动窗口在这条 token 序列上滚动切片，
    生成一系列 (输入片段, 目标片段) 样本对。其中目标片段相对输入片段整体
    右移一位，即"预测下一个 token"的自回归训练目标。

    参数:
        txt (str): 原始训练文本。
        tokenizer: 具备 `encode` 方法的分词器（此处配合 tiktoken 的 GPT-2 编码器使用）。
        max_length (int): 每个训练样本（上下文窗口）包含的 token 数量。
        stride (int): 滑动窗口每次移动的步长；stride < max_length 时窗口之间会有重叠。

    每个样本的张量形状:
        input_ids[i]:  形状为 (max_length,) 的一维张量。
        target_ids[i]: 形状为 (max_length,) 的一维张量，内容是 input_ids[i] 整体右移一位。
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 中文：先把整段文本一次性编码为 token id 列表；
        # allowed_special 允许文本中出现的 "<|endoftext|>" 特殊标记被正常编码，
        # 而不是被当成非法字符抛异常。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 中文：用滑动窗口把整本书切成长度为 max_length、可能相互重叠的片段。
        # 循环上界 len(token_ids) - max_length 保证最后一个窗口仍能取到完整的
        # max_length + 1 个 token（因为 target 要比 input 多取一位）。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            # 目标序列相对输入序列整体右移一位，即“预测下一个 token”
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 中文：数据集样本总数，即切出来的窗口数量。
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 中文：按索引返回一对 (输入 token 序列, 目标 token 序列)，供 DataLoader 取用。
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """构造第 2 章使用的 GPT 数据加载器（DataLoader）。

    内部会先用 GPT-2 的 BPE 分词器把文本转成 token id，再包装成
    `GPTDatasetV1`，最后用 PyTorch 的 `DataLoader` 按批次输出。

    参数:
        txt (str): 原始训练文本。
        batch_size (int): 每个训练批次的样本数量。
        max_length (int): 每个样本（上下文窗口）的 token 长度。
        stride (int): 滑动窗口步长。
        shuffle (bool): 是否打乱样本顺序。
        drop_last (bool): 若最后一个 batch 不满 batch_size 是否丢弃（训练时常设 True 以保证批次大小一致）。
        num_workers (int): 数据加载的子进程数量。

    返回:
        torch.utils.data.DataLoader: 每次迭代产出形状为
        (batch_size, max_length) 的 (输入, 目标) 张量对。
    """
    # Initialize the tokenizer
    # 中文：使用 tiktoken 提供的 GPT-2 官方编码器（BPE 分词）
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    # 中文：构造滑动窗口数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # 中文：包装成 PyTorch DataLoader，负责批处理、打乱顺序等
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """多头因果自注意力模块（第 3 章）。

    将输入序列先线性投影得到 Query/Key/Value，再把每个头的维度拆分出来
    并行计算缩放点积注意力，同时用上三角掩码（causal mask）屏蔽掉"看到未来
    token"的位置，从而保证模型只能利用当前及之前的 token 做预测。最后将
    各个头的注意力输出拼接（reshape）回原始维度，并做一次线性投影融合。

    参数:
        d_in (int): 输入特征维度。
        d_out (int): 输出特征维度（同时也是所有头拼接后的总维度）。
        context_length (int): 支持的最大序列长度，用于预先构造因果掩码矩阵。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头数，需要满足 d_out 能被 num_heads 整除。
        qkv_bias (bool): Q/K/V 线性层是否使用偏置项。

    关键张量形状（b=batch size, num_tokens=序列长度）:
        输入 x:            (b, num_tokens, d_in)
        keys/queries/values 投影后: (b, num_tokens, d_out)
        拆分成多头后:       (b, num_tokens, num_heads, head_dim)
        转置后:            (b, num_heads, num_tokens, head_dim)
        注意力分数 attn_scores: (b, num_heads, num_tokens, num_tokens)
        上下文向量 context_vec（拼接多头后）: (b, num_tokens, d_out)
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头分到的维度大小，多头拼接后总维度仍等于 d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        # 中文：register_buffer 让掩码随模型一起 .to(device)/保存，但不会被当作可训练参数。
        # torch.triu(..., diagonal=1) 生成一个严格上三角矩阵（对角线以上为 1），
        # 用来标记"未来位置"，后续在 forward 中会把这些位置的注意力分数设为 -inf。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # 中文：x 的形状为 (batch_size, num_tokens, d_in)
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆分成 (num_heads, head_dim)，从而“隐式地”
        # 将一次大投影切分成多个头各自的小投影，不需要为每个头单独定义线性层。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 提到 batch 维之后，方便对每个头独立做矩阵乘法（批量矩阵乘）
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文：queries @ keys^T，对每个头分别做点积，
        # 结果形状 (b, num_heads, num_tokens, num_tokens)，
        # 元素 [.., i, j] 表示第 i 个 token 对第 j 个 token 的注意力打分（尚未归一化）

        # Original mask truncated to the number of tokens and converted to boolean
        # 中文：因为掩码是按 context_length 预先构造好的最大尺寸矩阵，
        # 这里截取当前实际序列长度 num_tokens 对应的左上角子矩阵，并转为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 中文：把掩码中标记为 True（即未来位置）的注意力分数填充为 -inf，
        # 这样经过 softmax 后这些位置的权重会变成 0，实现"因果"（只能看过去）的效果
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：按最后一维做 softmax 得到注意力权重；除以 sqrt(head_dim) 是缩放点积注意力
        # 的“缩放”部分，用于避免维度过高时点积值过大导致 softmax 梯度消失
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：注意力权重与 values 加权求和得到每个头的上下文向量，
        # 再转置回 (b, num_tokens, num_heads, head_dim) 便于后续拼接
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多头拼接（reshape）回单一的 d_out 维度
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：最后再做一次线性投影，融合各头的信息（可选，但原始 Transformer 论文中包含此步）

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化模块（第 4 章，手写实现版）。

    对最后一个维度（特征维度）做均值/方差归一化，然后用可学习的缩放
    (scale) 和平移 (shift) 参数做仿射变换。与 `nn.LayerNorm` 的效果类似，
    但这里显式手写以便读者理解其内部计算过程。

    参数:
        emb_dim (int): 特征（embedding）维度，即在该维度上做归一化。

    张量形状:
        输入/输出 x: (..., emb_dim)，通常是 (batch_size, num_tokens, emb_dim)。
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除以 0 的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 中文：可学习缩放参数，初始为全 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 中文：可学习平移参数，初始为全 0

    def forward(self, x):
        # 中文：沿最后一维（特征维）计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False 表示使用有偏方差估计（除以 N 而非 N-1），与 GPT-2 官方实现保持一致
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 中文：归一化：减均值除以标准差，使得每个 token 的特征分布均值为 0、方差为 1
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 中文：再做一次可学习的仿射变换，让网络可以自行调节归一化后的尺度和偏移
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数（第 4 章，tanh 近似实现版）。

    GPT-2 使用的是 GELU 的近似（tanh-based approximation）形式，而不是
    精确的高斯误差函数形式，计算更快且效果接近。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 中文：GELU 的 tanh 近似公式：
        # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        # 相比 ReLU，GELU 在 0 附近更平滑，能保留一定的负值梯度信息
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer 块中的前馈网络（第 4 章）。

    结构为：线性层升维 (emb_dim -> 4*emb_dim) -> GELU 激活 -> 线性层降维
    (4*emb_dim -> emb_dim)。中间维度放大 4 倍是 GPT/Transformer 系列的常见设计，
    用于增加模型的非线性表达能力。

    参数:
        cfg (dict): 配置字典，需包含 "emb_dim" 键。

    张量形状:
        输入/输出: (batch_size, num_tokens, emb_dim)
        中间层:    (batch_size, num_tokens, 4 * emb_dim)
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        # 中文：依次通过升维线性层 -> GELU -> 降维线性层
        return self.layers(x)


class TransformerBlock(nn.Module):
    """标准 GPT 风格的 Transformer 块（第 4 章）。

    采用 Pre-LayerNorm 结构：先归一化再进入子层（注意力/前馈），
    子层输出经过 dropout 后再与残差（shortcut）相加，这种结构比
    Post-LayerNorm 更利于深层网络的训练稳定性。

    参数:
        cfg (dict): 配置字典，需包含 emb_dim、context_length、n_heads、
            drop_rate、qkv_bias 等键。

    张量形状:
        输入/输出 x: (batch_size, num_tokens, emb_dim)
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        # 中文：注意力子层的残差连接：保存原始输入 x 作为 shortcut
        shortcut = x
        x = self.norm1(x)   # 中文：Pre-LN，先归一化再进入注意力层
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)  # 中文：对子层输出做 dropout，缓解过拟合
        x = x + shortcut  # Add the original input back
        # 中文：残差相加，缓解深层网络的梯度消失问题

        # Shortcut connection for feed-forward block
        # 中文：前馈子层的残差连接，结构与上面完全对称
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的 GPT 语言模型（第 4 章）。

    结构：token embedding + 位置 embedding -> dropout -> 堆叠若干个
    TransformerBlock -> 最终 LayerNorm -> 线性输出头（映射到词表维度，得到
    每个位置对下一个 token 的预测 logits）。

    参数:
        cfg (dict): 配置字典，需包含 vocab_size、emb_dim、context_length、
            drop_rate、n_layers 等键。

    张量形状:
        输入 in_idx:  (batch_size, seq_len)，元素为 token id（整数）。
        输出 logits:  (batch_size, seq_len, vocab_size)。
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # 中文：按配置堆叠 n_layers 个相同结构（参数各自独立）的 Transformer 块

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 中文：输出头把最终隐藏状态映射回词表大小，得到每个位置的下一个 token 预测分布（logits）

    def forward(self, in_idx):
        # 中文：in_idx 形状 (batch_size, seq_len)，元素是 token id
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文：位置编码使用可学习的绝对位置 embedding，索引为 0..seq_len-1
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：token embedding 与位置 embedding 相加，为模型注入序列位置信息
        x = self.drop_emb(x)
        x = self.trf_blocks(x)     # 中文：依次经过所有 Transformer 块
        x = self.final_norm(x)     # 中文：最终归一化，稳定输出分布
        logits = self.out_head(x)  # 中文：映射到词表维度，得到未归一化的预测分数
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """最简单的贪心解码文本生成函数（第 4 章）。

    每一步都取模型输出中概率（logits）最大的 token 作为下一个 token
    （即贪心搜索，不做采样/温度/top-k 等处理），并将其拼接到已有序列末尾，
    循环 max_new_tokens 次。

    参数:
        model: GPTModel 实例，前向传播输出 logits。
        idx (torch.Tensor): 形状 (batch_size, num_tokens) 的当前上下文 token id。
        max_new_tokens (int): 要生成的新 token 数量。
        context_size (int): 模型支持的最大上下文长度，用于裁剪过长的输入。

    返回:
        torch.Tensor: 形状 (batch_size, num_tokens + max_new_tokens) 的完整序列。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 中文：只保留最近 context_size 个 token 作为输入，避免超过模型支持的最大长度
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # 中文：生成阶段不需要计算梯度，用 no_grad 节省显存/加速
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 中文：只关心序列最后一个位置的预测（即"下一个 token"的分布）
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 中文：贪心选择：直接取 logits 最大的词表索引，而非按概率采样
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 中文：把新生成的 token 拼接到序列末尾，作为下一轮的输入上下文
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# Chapter 5
#####################################
def assign(left, right):
    """将 numpy/torch 权重值安全地包装成可训练参数（第 5 章）。

    用于把从 OpenAI 官方 GPT-2 checkpoint 中读取出来的权重（right，通常是
    numpy 数组）赋值给我们自己模型中对应的参数（left，nn.Parameter），
    赋值前会先检查两者形状是否一致，避免因权重维度错位导致的隐蔽 bug。

    参数:
        left (torch.Tensor/nn.Parameter): 模型中原有的参数，仅用来对比形状。
        right (np.ndarray 或类似结构): 待加载的预训练权重数值。

    返回:
        torch.nn.Parameter: 包装好的新参数，形状与 left 相同、数值来自 right。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """把 OpenAI 官方发布的 GPT-2 预训练权重加载进自定义的 GPTModel（第 5 章）。

    `params` 是从 OpenAI 官方 TensorFlow checkpoint 解析出来的嵌套字典结构
    （典型结构包含 "wte"/"wpe" 词嵌入与位置嵌入、"blocks" 列表，每个 block
    内含注意力 "attn"、前馈 "mlp"、层归一化 "ln_1"/"ln_2" 等权重）。
    由于 OpenAI 的实现把 Q/K/V 三个投影矩阵合并存成了一个 "c_attn"，
    这里需要手动用 np.split 沿最后一维拆成三份，再分别赋值给我们模型中
    独立的 W_query/W_key/W_value 线性层；同时注意 TensorFlow 的线性层权重
    与 PyTorch nn.Linear 的权重是转置关系，所以多处用了 `.T`。

    参数:
        gpt (GPTModel): 待加载权重的目标模型实例（会被原地修改）。
        params (dict): 解析自 OpenAI GPT-2 checkpoint 的权重字典。

    返回:
        None（原地修改 gpt 的参数）。
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # 中文：OpenAI 把 Q/K/V 的权重矩阵拼接存成一个 c_attn，
        # 沿最后一维（axis=-1）平均切成三份，分别对应 query/key/value
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        # 中文：TensorFlow Conv1D 风格权重与 PyTorch nn.Linear 权重互为转置，故需要 .T
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 中文：偏置同理，也拆成 q/k/v 三份
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 中文：注意力输出投影层（对应 MultiHeadAttention.out_proj）
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 中文：前馈网络第一层（升维线性层），对应 FeedForward.layers[0]
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        # 中文：前馈网络第二层（降维线性层），对应 FeedForward.layers[2]
        # （layers[1] 是 GELU，无参数，故索引直接跳到 2）
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 中文：两个 LayerNorm 的缩放(scale/g)与平移(shift/b)参数
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 中文：最终归一化层与输出头。GPT-2 采用权重共享（weight tying），
    # 输出头 out_head 的权重直接复用词嵌入 wte，而不是单独训练一套参数
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    """将原始文本编码为带 batch 维度的 token id 张量（第 5 章辅助函数）。

    参数:
        text (str): 待编码的原始文本。
        tokenizer: 具备 `encode` 方法的分词器。

    返回:
        torch.Tensor: 形状 (1, num_tokens) 的 token id 张量（batch_size=1）。
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    # 中文：unsqueeze(0) 在最前面新增一个 batch 维，形状从 (num_tokens,) 变为 (1, num_tokens)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将 token id 张量解码回可读文本（第 5 章辅助函数）。

    参数:
        token_ids (torch.Tensor): 形状 (1, num_tokens) 的 token id 张量（batch_size=1）。
        tokenizer: 具备 `decode` 方法的分词器。

    返回:
        str: 解码后的文本字符串。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    # 中文：squeeze(0) 去掉 batch 维，形状从 (1, num_tokens) 变回 (num_tokens,)
    return tokenizer.decode(flat.tolist())
