# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-6.
# This file can be run as a standalone script.

"""
中文模块说明
============
本文件是《从零构建大语言模型》("Build a Large Language Model From Scratch")
一书第 7 章（指令微调 / Instruction Fine-Tuning）代码的"前置依赖模块"。

它把第 2~6 章中逐步搭建的核心组件汇总到一起，方便第 7 章的脚本直接
`from previous_chapters import ...` 复用，而不需要在每一章重复粘贴代码。

具体包含：
    - 第 2 章：文本 -> Token 的数据集与 DataLoader 构建（GPTDatasetV1 / create_dataloader_v1）
    - 第 3 章：多头自注意力机制（MultiHeadAttention）
    - 第 4 章：LayerNorm、GELU 激活函数、前馈网络、Transformer Block、
               完整的 GPTModel 结构，以及最朴素的贪心解码 generate_text_simple
    - 第 5 章：带温度采样 / top-k 采样的文本生成函数 generate、
               训练循环 train_model_simple、评估函数、
               OpenAI 预训练权重加载工具（assign / load_weights_into_gpt）、
               文本与 Token ID 互转的小工具、损失计算函数、训练曲线绘制函数

本文件只在注释层面做了中文补充，未改动任何可执行代码、变量名、函数签名、
逻辑或缩进，原有英文注释全部保留。
"""


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    """GPT 风格的自回归语言建模数据集（第 2 章）。

    利用滑动窗口把长文本切分成若干个长度为 max_length 的重叠序列，
    每个样本的目标序列（target）就是输入序列（input）整体右移一位，
    从而构造出"预测下一个 token"的训练样本。

    参数：
        txt (str): 原始训练文本（未分词的字符串）。
        tokenizer: 具备 encode 方法的分词器（本文件中使用 tiktoken 的 gpt2 编码器）。
        max_length (int): 每个输入/目标序列的 token 长度（即上下文窗口大小）。
        stride (int): 滑动窗口每次移动的步长；stride < max_length 时窗口之间会重叠。

    属性：
        input_ids (List[Tensor]): 每个元素形状为 (max_length,) 的输入 token 序列。
        target_ids (List[Tensor]): 每个元素形状为 (max_length,) 的目标 token 序列，
            即 input_ids 整体右移一位后的结果。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 中文：先把整段文本一次性编码成 token id 列表；
        # allowed_special 允许文本中出现特殊标记 <|endoftext|> 而不报错。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 中文：用步长为 stride 的滑动窗口，把长 token 序列切分成长度为 max_length 的小块。
        # 每次取 [i, i+max_length) 作为输入，[i+1, i+max_length+1) 作为目标（整体右移一位），
        # 这样模型的训练目标就是"给定前文，预测下一个 token"。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集中样本（切分出的序列块）的总数量。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """按索引取出一对 (输入序列, 目标序列)，形状均为 (max_length,)。"""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """构建用于 GPT 预训练的 DataLoader（第 2 章）。

    内部会先用 gpt2 分词器把文本编码，再交给 GPTDatasetV1 做滑动窗口切分，
    最后包装成 PyTorch 的 DataLoader 以支持批量、打乱、多进程加载等功能。

    参数：
        txt (str): 原始训练文本。
        batch_size (int): 每个 batch 的样本数。
        max_length (int): 上下文窗口长度（每条序列的 token 数）。
        stride (int): 滑动窗口步长。
        shuffle (bool): 是否打乱样本顺序。
        drop_last (bool): 是否丢弃最后一个不满 batch_size 的批次（训练时常设 True 以保持批次形状一致）。
        num_workers (int): 数据加载的子进程数。

    返回：
        DataLoader: 每次迭代产出 (input_batch, target_batch)，
            形状均为 (batch_size, max_length)。
    """
    # Initialize the tokenizer
    # 中文：使用 GPT-2 的 BPE 分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    # 中文：构造滑动窗口数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # 中文：包装成 DataLoader，负责批处理、打乱顺序等
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """多头因果自注意力模块（第 3 章）。

    将输入投影为 Query/Key/Value 三组张量，拆分成多个注意力头并行计算
    带因果掩码（causal mask）的缩放点积注意力，再把各头的输出拼接、
    经过一次线性投影得到最终的上下文向量。

    参数：
        d_in (int): 输入特征维度。
        d_out (int): 输出特征维度（也是 Q/K/V 投影后的总维度）。
        context_length (int): 支持的最大序列长度，用于预生成因果掩码。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头数，需满足 d_out 能被 num_heads 整除。
        qkv_bias (bool): Q/K/V 线性层是否使用偏置项。

    前向输入/输出：
        输入 x 形状为 (batch, num_tokens, d_in)，
        输出 context_vec 形状为 (batch, num_tokens, d_out)。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：把总输出维度 d_out 均分给每个头，head_dim 即单个头的维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # 中文：out_proj 用于把多头拼接后的向量再做一次线性变换（混合各头信息）
        self.dropout = nn.Dropout(dropout)
        # 中文：预先构造上三角矩阵作为因果掩码（对角线以上为 1，表示"未来"位置需要被屏蔽），
        # 用 register_buffer 注册，使其随模型 .to(device) 移动但不参与梯度更新。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """前向传播：计算带因果掩码的多头自注意力。

        形状变化：
            x: (b, num_tokens, d_in)
            -> keys/queries/values: (b, num_tokens, d_out)
            -> view: (b, num_tokens, num_heads, head_dim)
            -> transpose: (b, num_heads, num_tokens, head_dim)
            -> attn_scores: (b, num_heads, num_tokens, num_tokens)
            -> context_vec: (b, num_tokens, d_out)（合并所有头之后）
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆分为 (num_heads, head_dim)，即"隐式"地切分出多个头
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度提前，方便后续对每个头独立做矩阵乘法（批量矩阵乘法会广播 batch 和 head 维度）
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 中文：Q 与 K^T 做点积，得到每个 token 对其它 token 的注意力得分
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # attn_scores 形状: (b, num_heads, num_tokens, num_tokens)

        # Original mask truncated to the number of tokens and converted to boolean
        # 中文：把预先注册的最大长度掩码截取到当前实际序列长度，并转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 中文：将掩码为 True（即"未来"位置）的注意力得分置为 -inf，
        # 这样 softmax 后这些位置的权重会变为 0，从而保证因果性（不能看到未来 token）
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：缩放点积注意力——除以 sqrt(head_dim) 防止点积值过大导致 softmax 梯度消失
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：用注意力权重对 V 加权求和，再把 head 维度换回第 2 维，方便后续拼接
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多头的输出在最后一维拼接回 d_out，得到与输入等价形状的上下文向量
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：额外的线性投影层，让模型能够融合各个头提取到的不同信息

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化模块（第 4 章）。

    对最后一维（特征维度）做均值方差归一化，并用可学习的
    缩放（scale）和偏移（shift）参数恢复模型的表达能力。

    参数：
        emb_dim (int): 特征维度大小，即需要归一化的最后一维长度。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除以零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 中文：可学习的缩放参数，初始为 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 中文：可学习的偏移参数，初始为 0

    def forward(self, x):
        """对输入 x 的最后一维做归一化。

        x 形状: (..., emb_dim)，输出形状不变。
        """
        mean = x.mean(dim=-1, keepdim=True)
        # 中文：使用 unbiased=False（即除以 N 而非 N-1）与 GPT-2 原始实现保持一致
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数的近似实现（tanh 近似版本，与 GPT-2 一致）（第 4 章）。"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """按 GPT-2 论文中的 tanh 近似公式计算 GELU 激活值，形状与输入相同。"""
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer Block 中的逐位置前馈网络（第 4 章）。

    结构为：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)，
    先升维再降维，增加模型的非线性表达能力。

    参数：
        cfg (dict): 模型配置字典，需包含键 "emb_dim"。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 中文：升维到 4 倍
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 中文：再降维回原始维度
        )

    def forward(self, x):
        """前向传播，输入输出形状均为 (batch, num_tokens, emb_dim)。"""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """标准的 GPT Transformer Block（第 4 章）。

    结构为 Pre-LayerNorm 形式：
        x = x + Dropout(Attention(LayerNorm(x)))
        x = x + Dropout(FeedForward(LayerNorm(x)))
    每个子层都带有残差连接（shortcut），有利于深层网络的梯度传播。

    参数：
        cfg (dict): 模型配置字典，需包含 emb_dim、context_length、n_heads、
            drop_rate、qkv_bias 等键。
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
        """前向传播：依次经过"注意力子层 + 残差"和"前馈子层 + 残差"。

        输入输出形状均为 (batch, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        # 中文：保存原始输入用于残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差相加，缓解深层网络梯度消失问题

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的 GPT 模型（第 4 章）。

    结构：Token Embedding + 位置 Embedding -> Dropout -> N 层 TransformerBlock
    -> 最终 LayerNorm -> 线性输出层（映射到词表大小，得到 logits）。

    参数：
        cfg (dict): 模型配置字典，需包含：
            vocab_size, emb_dim, context_length, drop_rate, n_layers 等键。
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # 中文：堆叠 n_layers 个 Transformer Block

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 中文：输出头把隐藏状态映射回词表维度，得到每个位置对下一个 token 的预测分布（logits）

    def forward(self, in_idx):
        """前向传播，计算下一个 token 的预测 logits。

        参数：
            in_idx (Tensor): 形状 (batch_size, seq_len) 的输入 token id 序列。

        返回：
            logits (Tensor): 形状 (batch_size, seq_len, vocab_size)。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文：位置编码使用 0..seq_len-1 的下标查表，得到与序列等长的位置向量
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：token embedding 与位置 embedding 相加（利用广播机制），融合"是什么词"和"在哪个位置"两类信息
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """最朴素的自回归文本生成函数：贪心解码（第 4 章）。

    每一步都选取概率（logits）最大的 token 作为下一个 token，
    不涉及随机采样或温度控制。

    参数：
        model (GPTModel): 已训练好的 GPT 模型。
        idx (Tensor): 形状 (B, T) 的初始上下文 token id。
        max_new_tokens (int): 需要新生成的 token 数量。
        context_size (int): 模型支持的最大上下文长度（超出部分会被截断）。

    返回：
        Tensor: 形状 (B, T + max_new_tokens) 的完整 token 序列。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 中文：只保留最近 context_size 个 token 作为模型输入，避免超出模型支持的最大长度
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # 中文：推理阶段不需要计算梯度，节省显存并加速
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 中文：只关心序列最后一个位置的预测，因为它对应"下一个 token"的分布
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 中文：贪心解码——直接取概率最大的 token id
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 中文：把新生成的 token 拼接到序列末尾，作为下一轮迭代的上下文
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# Chapter 5
#####################################
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """支持温度采样与 top-k 采样的文本生成函数（第 5 章）。

    相比 generate_text_simple，这里增加了：
        1. top_k 采样：只在概率最高的 k 个候选中采样，过滤掉低概率的"长尾"token；
        2. 温度缩放：temperature > 0 时对 logits 做缩放后按概率分布采样，
           temperature 越大生成结果越随机，越小越接近贪心解码；
        3. 提前停止：遇到 eos_id 指定的结束符时提前终止生成。

    参数：
        model (GPTModel): 已训练好的 GPT 模型。
        idx (Tensor): 形状 (batch_size, T) 的初始上下文 token id。
        max_new_tokens (int): 最多新生成的 token 数。
        context_size (int): 模型支持的最大上下文长度。
        temperature (float): 采样温度，0.0 表示退化为贪心解码。
        top_k (int, optional): 只保留 top_k 个最高概率的候选 token。
        eos_id (int, optional): 结束符 token id，遇到即停止生成。

    返回：
        Tensor: 形状 (batch_size, T + 实际生成的 token 数) 的 token 序列。
    """

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        # 中文：top-k 采样——只保留 logits 最大的 k 个值，其余全部置为 -inf，
        # 这样 softmax 后这些低概率候选的权重会变为 0，避免采样到明显不合理的词
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        # 中文：温度缩放——logits 除以 temperature，
        # temperature < 1 会让分布更尖锐（更确定），temperature > 1 会让分布更平缓（更随机）
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            # 中文：减去每行最大值再做 softmax，是常见的数值稳定技巧，
            # 在 mps 设备上能避免精度问题导致结果与其它设备不一致
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            # 中文：按概率分布随机采样一个 token，而非总是取最大值，增加生成多样性
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """朴素的训练循环（第 5 章）。

    对模型做 num_epochs 轮训练，每个 batch 执行标准的
    "前向 -> 计算损失 -> 反向传播 -> 参数更新"流程，
    并周期性地在训练集/验证集上评估损失，每个 epoch 结束后打印一段生成样本。

    参数：
        model (GPTModel): 待训练的模型。
        train_loader / val_loader (DataLoader): 训练 / 验证数据加载器。
        optimizer: PyTorch 优化器。
        device: 训练所用设备（cpu/cuda/mps）。
        num_epochs (int): 训练轮数。
        eval_freq (int): 每隔多少个全局训练步做一次评估。
        eval_iter (int): 每次评估时使用的 batch 数量。
        start_context (str): 每个 epoch 结束后用于生成示例文本的起始提示词。
        tokenizer: 分词器，用于编码/解码生成示例文本。

    返回：
        (train_losses, val_losses, track_tokens_seen): 三个列表，
        分别记录评估时刻的训练损失、验证损失，以及累计已训练的 token 数，
        便于后续用 plot_losses 绘制训练曲线。
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # 中文：切换到训练模式，启用 dropout 等训练专属行为

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            # 中文：每隔 eval_freq 步做一次评估，记录损失曲线，便于监控训练/验证损失的变化趋势
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        # 中文：每个 epoch 结束后打印一段生成文本，直观感受模型当前的生成质量
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """在训练集和验证集上分别计算平均损失，用于训练过程中的监控（第 5 章）。

    参数：
        model (GPTModel): 待评估的模型。
        train_loader / val_loader (DataLoader): 训练 / 验证数据加载器。
        device: 计算设备。
        eval_iter (int): 每个数据集用于评估的 batch 数量（避免遍历整个数据集，节省时间）。

    返回：
        (train_loss, val_loss): 两个浮点数，分别是训练集和验证集上的平均交叉熵损失。
    """
    model.eval()
    # 中文：评估模式下关闭 dropout 等随机性，保证评估结果稳定可复现
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    # 中文：评估结束后切回训练模式，继续后续训练
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """基于给定起始文本生成一段样例并打印，用于直观查看训练效果（第 5 章）。

    参数：
        model (GPTModel): 当前模型。
        tokenizer: 分词器。
        device: 计算设备。
        start_context (str): 生成的起始提示文本。
    """
    model.eval()
    # 中文：从模型的位置编码 Embedding 层权重形状中取出 context_size（第一维即支持的最大位置数）
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def assign(left, right):
    """将 numpy/常规数组形式的权重（right）赋值给模型参数（left），并做形状校验。

    参数：
        left (Tensor): 模型中原有的参数张量（仅用于读取 shape 做比对）。
        right (array-like): 待加载的预训练权重（如从 OpenAI 官方 GPT-2 checkpoint 读取的 numpy 数组）。

    返回：
        torch.nn.Parameter: 用 right 的数值构造的新参数，形状与 left 一致。

    异常：
        ValueError: 当 left 与 right 的形状不一致时抛出，避免错误地加载权重。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """把 OpenAI 官方发布的 GPT-2 预训练权重（params）加载到自定义 GPTModel（gpt）中（第 5 章）。

    params 是从 OpenAI TensorFlow checkpoint 解析出的嵌套字典结构，
    本函数负责把其中的权重矩阵/偏置按照本项目 GPTModel 的模块命名与形状
    一一对应地拷贝过去（注意 TensorFlow 的 Linear 权重与 PyTorch 的 nn.Linear
    权重是转置关系，所以很多地方用了 .T）。

    参数：
        gpt (GPTModel): 待加载权重的模型实例（结构需与 GPT-2 一致）。
        params (dict): OpenAI 预训练权重的嵌套字典，包含 wpe/wte、
            每层 blocks 的 attn/mlp/ln 权重，以及最终的 g/b（LayerNorm 参数）。
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # 中文：GPT-2 原始实现把 Q/K/V 的权重合并存储在一个 c_attn 矩阵里，
        # 这里沿最后一维（axis=-1）把它拆分成三份，分别对应 Q、K、V
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 中文：同理拆分 Q/K/V 对应的偏置项
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 中文：注意力输出投影层（对应 GPT-2 中的 c_proj）
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 中文：前馈网络第一层（升维，对应 GPT-2 的 c_fc）
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        # 中文：前馈网络第二层（降维，对应 GPT-2 的 c_proj）
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 中文：两个 LayerNorm（注意力前、前馈网络前）的缩放/偏移参数
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

    # 中文：最终的 LayerNorm 参数，以及输出头权重
    # 注意 GPT-2 采用了权重共享（weight tying）：输出层权重与 token embedding 权重相同
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    """将字符串文本编码为模型输入所需的 token id 张量。

    参数：
        text (str): 原始文本。
        tokenizer: 分词器。

    返回：
        Tensor: 形状 (1, seq_len)，第 0 维是新增的 batch 维度。
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    # 中文：增加一个 batch 维度，使其符合模型 (batch, seq_len) 的输入格式要求
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将模型输出的 token id 张量解码回可读文本。

    参数：
        token_ids (Tensor): 形状 (1, seq_len) 的 token id 张量。
        tokenizer: 分词器。

    返回：
        str: 解码后的文本。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    # 中文：去掉 batch 维度后转成 python list，再交给分词器解码
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个 batch 的交叉熵损失（第 5 章）。

    参数：
        input_batch (Tensor): 形状 (batch_size, seq_len) 的输入 token。
        target_batch (Tensor): 形状 (batch_size, seq_len) 的目标 token。
        model (GPTModel): 模型。
        device: 计算设备。

    返回：
        Tensor: 标量损失值。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # 中文：把 (batch, seq_len, vocab_size) 展平为 (batch*seq_len, vocab_size)，
    # 目标同样展平为 (batch*seq_len,)，以匹配 cross_entropy 的输入要求，
    # 相当于把"每个位置的下一个 token 预测"当作独立的多分类问题
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """遍历 DataLoader 中若干个 batch，计算平均交叉熵损失（第 5 章）。

    参数：
        data_loader (DataLoader): 数据加载器。
        model (GPTModel): 模型。
        device: 计算设备。
        num_batches (int, optional): 只计算前 num_batches 个 batch 的损失；
            为 None 时计算整个 data_loader。

    返回：
        float: 平均损失；若 data_loader 为空则返回 nan。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # 中文：防止用户传入的 num_batches 超过数据加载器实际的 batch 数
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制训练/验证损失曲线，并同时展示"已训练 token 数"作为第二 x 轴（第 5 章）。

    参数：
        epochs_seen (List[float]): 每次记录点对应的 epoch 数（可以是小数，表示 epoch 内的某个时刻）。
        tokens_seen (List[int]): 每次记录点对应的累计已训练 token 数。
        train_losses (List[float]): 对应的训练集损失。
        val_losses (List[float]): 对应的验证集损失。

    副作用：
        将图像保存为 "loss-plot.pdf" 并调用 plt.show() 展示。
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    # 中文：主坐标轴（下方 x 轴）以 epoch 为单位画出训练/验证损失曲线
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    # 中文：新增一个共享 y 轴、位于图像上方的第二 x 轴，用来标注"已训练 token 数"，
    # 方便同时从 epoch 和 token 两个维度理解训练进度
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    # 中文：这里画一条透明（alpha=0）的曲线，只是为了让 matplotlib 根据 tokens_seen 的数值范围
    # 自动生成第二 x 轴的刻度，并不会在图上真正显示出来
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()
