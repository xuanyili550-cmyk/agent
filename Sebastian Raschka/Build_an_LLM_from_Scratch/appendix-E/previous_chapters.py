# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-6.
# This file can be run as a standalone script.

# ============================================================
# 中文模块说明（模块级 docstring）
# ------------------------------------------------------------
# 本文件是《从零构建大语言模型》(Build a Large Language Model From
# Scratch) 一书附录 E (appendix-E) 所依赖的"前置代码合集"。
#
# 它把第 2 章到第 6 章中逐步讲解、逐步搭建起来的核心组件汇总到一个
# 文件里，方便附录 E（讲解如 LoRA 等参数高效微调 / 分类微调进阶技巧）
# 直接 `from previous_chapters import ...` 导入使用，而不用重复贴代码。
#
# 具体包含（按章节顺序）：
#   - 第 2 章：GPTDatasetV1、create_dataloader_v1
#       —— 用滑动窗口把原始文本切分成 (输入, 目标) 训练样本对，
#          构建 PyTorch DataLoader。
#   - 第 3 章：MultiHeadAttention
#       —— 带因果掩码（causal mask）的多头自注意力机制，是 Transformer
#          的核心构件。
#   - 第 4 章：LayerNorm、GELU、FeedForward、TransformerBlock、
#             GPTModel、generate_text_simple
#       —— 组装出完整的 GPT 架构（层归一化 + 注意力 + 前馈网络 +
#          残差连接），以及最朴素的贪心解码生成函数。
#   - 第 5 章：assign、load_weights_into_gpt、text_to_token_ids、
#             token_ids_to_text、calc_loss_loader、evaluate_model
#       —— 预训练相关：把 OpenAI 官方发布的 GPT-2 权重加载进自己实现
#          的模型里，文本与 token id 的相互转换，以及损失计算/评估。
#   - 第 6 章：download_and_unzip_spam_data、create_balanced_dataset、
#             random_split、SpamDataset、calc_accuracy_loader、
#             calc_loss_batch、train_classifier_simple、plot_values
#       —— 垃圾邮件（spam）二分类微调任务：数据下载、类别平衡、
#          数据集划分、Dataset 封装、训练循环、准确率评估与画图。
#
# 阅读本文件时可以把它当作"知识地图"：想复习某一章的实现细节，
# 直接跳到对应的 "##### Chapter N #####" 分隔注释处即可。
# ============================================================

import os
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    """第 2 章：GPT 预训练用的滑动窗口数据集。

    把一整段原始文本（txt）用 tokenizer 编码成 token id 序列后，
    使用固定长度 max_length、固定步长 stride 的滑动窗口切分成
    多个训练样本；每个样本的 target（预测目标）就是 input 整体
    向右平移一位（"预测下一个 token"这一自回归训练目标）。

    参数：
        txt: 原始文本字符串。
        tokenizer: 具备 .encode() 方法的分词器（如 tiktoken 的 GPT-2 编码器）。
        max_length: 每个训练样本的 token 序列长度（即上下文窗口大小）。
        stride: 相邻两个样本起始位置之间的步长；stride < max_length 时
            样本之间会有重叠，stride == max_length 时样本不重叠。

    产出：
        self.input_ids / self.target_ids 是两个等长的列表，
        每个元素都是形状为 (max_length,) 的一维 LongTensor。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 先把整段文本一次性编码为 token id 列表；<|endoftext|> 是 GPT-2
        # 用来分隔不同文档的特殊 token，这里显式允许它被编码。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 用滑动窗口把长文本切成多个 (max_length,) 长度的重叠片段：
        # i 每次前进 stride 步；input_chunk 取 [i, i+max_length)，
        # target_chunk 是 input_chunk 整体右移一位，即 [i+1, i+max_length+1)。
        # 这样 target 的第 t 个位置正好是 input 第 t 个位置要预测的"下一个 token"。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集中样本（切分出的窗口）总数。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """按索引取出一对 (input_ids, target_ids)，形状均为 (max_length,)。"""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True):
    """第 2 章：快捷函数——把原始文本一步转换为可迭代的 DataLoader。

    内部固定使用 GPT-2 的 BPE 分词器（tiktoken "gpt2"），先构建
    GPTDatasetV1，再包装成标准的 torch DataLoader。

    参数：
        txt: 原始文本。
        batch_size: 每个 batch 的样本数。
        max_length: 每个样本的 token 序列长度（上下文窗口）。
        stride: 滑动窗口步长。
        shuffle: 是否打乱样本顺序。
        drop_last: 是否丢弃最后一个不满 batch_size 的不完整批次
            （训练时常设 True，避免 batch 尺寸不一致带来的问题）。

    返回：
        一个 torch.utils.data.DataLoader，每次迭代产出
        (input_batch, target_batch)，形状均为 (batch_size, max_length)。
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """第 3 章：带因果掩码的多头自注意力（Multi-Head Self-Attention）。

    这是 GPT 这类"仅解码器"（decoder-only）Transformer 的核心模块：
    通过 Q（query）/K（key）/V（value）三个线性投影，计算 token 之间
    两两的相关性得分（注意力分数），再用因果掩码屏蔽"看到未来 token"
    的路径，从而实现自回归语言建模。

    参数：
        d_in: 输入向量维度（embedding 维度）。
        d_out: 输出向量维度，同时也是 Q/K/V 投影后的总维度，
            会被均分到每个注意力头上。
        context_length: 支持的最大序列长度，用来预先构建因果掩码矩阵。
        dropout: 注意力权重上的 dropout 概率。
        num_heads: 注意力头数；要求 d_out 能被 num_heads 整除。
        qkv_bias: Q/K/V 线性层是否使用偏置项（bias）。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头分到的维度 = 总输出维度 / 头数，
        # 这样多头拼接回去正好还原成 d_out，不额外增加参数量。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        # 中文：register_buffer 注册的 mask 不是可训练参数，但会随模型
        # .to(device) 一起搬到 GPU/CPU；torch.triu(..., diagonal=1) 生成
        # 一个上三角（不含对角线）为 1 的矩阵，1 的位置代表"未来 token"，
        # 后面会用它把对应位置的注意力分数置为 -inf，实现因果（causal）掩码。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """前向传播。

        输入 x 形状: (b, num_tokens, d_in)，其中 b 为 batch size，
        num_tokens 为当前序列长度。
        输出形状: (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把 d_out 这一维"拆分"成 (num_heads, head_dim)，
        # 相当于把一次大矩阵投影结果切成多份，分给每个头独立处理。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 提到 num_tokens 前面，方便后续对每个头
        # 独立做 (num_tokens, head_dim) x (head_dim, num_tokens) 的矩阵乘法。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 中文：queries @ keys^T 得到每个头内部、每对 token 之间的原始注意力分数，
        # 形状为 (b, num_heads, num_tokens, num_tokens)。
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        # 中文：因为 mask 是按 context_length 预先建好的最大尺寸，
        # 这里裁剪到当前实际序列长度 num_tokens，避免尺寸不匹配。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 中文：把"未来位置"(mask 为 True 处) 的注意力分数填为 -inf，
        # 这样后面 softmax 之后这些位置的权重会变成 0，
        # 保证每个 token 只能"看到"自己和它之前的 token（因果性）。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：除以 sqrt(head_dim) 做缩放（scaled dot-product），
        # 防止维度较大时点积数值过大导致 softmax 梯度消失。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：注意力权重与 values 加权求和，得到每个 token 的上下文向量；
        # transpose(1, 2) 把维度顺序换回 (b, num_tokens, num_heads, head_dim)，
        # 便于下一步把多头拼接在一起。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把 (num_heads, head_dim) 两维合并回 d_out，
        # 即把各个头的输出拼接（concat）成一个整体向量。
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：out_proj 是多头注意力最后的线性混合层，
        # 让不同头之间的信息可以进一步融合。

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """第 4 章：层归一化（Layer Normalization）。

    对每个 token 自身的 embedding 向量（最后一维）做归一化：
    减去均值、除以标准差，再用可学习的 scale（缩放）和 shift（平移）
    参数进行仿射变换。这能让每一层的输入分布更稳定，帮助训练收敛。

    参数：
        emb_dim: embedding 维度，也是 scale/shift 参数的长度。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除以 0 的极小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """输入/输出形状均为 (..., emb_dim)，在最后一维上做归一化。"""
        mean = x.mean(dim=-1, keepdim=True)
        # 中文：unbiased=False 表示使用有偏方差估计（除以 N 而非 N-1），
        # 与常见深度学习框架中 LayerNorm 的默认实现保持一致。
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """第 4 章：GELU 激活函数（这里用的是其 tanh 近似公式，与 GPT-2 一致）。

    GELU 相比 ReLU 更平滑，在 Transformer 的前馈网络中被广泛使用。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """输入/输出形状相同（逐元素激活函数）。"""
        # 中文：0.5x * (1 + tanh( sqrt(2/π) * (x + 0.044715 x^3) ))
        # 是 GELU 的近似解析式，比直接计算高斯误差函数 erf 更快。
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """第 4 章：Transformer block 内部的前馈网络（FFN / MLP）。

    结构为：线性升维（emb_dim -> 4*emb_dim）-> GELU 激活
    -> 线性降维（4*emb_dim -> emb_dim）。
    "先升维再降维"的设计让网络有更大的中间表示空间去做非线性变换。

    参数：
        cfg: 配置字典，需包含 "emb_dim" 键。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """输入形状 (b, num_tokens, emb_dim)，输出同形状。"""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """第 4 章：单个 Transformer block（GPT 风格，Pre-LayerNorm 结构）。

    每个 block 由两个子层组成，均带残差连接（shortcut）：
      1) 多头自注意力子层：LayerNorm -> MultiHeadAttention -> Dropout -> 残差相加
      2) 前馈网络子层：LayerNorm -> FeedForward -> Dropout -> 残差相加
    "Pre-Norm"指 LayerNorm 放在子层之前，这是 GPT-2 及之后大多数
    LLM 采用的结构，有利于训练更深的网络。

    参数：
        cfg: 配置字典，需包含 emb_dim / context_length / n_heads /
            drop_rate / qkv_bias 等键。
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
        """输入/输出形状均为 (batch_size, num_tokens, emb_dim)。"""
        # Shortcut connection for attention block
        # 中文：先保存原始输入，供残差连接使用；
        # 残差连接能缓解深层网络的梯度消失问题，让信息/梯度可以直接跨层传递。
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """第 4 章：完整的 GPT 模型（仅解码器 Transformer 架构）。

    结构：token embedding + 可学习的位置 embedding 相加 -> Dropout
    -> N 层 TransformerBlock 堆叠 -> 最终 LayerNorm -> 线性输出头
    （投影到词表大小，得到每个位置对下一个 token 的预测 logits）。

    参数：
        cfg: 配置字典，需包含 vocab_size / emb_dim / context_length /
            drop_rate / n_layers 等键。
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """前向传播。

        输入 in_idx 形状: (batch_size, seq_len)，元素为 token id（整数）。
        输出 logits 形状: (batch_size, seq_len, vocab_size)，
        表示模型对每个位置"下一个 token"的未归一化预测分数。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文：位置编码使用可学习的绝对位置 embedding（区别于原始
        # Transformer 论文里固定的正弦/余弦编码），按 0..seq_len-1 取值。
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：token embedding 与 position embedding 直接相加（而非拼接），
        # 这样输出维度仍是 emb_dim，且模型能同时感知"是什么词"和"在哪个位置"。
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """第 4 章：最朴素的自回归文本生成（贪心解码，greedy decoding）。

    每一步都取模型输出中概率（logits）最大的 token 作为下一个 token，
    不使用任何采样随机性（如温度、top-k、top-p），因此生成结果是确定性的。

    参数：
        model: GPTModel 实例。
        idx: 初始上下文 token id，形状 (batch_size, num_tokens)。
        max_new_tokens: 要新生成的 token 数量。
        context_size: 模型支持的最大上下文长度，用于裁剪过长的输入。

    返回：
        扩展后的 token id 序列，形状 (batch_size, num_tokens + max_new_tokens)。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 中文：只保留最近 context_size 个 token 作为输入，
        # 因为位置 embedding 和注意力掩码都是按 context_size 构建的，
        # 序列过长会超出模型能处理的范围。
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # 中文：生成阶段不需要反向传播，用 no_grad 节省显存、加速推理。
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 中文：只关心序列最后一个位置的预测，因为它才是"下一个新 token"的分布。
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 中文：贪心解码——直接选 logits 最大的那个词，不做随机采样。
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 中文：把新生成的 token 拼接到序列末尾，作为下一步生成的上下文。
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# Chapter 5
#####################################
def assign(left, right):
    """第 5 章：辅助函数——把 right（通常是 numpy 数组形式的预训练权重）
    包装成 nn.Parameter，并赋值给 left 对应的参数。

    在赋值前会校验两者的形状（shape）是否一致，形状不匹配时立即报错，
    避免"悄悄"把权重加载错位置这种难以察觉的 bug。

    参数：
        left: 目标模型中的参数（用于取得期望的 shape，仅做校验，不使用其数值）。
        right: 待加载的权重数据（如 numpy 数组）。

    返回：
        torch.nn.Parameter，包裹了 right 的数据，可直接赋值给模型属性。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """第 5 章：把 OpenAI 官方发布的 GPT-2 预训练权重（params，通常来自
    TensorFlow checkpoint 转换出的嵌套字典/数组结构）加载进我们自己
    用 PyTorch 实现的 GPTModel（gpt）中。

    核心难点在于两边参数的"命名方式"和"矩阵排列方式"不同：
      - OpenAI 的实现把 Q/K/V 的权重拼接（concat）在一起存成一个大矩阵
        c_attn，这里需要用 np.split 按最后一维切成三份分别对应
        W_query / W_key / W_value。
      - OpenAI 使用的 Linear 权重是 (in_features, out_features) 排列，
        而 PyTorch nn.Linear 的权重是 (out_features, in_features)，
        所以处处需要 .T 转置后再赋值。

    参数：
        gpt: 目标 GPTModel 实例（会被原地修改，替换其参数）。
        params: 从 OpenAI 官方 checkpoint 解析出的权重字典。

    返回：
        无（None）；通过原地赋值修改 gpt 的参数。
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # 中文：c_attn 里把 Q/K/V 三个权重矩阵在最后一维拼在一起存储，
        # 这里按最后一维（axis=-1）均分成三份，还原出各自的权重矩阵。
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 中文：偏置项同理，也是拼接存储，按最后一维切分为 Q/K/V 三份。
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 中文：c_proj 对应多头注意力最后的输出投影层 out_proj。
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 中文：mlp.c_fc 对应 FeedForward 的第一层线性层（升维），
        # mlp.c_proj 对应第三层线性层（降维），layers[1] 是无参数的 GELU。
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 中文：ln_1 / ln_2 分别对应注意力子层前、前馈子层前的两个 LayerNorm；
        # OpenAI 用 g（gain，即缩放）/ b（bias，即平移）命名，
        # 对应我们实现里的 scale / shift。
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

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 中文：GPT-2 的输出头（预测词表 logits 的线性层）与输入的
    # token embedding 共享权重（weight tying），因此这里直接复用 wte。
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    """第 5 章：把字符串文本编码为模型可用的 token id 张量。

    参数：
        text: 原始文本字符串。
        tokenizer: 分词器（需支持 .encode 方法）。

    返回：
        形状为 (1, num_tokens) 的 LongTensor（多出的第 0 维是 batch 维度）。
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """第 5 章：把模型输出/输入的 token id 张量解码回字符串文本。

    参数：
        token_ids: 形状为 (1, num_tokens) 的张量（batch 维度大小须为 1）。
        tokenizer: 分词器（需支持 .decode 方法）。

    返回：
        解码后的字符串。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """第 5 章：在整个（或部分）data_loader 上计算平均交叉熵损失。

    参数：
        data_loader: 产出 (input_batch, target_batch) 的 DataLoader。
        model: 语言模型。
        device: 计算设备（"cpu" / "cuda" / "mps" 等）。
        num_batches: 只评估前 num_batches 个 batch；为 None 时评估全部。

    返回：
        float 类型的平均损失；若 data_loader 为空则返回 nan。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # 中文：如果用户传入的 num_batches 比 data_loader 实际的 batch 数还大，
        # 就取两者中较小的一个，避免越界。
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """第 5 章：在训练/验证集上各评估 eval_iter 个 batch 的平均损失。

    评估前把模型切到 eval() 模式（关闭 dropout 等），
    并在 torch.no_grad() 下运行以节省显存、加快速度；
    评估结束后把模型切回 train() 模式，避免影响后续训练。

    参数：
        model: 语言模型。
        train_loader / val_loader: 训练集 / 验证集 DataLoader。
        device: 计算设备。
        eval_iter: 每个 loader 上抽样评估的 batch 数量。

    返回：
        (train_loss, val_loss) 二元组，均为 float。
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


#####################################
# Chapter 6
#####################################


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """第 6 章：下载并解压 UCI SMS Spam Collection 数据集（垃圾短信分类数据）。

    若目标数据文件已存在则直接跳过，避免重复下载。

    参数：
        url: 数据集 zip 文件的下载地址。
        zip_path: 下载后 zip 文件的本地保存路径。
        extracted_path: 解压目标目录。
        data_file_path: 最终期望得到的数据文件路径（Path 对象）。

    返回：
        无（None）；执行下载、解压、重命名等副作用操作。
    """
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    # 中文：stream=True + iter_content 分块写入，避免一次性把大文件读入内存。
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    # 中文：原始解压出来的文件没有扩展名，这里重命名并加上期望的文件名/后缀，
    # 方便后续用 pandas 按 tsv 格式读取。
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):
    """第 6 章：从原始（类别不平衡的）数据集中构造类别平衡的子集。

    垃圾短信数据集中 "ham"（正常短信）远多于 "spam"（垃圾短信），
    直接训练容易让分类器"偷懒"地把所有样本都预测为多数类。
    这里通过对 "ham" 做欠采样（下采样），使其数量与 "spam" 持平。

    参数：
        df: 包含 "Label" 列（取值为 "spam" 或 "ham"）的 pandas DataFrame。

    返回：
        类别数量 1:1 平衡后的新 DataFrame。
    """

    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham' instances to match the number of 'spam' instances
    # 中文：固定 random_state=123 保证每次运行采样结果可复现。
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    """第 6 章：把 DataFrame 随机打乱后按比例切分为训练/验证/测试三份。

    参数：
        df: 待切分的 DataFrame。
        train_frac: 训练集所占比例（0~1）。
        validation_frac: 验证集所占比例（0~1）；
            测试集比例 = 1 - train_frac - validation_frac。

    返回：
        (train_df, validation_df, test_df) 三元组。
    """
    # Shuffle the entire DataFrame
    # 中文：frac=1 表示对全部行做随机重排（相当于 shuffle），
    # reset_index(drop=True) 重建从 0 开始的连续索引。
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    """第 6 章：垃圾短信分类任务的 Dataset 封装。

    读取 csv 文件（需含 "Text" 和 "Label" 两列），把每条短信文本
    预先编码为 token id，并统一 padding/截断到相同长度 max_length，
    以便组成规则形状的 batch。

    参数：
        csv_file: 数据文件路径（csv 格式，含 Text/Label 列）。
        tokenizer: 分词器（需支持 .encode 方法）。
        max_length: 统一的序列长度；为 None 时自动取数据集中最长样本的长度。
        pad_token_id: 用于填充（padding）的 token id，默认用 GPT-2 的
            <|endoftext|> 对应 id（50256）来占位。
    """
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        # 中文：提前把所有文本编码好并缓存，避免每次取 batch 时重复分词，
        # 用空间换时间。
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            # 中文：超过 max_length 的样本直接截断，只保留前 max_length 个 token。
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        # 中文：不足 max_length 的样本在末尾补 pad_token_id，
        # 使所有样本长度一致，从而能堆叠（stack）成规则的 batch 张量。
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        """返回 (encoded, label)：
        encoded 形状为 (max_length,) 的 LongTensor；
        label 为标量 LongTensor（0/1 类别标签）。
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        """数据集样本总数。"""
        return len(self.data)

    def _longest_encoded_length(self):
        """遍历所有已编码文本，找出其中最长序列的长度，
        用作未显式指定 max_length 时的默认填充长度。
        """
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
        # Note: A more pythonic version to implement this method
        # is the following, which is also used in the next chapter:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)
        # 中文：等价的更 Pythonic 写法见上面注释，用生成器表达式 + max() 一行搞定。


@torch.no_grad()  # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """第 6 章：在 data_loader 上计算分类准确率（accuracy）。

    使用 @torch.no_grad() 装饰整个函数，因为评估阶段不需要梯度，
    这样可以省显存、加速。

    参数：
        data_loader: 产出 (input_batch, target_batch) 的 DataLoader，
            target_batch 是分类标签（0/1）。
        model: 分类模型（复用 GPTModel 结构，取最后一个 token 的 logits 做分类）。
        device: 计算设备。
        num_batches: 只评估前 num_batches 个 batch；为 None 时评估全部。

    返回：
        float 类型的准确率（正确预测数 / 总样本数）。
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # 中文：取序列最后一个位置（-1）的 logits 作为整条序列的分类依据——
            # 因为因果注意力下，最后一个 token 的隐藏状态已经"看过"了
            # 前面所有 token 的信息，可视为整句话的汇总表示。
            logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    """第 6 章：计算单个 batch 的分类交叉熵损失。

    参数：
        input_batch: 形状 (batch_size, seq_len) 的 token id。
        target_batch: 形状 (batch_size,) 的分类标签（0/1）。
        model: 分类模型。
        device: 计算设备。

    返回：
        标量损失张量（可直接调用 .backward()）。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 中文：同样只取最后一个 token 位置的 logits 用于分类任务的交叉熵计算，
    # 形状从 (batch_size, seq_len, num_classes) 变为 (batch_size, num_classes)。
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# Overall the same as `train_model_simple` in chapter 5
# 中文：整体结构与第 5 章的 train_model_simple 基本一致，
# 区别在于这里训练的是分类任务（交叉熵 + 准确率），而不是语言建模任务。
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    """第 6 章：分类微调的训练主循环。

    标准的"前向 -> 计算损失 -> 反向传播 -> 优化器更新参数"循环，
    并按固定步数间隔（eval_freq）在训练/验证集上评估损失，
    每个 epoch 结束后额外评估一次分类准确率。

    参数：
        model: 待微调的分类模型。
        train_loader / val_loader: 训练集 / 验证集 DataLoader。
        optimizer: 优化器（如 AdamW）。
        device: 计算设备。
        num_epochs: 训练总轮数。
        eval_freq: 每隔多少个训练 step 评估一次损失。
        eval_iter: 每次评估时抽样的 batch 数量。

    返回：
        (train_losses, val_losses, train_accs, val_accs, examples_seen)：
        分别是训练/验证损失历史列表、训练/验证准确率历史列表，
        以及训练过程中累计"见过"的样本总数。
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            # 中文：这里统计的是"样本数"而不是第 5 章语言建模里统计的"token 数"，
            # 因为分类任务关心的是处理过多少条短信样本。
            global_step += 1

            # Optional evaluation step
            # 中文：每 eval_freq 步做一次训练/验证损失评估，用于监控训练过程、
            # 判断是否过拟合（train_loss 持续下降但 val_loss 上升）。
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        # 中文：准确率计算比损失更耗时（需要 argmax + 逐样本比较），
        # 所以只在每个 epoch 结束后计算一次，而不是每个 step 都算。
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """第 6 章：绘制训练/验证指标（损失或准确率）随 epoch 变化的曲线图。

    图上有两条横轴：下方主轴是"epoch 数"，上方副轴是"已见样本数"，
    两者共享同一纵轴（指标数值），方便同时观察"训练进度"和"数据吞吐量"。

    参数：
        epochs_seen: 与 train_values/val_values 一一对应的 epoch 坐标序列。
        examples_seen: 与 train_values/val_values 一一对应的累计样本数序列。
        train_values / val_values: 要绘制的训练 / 验证指标数值列表
            （如损失值或准确率）。
        label: 指标名称，用于图例和纵轴标签（默认 "loss"）。

    返回：
        无（None）；函数会把图像保存为 "{label}-plot.pdf" 并调用 plt.show() 展示。
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for tokens seen
    # 中文：twiny() 创建一个共享纵轴、独立横轴的"孪生"坐标轴，
    # 用来在图的上方额外标出"已处理样本数"这一维度信息。
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    # 中文：alpha=0 画一条完全透明的曲线，只是为了让 ax2 的刻度范围
    # 与 examples_seen 对齐，实际可见的曲线还是 ax1 上的那两条。
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
