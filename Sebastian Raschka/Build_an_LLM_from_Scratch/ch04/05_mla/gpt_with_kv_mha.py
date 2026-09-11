# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

"""
【模块中文说明】
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 3-4 章代码的汇总版本，并在标准多头注意力 (Multi-Head Attention, MHA) 的基础上
新增了 **KV 缓存 (KV cache)** 支持，用于在自回归文本生成 (autoregressive generation)
时避免对已经计算过的 token 重复计算 Key/Value 投影，从而大幅提升逐 token 生成的推理速度。

文件在整本书中的角色：
- 第 3 章内容：`MultiHeadAttention`（多头自注意力机制，含因果掩码 causal mask）。
- 第 4 章内容：`LayerNorm`、`GELU`、`FeedForward`、`TransformerBlock`、`GPTModel`
  —— 构成一个完整的、类似 GPT-2 结构的仅解码器 (decoder-only) Transformer 语言模型。
- 本文件相对书中基础版本的增量：为 `MultiHeadAttention` 和 `GPTModel` 增加了
  KV 缓存机制（见代码中标注为 "KV cache-related" 的部分），并提供了
  `generate_text_simple_cached` 函数演示如何利用 KV 缓存做增量式（每次只喂一个新
  token）的贪婪解码 (greedy decoding)，同时保留了不使用缓存的朴素版本作对比。
- 文件名中的 "mha" 指的是使用标准的 Multi-Head Attention（区别于同目录下可能存在的
  MLA / GQA 等变体实现），可作为理解 KV 缓存工作原理的基准实现。

可直接作为脚本运行：`python gpt_with_kv_mha.py`，会构造一个小型 GPT 模型并测试
带 KV 缓存的文本生成速度。
"""

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """标准多头自注意力 (Multi-Head Attention) 模块，并内置了 KV 缓存 (KV cache) 支持。

    核心思想：把输入通过三个线性层分别投影为 Query、Key、Value，然后拆分成
    `num_heads` 个「头」并行计算缩放点积注意力 (scaled dot-product attention)，
    每个头只关注 `head_dim = d_out / num_heads` 维的子空间，最后再把所有头的结果
    拼接起来经过一次输出投影。

    相比教材基础版本，这里额外维护了 `cache_k` / `cache_v` 两个缓冲区：在自回归生成
    时，新 token 的 Key/Value 会被追加(concat)到缓存里，避免重新计算历史 token 的
    Key/Value，从而将每步生成的计算量从 O(seq_len) 降到 O(1)（相对新 token 而言）。

    参数:
        d_in (int): 输入特征维度（即输入张量最后一维大小）。
        d_out (int): 输出特征维度，同时也是 Q/K/V 投影后的总维度
            （之后会被均分给 num_heads 个头）。
        dropout (float): 注意力权重上使用的 dropout 概率。
        num_heads (int): 注意力头的数量，要求 d_out 能被其整除。
        qkv_bias (bool): Q/K/V 三个线性层是否使用偏置项 (bias)。
    """
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头分到的维度 = 总输出维度 / 头数，保证多头拼接后维度仍等于 d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # KV cache-related code
        # 中文：KV 缓存相关代码。
        # register_buffer 注册的是「非可训练参数」的张量，persistent=False 表示
        # 它不会被保存进 state_dict（因为缓存是运行时状态，不是模型权重）。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0  # 中文：记录当前已经处理到的绝对位置（token 下标），用于生成因果掩码
        ####################################################

    def forward(self, x, use_cache=False):
        """前向传播：计算带因果掩码的多头自注意力。

        参数:
            x (Tensor): 输入张量，形状为 (b, num_tokens, d_in)。
                - 当 use_cache=True 且处于增量解码阶段时，num_tokens 通常为 1
                  （只输入最新生成的那个 token）。
            use_cache (bool): 是否启用 KV 缓存。True 时会把本次计算出的 Key/Value
                追加到历史缓存中，并基于缓存里的全部 Key/Value 计算注意力。

        返回:
            Tensor: 注意力输出，形状为 (b, num_tokens, d_out)，
                与输入的 token 数一致（即每个 query token 对应一个输出向量）。
        """
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)
        # 中文：queries 的形状同样是 (b, num_tokens, d_out)；
        # 注意这里的 queries 只对应「当前输入」的 token，不会像 keys/values 那样被缓存，
        # 因为每个 query 只需要跟历史 + 当前的所有 key/value 做一次注意力计算即可。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆分为 (num_heads, head_dim)，实现「多头」的效果，
        # 这一步并不改变数值，只是重新排列张量的形状（view 不复制数据）。
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache-related
        # 中文：KV 缓存核心逻辑。
        if use_cache:
            if self.cache_k is None:
                # 中文：第一次调用（比如喂入完整 prompt），直接把新算出的 K/V 作为缓存初始值
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                # 中文：后续调用（比如每次只喂 1 个新 token），把新 K/V 沿着 seq_len 维度（dim=1）
                # 拼接到历史缓存后面，使缓存长度随生成过程不断增长
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
            # 中文：最终用于注意力计算的 keys/values 是「历史全部 + 当前新增」token 的集合，
            # 形状为 (b, total_seen_tokens, num_heads, head_dim)
        else:
            # 中文：不使用缓存时，keys/values 就只是当前这次前向传播里算出的值（标准做法）
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 提到 batch 维之后，方便对每个头独立做矩阵乘法（批量矩阵乘）
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文：queries (b, h, num_tokens_Q, head_dim) 与 keys^T (b, h, head_dim, num_tokens_K)
        # 相乘，得到注意力分数矩阵，形状为 (b, num_heads, num_tokens_Q, num_tokens_K)。
        # 当使用 KV 缓存时，num_tokens_Q 通常很小（甚至=1），而 num_tokens_K 是累计的历史长度。

        ####################################################
        # causal mask
        # 中文：构造因果掩码 (causal mask)，确保每个 query 只能看到「自己以及更早」位置的 key，
        # 不能看到未来的 token —— 这是自回归语言模型的核心约束。
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            # 中文：使用缓存时，query 对应的「绝对位置」不是从 0 开始的，而是从
            # ptr_current_pos（上一次处理到的位置）继续往后编号，
            # 例如 prompt 长度为 5，那么下一个新 token 的绝对位置就是 5。
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q  # 中文：更新指针，为下一次调用做准备
        else:
            # 中文：不使用缓存时，每次都是从头开始的完整序列，位置编号从 0 开始
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        # 中文：mask_bool[i, j] = True 表示「query 的绝对位置 i 小于 key 的绝对位置 j」，
        # 即 key 在未来，需要被屏蔽掉。这种基于绝对位置比较的写法，
        # 天然兼容 KV 缓存场景下 query 位置和 key 位置范围不一致的情况。
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        # Use the mask to fill attention scores
        # 中文：把被掩盖位置的注意力分数设为负无穷，softmax 后这些位置的权重会趋近于 0
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：缩放点积注意力的「缩放」部分——除以 sqrt(head_dim)，防止点积数值过大导致
        # softmax 梯度消失（这是 Attention is All You Need 论文中的标准做法）
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：注意力权重 (b, h, num_tokens_Q, num_tokens_K) 与 values (b, h, num_tokens_K, head_dim)
        # 相乘，得到每个 query 位置的加权上下文向量；随后转置回 (b, num_tokens_Q, h, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多头拼接回单一向量，形状变回 (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：输出投影层，让多头拼接后的结果再融合一次信息（可学习的线性变换）

        return context_vec

    def reset_cache(self):
        """清空 KV 缓存，并将位置指针归零。

        通常在开始处理一个全新的序列（比如新的一次生成任务）之前调用，
        避免历史缓存污染新序列的注意力计算。
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化 (Layer Normalization)。

    对每个样本、每个位置的特征向量（最后一维）做零均值、单位方差的归一化，
    再通过可学习的缩放 (scale) 和平移 (shift) 参数恢复模型需要的表达能力。
    与 BatchNorm 不同，LayerNorm 的统计量是在特征维上计算的，不依赖 batch 内其他样本，
    因此在处理变长序列 / 小 batch 的 Transformer 中更常用。

    参数:
        emb_dim (int): 特征维度（embedding 维度），归一化在这一维上进行。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 中文：可学习的缩放参数 gamma，初始为 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 中文：可学习的平移参数 beta，初始为 0

    def forward(self, x):
        """
        参数:
            x (Tensor): 形状为 (..., emb_dim)，通常是 (batch, seq_len, emb_dim)。
        返回:
            Tensor: 与输入同形状，已完成归一化 + 仿射变换。
        """
        mean = x.mean(dim=-1, keepdim=True)  # 中文：在最后一维（特征维）求均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 中文：有偏方差估计（除以 N 而非 N-1）
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 中文：标准化到均值 0、方差 1
        return self.scale * norm_x + self.shift  # 中文：仿射变换，让网络自己学习最合适的尺度和偏移


class GELU(nn.Module):
    """GELU (Gaussian Error Linear Unit) 激活函数的 tanh 近似实现。

    GPT-2 等模型使用 GELU 而非 ReLU 作为前馈网络的激活函数，
    因为它在 0 附近更平滑，实验上通常能带来更好的训练效果。
    这里实现的是论文中常见的 tanh 近似公式，而非精确的误差函数 (erf) 版本。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        参数:
            x (Tensor): 任意形状的输入张量。
        返回:
            Tensor: 与输入同形状，逐元素应用 GELU 激活后的结果。
        """
        # 中文：0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        # 这是 GELU 的近似公式，比精确公式（依赖 erf）计算更快
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer 中的逐位置前馈网络 (Position-wise Feed-Forward Network)。

    结构为：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)。
    先升维再降维的「瓶颈」结构（这里是 4 倍扩展）是 GPT 系列模型的标准设计，
    用于在每个 token 位置上独立地做非线性特征变换（不涉及 token 之间的交互，
    token 间交互完全由注意力层负责）。

    参数:
        cfg (dict): 配置字典，需要包含键 "emb_dim"（模型的 embedding 维度）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 中文：升维到 4 倍
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 中文：再降维回原始维度
        )

    def forward(self, x):
        """
        参数:
            x (Tensor): 形状为 (batch, seq_len, emb_dim)。
        返回:
            Tensor: 形状同输入 (batch, seq_len, emb_dim)。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """一个标准的 (Pre-LayerNorm 结构的) Transformer 解码器块。

    结构：
        x -> LayerNorm -> 多头注意力 -> Dropout -> 残差相加
          -> LayerNorm -> 前馈网络   -> Dropout -> 残差相加

    这种「先归一化再进入子层，子层输出与子层输入的残差相加」的写法称为
    Pre-LN（Pre-LayerNorm），相比 Post-LN 在深层网络中训练更稳定。

    参数:
        cfg (dict): 配置字典，需要包含 "emb_dim"、"n_heads"、"drop_rate"、"qkv_bias" 等键。
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        """
        参数:
            x (Tensor): 形状为 (batch, num_tokens, emb_dim) 的输入张量。
            use_cache (bool): 是否启用 KV 缓存（会透传给内部的注意力层）。
        返回:
            Tensor: 形状与输入相同 (batch, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        # 中文：注意力子层的残差连接——先保存原始输入，供后面相加使用
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        # 中文：KV 缓存相关——把 use_cache 参数透传给注意力层，
        # 由 MultiHeadAttention 内部决定是否读写缓存
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差相加，缓解深层网络的梯度消失问题，也让信息可以「跳过」子层直接传递

        # Shortcut connection for feed-forward block
        # 中文：前馈子层的残差连接，逻辑与上面对称
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的类 GPT-2 仅解码器 (decoder-only) Transformer 语言模型，内置 KV 缓存支持。

    结构：Token Embedding + 位置 Embedding -> Dropout -> N 层 TransformerBlock
          -> 最终 LayerNorm -> 线性输出头（映射到词表大小，得到 logits）。

    参数:
        cfg (dict): 配置字典，典型键包括：
            - "vocab_size": 词表大小
            - "emb_dim": embedding / 隐藏层维度
            - "context_length": 支持的最大上下文长度（决定位置 embedding 表大小）
            - "n_heads": 注意力头数
            - "n_layers": Transformer 块的层数
            - "drop_rate": dropout 概率
            - "qkv_bias": Q/K/V 投影是否使用 bias
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 中文：词元 (token) 嵌入表
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 中文：可学习的绝对位置嵌入表
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
         #  KV cache-related
        # 中文：KV 缓存相关——这里改用 nn.ModuleList 而非 nn.Sequential，
        # 是因为需要在遍历时手动把 use_cache 参数传给每一层（Sequential 不支持多参数转发）
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0  # 中文：记录全局已生成/已处理到的 token 绝对位置，用于取正确的位置嵌入
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 中文：输出头把隐藏状态映射到词表维度，得到每个位置对下一个 token 的预测分数 (logits)

    def forward(self, in_idx, use_cache=False):
        """
        参数:
            in_idx (LongTensor): 输入 token id 序列，形状为 (batch_size, seq_len)。
                - use_cache=True 且处于增量解码阶段时，seq_len 通常为 1。
            use_cache (bool): 是否启用 KV 缓存，会透传给每一个 TransformerBlock。

        返回:
            Tensor: logits，形状为 (batch_size, seq_len, vocab_size)，
                表示每个输入位置对「下一个 token」在整个词表上的预测分数。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 中文：形状 (batch_size, seq_len, emb_dim)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        # 中文：KV 缓存相关——位置编码不能简单地从 0 开始数，
        # 因为增量解码时每次只输入 1 个新 token，其真实位置是「历史长度 + 0」，
        # 所以需要用 self.current_pos 记录并累加全局位置。
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len  # 中文：累加已处理的 token 数，供下一次调用使用
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)  # 中文：增加 batch 维，形状变为 (1, seq_len, emb_dim)，可广播到 batch_size
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：词嵌入与位置嵌入逐元素相加，融合「词是什么」和「词在哪个位置」两类信息
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        # 中文：KV 缓存相关——手动逐层遍历 ModuleList，把 use_cache 显式传给每个 TransformerBlock，
        # 使得每一层内部的注意力模块都能各自维护自己的 KV 缓存
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)  # 中文：形状 (batch_size, seq_len, vocab_size)
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        """重置模型中所有层的 KV 缓存以及全局位置指针。

        中文：在开始一次全新的生成任务（新的 prompt）之前必须调用，
        否则上一次生成残留的缓存会与新序列的 token 混在一起，导致注意力计算错误。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用贪婪解码 (greedy decoding) 自回归生成文本，支持 KV 缓存加速。

    参数:
        model (GPTModel): 已实例化的 GPT 模型。
        idx (LongTensor): 起始 token id 序列（prompt），形状为 (batch_size, seq_len)。
        max_new_tokens (int): 要新生成的 token 数量。
        context_size (int, optional): 模型支持的最大上下文长度，用于在不使用缓存时
            对输入做截断（滑动窗口）。默认取 model.pos_emb.num_embeddings。
        use_cache (bool): 是否启用 KV 缓存。
            - True：只需在第一次前向传播时喂入完整 prompt 来「预热」缓存，
              之后每一步只需喂入上一步新生成的单个 token，计算量大幅降低。
            - False：朴素做法，每一步都要把当前完整序列（截断到 context_size）
              重新喂给模型，计算量随生成长度线性增长，速度更慢。

    返回:
        LongTensor: 形状为 (batch_size, seq_len + max_new_tokens)，
            即原始 prompt 拼接上新生成的 token id 序列。
    """
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():  # 中文：推理阶段不需要梯度，节省显存并加速
        if use_cache:
            # Init cache with full prompt
            # 中文：先清空缓存，再用完整 prompt 做一次前向传播，把 prompt 中所有 token 的
            # Key/Value 都写入缓存（相当于「预热」），同时得到最后一个位置的 logits 用于预测下一个词
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                # 中文：贪婪采样——直接取概率最高（即 logits 最大）的 token，不做随机采样
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                # 中文：把新 token 拼接到完整序列末尾，用于最终返回结果
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                # 中文：关键优化点——只把新生成的这 1 个 token 喂给模型，
                # 而不是把整个序列重新算一遍；历史信息已经在 KV 缓存里了
                logits = model(next_idx, use_cache=True)
        else:
            # 中文：不使用缓存的朴素版本，作为速度对比的基准。
            # 每一步都要把（截断后的）完整序列重新前向传播一次，
            # 越往后生成序列越长，单步计算量也越大。
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    """脚本入口：构建一个小型 GPT 模型，用 KV 缓存做贪婪解码生成文本，并统计耗时/显存。

    中文说明：
    - 通过命令行参数可以调整模型规模（emb_dim / n_heads / n_layers）和生成长度。
    - 使用 tiktoken 的 GPT-2 分词器对起始文本编码。
    - 模型以 bfloat16 精度运行在可用的设备（CUDA 优先，否则 CPU）上。
    - 最终打印生成结果、耗时、tokens/sec 吞吐量，以及（若在 GPU 上）峰值显存占用。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run GPT with standard multi-head attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")  # 中文：使用 GPT-2 的 BPE 分词器
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        # 中文：上下文长度设为「prompt 长度 + 待生成长度」，刚好覆盖整个生成过程中会用到的所有位置
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
    }
    torch.manual_seed(123)  # 中文：固定随机种子，保证模型初始化权重可复现
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)  # 中文：使用 bfloat16 半精度以节省显存、加速计算
    model.eval()  # disable dropout
    # 中文：注意，这里的模型权重是随机初始化的（未经训练），生成的文本内容不具备语义意义，
    # 本脚本主要目的是演示并测量 KV 缓存对推理速度/显存的影响

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 中文：GPU 异步执行，计时前需同步以获得准确耗时
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")  # 中文：打印 GPU 峰值显存占用，便于比较不同配置/是否使用缓存的显存开销


if __name__ == "__main__":
    main()
