# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

# ============================================================
# 中文说明（模块级 docstring）
# ------------------------------------------------------------
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
# 一书第 3~4 章内容的汇总实现，属于 ch04/10_kv-sharing 目录下的一个对照基线
# (baseline) 脚本：它实现的是"标准的多头注意力"(Multi-Head Attention, MHA)，
# 也就是每个注意力头都拥有各自独立的 Key/Value 投影矩阵，不做任何 KV 头共享。
#
# 该文件在本章（10_kv-sharing，探讨 KV 缓存与 KV 头共享/分组查询注意力 GQA 等
# 优化技巧）中的角色是"基准对照组"：
#   - 用于和同目录下的 GQA（Grouped-Query Attention，分组查询注意力）等
#     变体脚本做速度、显存占用的对比；
#   - 演示了如何在一个朴素的 GPT 结构上加入 KV 缓存 (KV cache) 机制，
#     从而在自回归生成 (autoregressive generation) 时避免重复计算历史
#     token 的 Key/Value，提升推理速度。
#
# 核心内容包括：
#   1. MultiHeadAttention：标准多头自注意力 + 因果掩码 + KV 缓存实现；
#   2. LayerNorm / GELU / FeedForward：Transformer 块中的归一化与前馈网络；
#   3. TransformerBlock：将注意力子层与前馈子层用残差连接组合起来；
#   4. GPTModel：完整的 GPT 结构（词嵌入 + 位置嵌入 + N 个 Transformer 块 +
#      输出头），并支持带 KV 缓存的推理；
#   5. generate_text_simple_cached：使用（或不使用）KV 缓存进行贪心解码生成；
#   6. main：命令行入口，构建一个可配置规模的 GPT 模型并测试生成速度/显存。
# ============================================================

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """标准多头自注意力模块（Multi-Head Attention），支持可选的 KV 缓存。

    这是《从零构建大语言模型》第3章介绍的多头注意力实现，在本章中被作为
    "无 KV 头共享"的基线版本，用来和分组查询注意力（GQA）等共享 KV 的变体
    做性能对比。

    核心思路：
      - 用三个线性层分别把输入 x 投影成 Query、Key、Value；
      - 把 d_out 维度拆分成 num_heads 个 head_dim 维的子空间，让每个头独立
        计算注意力（多头机制可以让模型从不同的"表示子空间"关注信息）；
      - 计算 Query 与 Key 的缩放点积注意力分数，施加因果掩码（causal mask，
        保证第 i 个 token 只能看到 <= i 的 token，防止"看到未来"）；
      - 用 softmax 得到注意力权重，加权求和 Value 得到上下文向量；
      - 最后拼接所有头的输出并做一次线性投影。

    参数：
      d_in (int): 输入特征维度。
      d_out (int): 输出特征维度，同时也是 Q/K/V 投影后的总维度
                   （d_out = num_heads * head_dim）。
      dropout (float): 注意力权重上的 dropout 比例。
      num_heads (int): 注意力头的数量。
      qkv_bias (bool): Q/K/V 线性层是否使用偏置项，默认为 False。
    """
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头分到的维度大小，d_out 会被平均切分给 num_heads 个头

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 中文：注意，这里 Key 和 Value 的头数与 Query 完全相同（num_heads），
        # 也就是"标准 MHA"——每个 Query 头都有自己独享的 Key/Value 头，
        # 不像 GQA/MQA 那样多个 Query 头共享同一组 Key/Value 头。
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # KV cache-related code
        # 中文：KV 缓存相关代码。
        # 推理（自回归生成）时，历史 token 的 Key/Value 一旦算过就不会变化，
        # 若每生成一个新 token 都重新对整个序列计算 K/V，会产生大量重复计算。
        # KV 缓存的做法是：把已经算好的 Key/Value 保存在 buffer 里，
        # 后续只需要对新增的 token 计算 K/V，然后拼接（concat）到缓存后面即可。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        # 中文：persistent=False 表示这个 buffer 不会被存进 state_dict
        # （因为它只是运行时的临时状态，不是需要保存的模型参数）。
        self.ptr_current_pos = 0
        # 中文：记录当前已经处理到序列中的第几个位置，用于在使用缓存时
        # 正确计算因果掩码里 Query 的绝对位置。
        ####################################################

    def forward(self, x, use_cache=False):
        """前向传播：计算多头自注意力的输出。

        参数：
          x (Tensor): 形状 (b, num_tokens, d_in)，输入的 token 表征序列。
          use_cache (bool): 是否启用 KV 缓存。True 时会把本次算出的
                             Key/Value 追加进缓存，并让因果掩码基于
                             "全局位置"（而不是当前这一小段输入的相对位置）
                             来计算，从而支持"只喂入新 token"的增量式生成。

        返回：
          Tensor，形状 (b, num_tokens, d_out)，融合了历史上下文信息的
          新表征序列（用于残差相加或送入下一层）。
        """
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)
        # 中文：以上三行分别把输入 x 线性投影为本次新增 token 的
        # Key、Value 和 Query，形状均为 (b, num_tokens, d_out)。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆成 (num_heads, head_dim)，
        # 这样就相当于把一个大的线性投影"隐式地"切成了多个头的投影。
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache-related
        # 中文：KV 缓存的核心逻辑——把本次新算出的 K/V 与历史缓存拼接起来。
        if use_cache:
            if self.cache_k is None:
                # 中文：第一次调用（比如处理完整的 prompt），缓存为空，
                # 直接把这次算出的 K/V 作为缓存的初始值。
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                # 中文：非第一次调用（生成阶段每次只传入 1 个新 token），
                # 沿着 seq_len 维度（dim=1）把新 K/V 拼接到历史缓存后面。
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
            # 中文：用于本次注意力计算的 keys/values 是"完整历史 + 新 token"，
            # 而 queries 仍然只是本次新输入 token 对应的 Query。
        else:
            # 中文：不使用缓存（如训练阶段，或一次性把整段序列喂进去），
            # 直接用本次算出的 K/V，不做任何拼接。
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度换到前面，方便把每个头当作独立的 batch
        # 维度来做批量矩阵乘法（bmm），从而实现"多头并行计算"。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文：queries 形状 (b, num_heads, num_tokens_Q, head_dim)，
        # keys.transpose(2,3) 形状 (b, num_heads, head_dim, num_tokens_K)，
        # 相乘后 attn_scores 形状为 (b, num_heads, num_tokens_Q, num_tokens_K)，
        # 表示每个 Query token 对每个 Key token 的原始注意力打分。

        ####################################################
        # causal mask
        # 中文：因果掩码——保证每个位置只能"看到"它自己以及它之前的 token，
        # 不能看到未来的 token（这是自回归语言模型的核心约束）。
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            # 中文：使用缓存时，keys 包含了完整历史，而 queries 只是新增的
            # 一小段 token；因此 Query 的"绝对位置"要从 ptr_current_pos
            # 开始计数（而不是从 0 开始），否则掩码会算错。
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q
            # 中文：处理完这次输入后，把指针向前推进 num_tokens_Q，
            # 为下一次调用（下一个新 token）做准备。
        else:
            # 中文：不使用缓存时，Query 就是整个序列，位置从 0 开始编号即可。
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
            # 中文：非缓存模式下重置指针，避免和之前可能的缓存调用状态混淆。
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)
        # 中文：mask_bool[i, j] = True 表示"Query 位置 i 小于 Key 位置 j"，
        # 也就是 j 是 i 的"未来" token，这种位置需要被掩盖掉（不能被看到）。
        # 形状：(num_tokens_Q, num_tokens_K)，会广播到 attn_scores 的最后两维。

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 中文：把"未来"位置的注意力分数设为负无穷，这样经过 softmax 之后
        # 这些位置的权重会变成 0，等价于完全不关注未来的 token。

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 中文：除以 sqrt(head_dim) 做缩放（Scaled Dot-Product Attention 中的
        # "Scaled"），防止点积数值过大导致 softmax 梯度消失；
        # 然后在最后一维（Key 维度）上做 softmax，得到归一化的注意力权重。
        attn_weights = self.dropout(attn_weights)
        # 中文：对注意力权重做 dropout，是一种正则化手段，训练时随机丢弃
        # 一部分注意力连接，防止过拟合（推理/eval 模式下 dropout 不生效）。

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 中文：attn_weights 形状 (b, num_heads, num_tokens_Q, num_tokens_K)，
        # values 形状 (b, num_heads, num_tokens_K, head_dim)，
        # 相乘后得到 (b, num_heads, num_tokens_Q, head_dim)，
        # 也就是每个头对 Value 的加权求和结果；再转置回
        # (b, num_tokens_Q, num_heads, head_dim) 方便后续拼接。

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 中文：.contiguous() 是因为 transpose 之后张量在内存中不再连续，
        # view 操作要求内存连续；这里把多个头的输出重新拼接回 d_out 维度。
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：最后再做一次线性投影，让不同头的信息可以相互混合
        # （这一层在有些实现里是可选的，但标准 Transformer 中通常保留）。

        return context_vec

    def reset_cache(self):
        """清空 KV 缓存与位置指针。

        在每次开始一段新的生成任务（新的 prompt）之前，需要调用此方法，
        避免上一次生成残留的 Key/Value 缓存污染本次的注意力计算。
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）模块。

    对每个 token 的特征向量（最后一维）做归一化：减去均值、除以标准差，
    再用可学习的缩放参数 scale 和平移参数 shift 做仿射变换。
    作用是稳定深层网络的训练，缓解梯度爆炸/消失问题。

    参数：
      emb_dim (int): 需要归一化的特征维度大小（即嵌入维度）。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # 中文：eps 是一个很小的常数，加在方差上防止除以 0。
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        # 中文：scale、shift 是可学习参数，分别初始化为全 1 和全 0，
        # 使得初始状态下 LayerNorm 近似恒等变换，训练过程中再自适应调整。

    def forward(self, x):
        """对输入 x 的最后一维做归一化。

        参数：
          x (Tensor): 形状 (..., emb_dim)，通常是 (batch, seq_len, emb_dim)。

        返回：
          Tensor，形状与输入相同，每个 token 的特征已被归一化并做仿射变换。
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 中文：unbiased=False 表示用有偏估计（除以 N 而不是 N-1）计算方差，
        # 这是深度学习框架中 LayerNorm 的标准做法。
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数（Gaussian Error Linear Unit）的近似实现。

    这里使用的是 GPT-2 论文中采用的 tanh 近似公式，而不是精确的
    高斯误差函数形式，计算更高效，效果也非常接近。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """对输入逐元素应用 GELU 激活。

        参数：
          x (Tensor): 任意形状的输入张量。

        返回：
          Tensor，形状与输入相同，经过 GELU 非线性变换后的结果。
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        # 中文：GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 x^3) ))
        # 相比 ReLU，GELU 在 0 附近更平滑，是 GPT 系列模型常用的激活函数。


class FeedForward(nn.Module):
    """Transformer 块中的前馈网络（Feed-Forward Network, FFN）。

    结构为：线性升维（emb_dim -> 4*emb_dim） -> GELU 激活 -> 线性降维
    （4*emb_dim -> emb_dim）。先升维再降维的"沙漏形"结构可以让模型在
    更高维的隐空间里做非线性变换，增强表达能力。

    参数：
      cfg (dict): 配置字典，需要包含键 "emb_dim"。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """前向传播。

        参数：
          x (Tensor): 形状 (batch, seq_len, emb_dim)。

        返回：
          Tensor，形状 (batch, seq_len, emb_dim)，与输入形状相同。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """一个完整的 Transformer 块：多头注意力子层 + 前馈网络子层，
    每个子层前置 LayerNorm（Pre-LN 结构），并各自带残差连接。

    参数：
      cfg (dict): 配置字典，需要包含 "emb_dim"、"n_heads"、"drop_rate"、
                  "qkv_bias" 等键（详见 GPTModel 中的用法）。
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
        """前向传播：依次经过"注意力子层 + 残差"和"前馈子层 + 残差"。

        参数：
          x (Tensor): 形状 (batch, seq_len, emb_dim)。
          use_cache (bool): 是否启用 KV 缓存，透传给内部的
                             MultiHeadAttention。

        返回：
          Tensor，形状 (batch, seq_len, emb_dim)，与输入形状相同。
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        # 中文：Pre-LN 结构——先做归一化，再送入注意力层，
        # 相比 Post-LN（先注意力再归一化）训练更稳定，是现代 LLM 的常见做法。

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        # 中文：把 use_cache 参数透传给底层的多头注意力模块，
        # 由它决定是否读写 KV 缓存。
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差连接（Residual Connection），把注意力子层的输出
        # 加回原始输入，有助于缓解深层网络的梯度消失问题。

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：前馈子层同样采用"Pre-LN + 残差"的结构。

        return x


class GPTModel(nn.Module):
    """完整的类 GPT-2 语言模型。

    结构为：
      词嵌入 (token embedding) + 位置嵌入 (position embedding)
      -> Dropout
      -> N 个 TransformerBlock 堆叠
      -> 最终 LayerNorm
      -> 线性输出头（映射到词表大小，得到每个位置的下一词 logits）

    参数：
      cfg (dict): 配置字典，需要包含：
        - "vocab_size": 词表大小
        - "context_length": 支持的最大上下文长度（决定位置嵌入表大小）
        - "emb_dim": 嵌入维度
        - "n_heads": 注意力头数
        - "n_layers": Transformer 块的层数
        - "drop_rate": dropout 比例
        - "qkv_bias": Q/K/V 线性层是否使用偏置
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache-related
        # 中文：这里改用 nn.ModuleList 而不是 nn.Sequential，
        # 是因为 nn.Sequential 的 forward 只能按固定方式依次调用各层，
        # 无法给每一层传递额外的 use_cache 参数；而 ModuleList 只是
        # 一个模块容器，需要我们自己手写循环来控制每层的调用方式。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0
        # 中文：记录当前生成到的绝对位置，用于在使用 KV 缓存时
        # 给新 token 分配正确的位置嵌入（position embedding）。
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        """前向传播。

        参数：
          in_idx (Tensor): 形状 (batch_size, seq_len)，输入的 token id 序列。
          use_cache (bool): 是否启用 KV 缓存。
                             - 训练/一次性推理时通常为 False：一次性把整段
                               输入喂进去。
                             - 增量式生成时为 True：可以只喂入"新增的
                               token"（比如每次只有 1 个 token），模型会
                               利用缓存的历史 K/V 补全上下文。

        返回：
          logits (Tensor): 形状 (batch_size, seq_len, vocab_size)，
                            每个位置上对下一个 token 的未归一化预测分数。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文：tok_embeds 形状 (batch_size, seq_len, emb_dim)。

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        # 中文：位置嵌入需要根据"绝对位置"而不是"本次输入内的相对位置"来取，
        # 否则使用缓存增量生成时，每次都会错误地从位置 0 开始编号。
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
            # 中文：处理完本次输入后，把全局位置指针向前推进 seq_len，
            # 以便下一次调用（比如生成下一个 token）时位置编号正确衔接。
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        # 中文：unsqueeze(0) 增加一个 batch 维度，形状变为
        # (1, seq_len, emb_dim)，便于和 (batch_size, seq_len, emb_dim)
        # 的 tok_embeds 做广播相加。
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：词嵌入 + 位置嵌入，得到融合了"词义"与"位置信息"的初始表征。
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        # 中文：手动遍历每个 TransformerBlock，并把 use_cache 逐层透传下去，
        # 这样每一层内部的注意力模块都能各自维护自己的 KV 缓存。
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        # 中文：logits 形状 (batch_size, seq_len, vocab_size)，
        # 即每个位置对词表中每个 token 的打分（尚未做 softmax）。
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        """重置整个模型的 KV 缓存与位置指针。

        在开始生成一段新文本（新的 prompt）之前必须调用，逐层清空每个
        TransformerBlock 内部注意力模块的缓存，并把全局位置计数器归零，
        避免不同生成任务之间的缓存状态互相干扰。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用贪心解码（greedy decoding）自回归生成文本，可选择是否使用 KV 缓存。

    参数：
      model (GPTModel): 已训练好的 GPT 模型实例。
      idx (Tensor): 形状 (batch_size, seq_len)，初始的 prompt token id 序列。
      max_new_tokens (int): 需要生成的新 token 数量。
      context_size (int, optional): 模型支持的最大上下文长度；若为 None，
                                     则使用模型位置嵌入表的大小
                                     （model.pos_emb.num_embeddings）。
      use_cache (bool): 是否启用 KV 缓存来加速生成。
                         - True：只需在第一次前向传播时喂入完整 prompt
                           （初始化缓存），之后每一步只需要喂入"刚生成的
                           那 1 个新 token"，极大减少重复计算；
                         - False：朴素做法，每生成一个新 token 都要把
                           "当前完整序列"重新喂给模型做一次完整前向传播，
                           计算量随生成长度增长而显著增加（重复计算历史
                           token 的注意力）。

    返回：
      idx (Tensor): 形状 (batch_size, seq_len + max_new_tokens)，
                    包含原始 prompt 和新生成 token 的完整序列。
    """
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        # 中文：生成阶段不需要计算梯度，用 no_grad 节省显存并加速。
        if use_cache:
            # Init cache with full prompt
            # 中文：先重置缓存（防止残留上一次生成的状态），
            # 再用完整的 prompt 做一次前向传播，把 prompt 对应的
            # Key/Value 全部写入缓存。
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                # 中文：贪心采样——直接取概率（logits）最大的那个 token，
                # 不做随机采样，因此每次运行结果是确定性的。
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                # 中文：得益于 KV 缓存，这里只需要把"刚生成的这 1 个 token"
                # 喂给模型，模型内部会自动结合缓存的历史 K/V 计算注意力，
                # 而不需要重新处理整个序列，这正是 KV 缓存加速的关键所在。
                logits = model(next_idx, use_cache=True)
        else:
            # 中文：不使用缓存的朴素版本——每一步都把"目前为止的完整序列"
            # （截断到最大上下文长度）重新喂给模型，重复计算了历史 token
            # 的 Key/Value，计算量更大，仅作为速度对比的基线。
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    """命令行入口：解析参数、构建 GPT 模型，并测试生成速度与显存占用。

    支持通过命令行参数自定义模型规模（嵌入维度、头数、层数）以及
    生成的新 token 数量，运行结束后打印生成的文本、耗时以及
    tokens/sec 的吞吐量（若在 GPU 上运行还会打印显存峰值占用）。
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run GPT with standard multi-head attention."
    )
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    # 中文：用 GPT-2 的 BPE 分词器把起始文本编码成 token id 列表。

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        # 中文：上下文长度设置为"待生成 token 数 + prompt 长度"，
        # 刚好覆盖本次生成过程中会用到的所有位置编号。
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
    }
    torch.manual_seed(123)
    # 中文：固定随机种子，保证模型参数初始化可复现。
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    # 中文：把模型搬到 GPU（若可用）并转换为 bfloat16 精度，
    # 可以减少显存占用、加快推理速度，是常见的推理加速手段。
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    # 中文：unsqueeze(0) 增加 batch 维度，形状变为 (1, seq_len)。
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 中文：GPU 操作是异步的，计时前需要同步，确保之前的操作都已完成，
        # 否则计时结果不准确。
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 中文：生成结束后同样需要同步，才能拿到准确的结束时间。
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
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")
        # 中文：打印本次运行过程中 GPU 显存的峰值占用，
        # 便于和其他 KV 共享变体（如 GQA）做显存开销对比。


if __name__ == "__main__":
    main()
