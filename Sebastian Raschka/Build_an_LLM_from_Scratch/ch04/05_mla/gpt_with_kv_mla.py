# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4, adapted to use Multi-Head Latent Attention (MLA).
# This file can be run as a standalone script.

# ============================================================
# 模块级中文说明（补充，非原作者内容）
# ------------------------------------------------------------
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
# 第 3-4 章代码的一个"变体"实现：把标准的多头注意力(Multi-Head Attention)
# 替换成了 Multi-Head Latent Attention（多头潜在注意力，简称 MLA，
# 由 DeepSeek 系列模型提出/使用），并同时实现了 KV 缓存(KV Cache)机制，
# 用于加速自回归文本生成。
#
# 核心知识点（本文件教学重点）：
#   1. MLA 的思路：不像标准 MHA 那样为每个头单独存储完整的 K/V，而是先把
#      输入压缩("下投影", down-projection)到一个低维的"潜在向量"
#      (latent vector, 记作 C)，推理时再把这个低维潜在向量"上投影"
#      (up-projection) 还原成各个头的 K 和 V。这样 KV 缓存中只需要保存
#      低维的潜在向量，而不是完整的多头 K/V，从而大幅降低显存占用。
#   2. KV 缓存：自回归生成时，每次只需要计算"新 token"的 Q/K/V，
#      历史 token 的信息通过缓存复用，避免重复计算，从而加速推理。
#      本文件里缓存的不是传统的 K、V 张量，而是 MLA 的"潜在向量" C，
#      这正是 MLA 相比标准 KV 缓存能节省显存的关键。
#   3. Transformer 的标准组件：LayerNorm、GELU 激活、FeedForward
#      前馈网络、残差连接(shortcut connection)、因果掩码(causal mask)等。
#   4. 一个可独立运行的贪心解码(greedy decoding)生成脚本，支持对比
#      "使用 KV 缓存"与"不使用 KV 缓存"两种推理方式的速度差异。
# ============================================================

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Multi-Head Latent Attention
#####################################
# The MLA code below is inspired by
# https://huggingface.co/bird-of-paradise/deepseek-mla
# 中文：下面的 MLA 实现参考了上述链接中的 DeepSeek MLA 实现思路。


class MultiHeadLatentAttention(nn.Module):
    """多头潜在注意力(Multi-Head Latent Attention, MLA)模块。

    与标准多头自注意力(Multi-Head Attention)的区别在于：
    - 标准 MHA：Q、K、V 都是从输入 x 直接线性投影得到，且 K、V 的维度
      与头数、头维度成正比，KV 缓存需要存储完整的多头 K、V。
    - MLA：Q 仍然直接投影；但 K、V 不再直接投影，而是先把输入 x
      通过 W_DKV "下投影"压缩成一个低维的潜在向量 C(latent，维度为
      latent_dim，通常远小于 d_out)，推理/生成时缓存的是这个低维的 C，
      需要参与注意力计算时，再用 W_UK、W_UV 把 C "上投影"还原成
      各头的 K、V。这样 KV 缓存所占的显存与 latent_dim 成正比，而不是
      与 num_heads * head_dim 成正比，从而显著节省显存，这是 DeepSeek
      系列模型采用 MLA 的核心动机。

    参数：
        d_in (int): 输入特征维度(通常等于 emb_dim)。
        d_out (int): 输出特征维度，也是所有头拼接后的总维度，
            必须能被 num_heads 整除。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool): 线性投影层是否使用偏置项。
        latent_dim (int, optional): 潜在向量 C 的维度。若不指定，
            默认取 max(16, d_out // 8)，即用一个较小的维度做压缩。
    """
    def __init__(self, d_in, d_out, dropout, num_heads,
                 qkv_bias=False, latent_dim=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # 若未显式指定 latent_dim，则用一个经验公式给出默认压缩维度：
        # 至少 16 维，或者是 d_out 的 1/8，取较大者，用于在压缩率和
        # 表达能力之间做折中。
        self.latent_dim = latent_dim if latent_dim is not None else max(16, d_out // 8)

        # Projections
        # 中文：以下是 MLA 用到的四个线性投影层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)              # per-head Q
        # W_query: 直接把输入投影成"每个头拼接后"的 Query，形状 (b, T, d_out)，
        # 这一点和标准 MHA 完全一样，Q 并不经过潜在空间压缩。
        self.W_DKV = nn.Linear(d_in, self.latent_dim, bias=qkv_bias)    # down to latent C
        # W_DKV (Down-projection to KV latent)：把输入 x 从 d_in 维
        # "下投影"压缩到 latent_dim 维，得到潜在向量 C，这是 MLA 的核心，
        # 也是需要被缓存的部分（比缓存完整 K、V 小得多）。
        self.W_UK = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)   # latent -> per-head K
        # W_UK (Up-projection to K)：推理时把潜在向量 C 从 latent_dim
        # "上投影"还原成所有头的 Key，形状变为 (b, T, d_out)。
        self.W_UV = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)   # latent -> per-head V
        # W_UV (Up-projection to V)：同理，把潜在向量 C 上投影还原成
        # 所有头的 Value。

        self.out_proj = nn.Linear(d_out, d_out)
        # 输出投影：多头拼接后的上下文向量再做一次线性变换，融合各头信息。
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # Latent-KV cache
        # 中文：这里注册一个"潜在 KV 缓存"缓冲区，用于存放历史 token
        # 的潜在向量 C。persistent=False 表示它不会被保存进
        # state_dict（因为它只是运行时的临时状态，不是模型参数）。
        self.register_buffer("cache_c_kv", None, persistent=False)
        self.ptr_current_pos = 0
        # ptr_current_pos：记录当前已经处理到序列的第几个位置，
        # 用于在使用缓存时正确地为新 token 计算绝对位置(用于构造因果掩码)。
        ####################################################

    def reset_cache(self):
        """清空该注意力层的潜在 KV 缓存，并把位置指针归零。

        通常在开始一次新的生成(generate)之前调用，避免把上一次生成
        残留的缓存错误地带入新的一轮推理。
        """
        self.cache_c_kv = None
        self.ptr_current_pos = 0

    @staticmethod
    def _reshape_to_heads(x, num_heads, head_dim):
        """把 (batch, seq_len, d_out) 的张量拆分成多头形式。

        参数：
            x: 形状 (b, T, d_out) 的张量，d_out = num_heads * head_dim。
            num_heads: 头数。
            head_dim: 每个头的维度。

        返回：
            形状 (b, num_heads, T, head_dim) 的张量，方便后续按头
            做批量矩阵乘法(注意力计算)。
        """
        # (b, T, d_out) -> (b, num_heads, T, head_dim)
        bsz, num_tokens, _ = x.shape
        # view 先把最后一维 d_out 拆成 (num_heads, head_dim)，
        # 再用 transpose 把 "头" 这一维换到序列长度之前，
        # 得到 (b, num_heads, T, head_dim)；.contiguous() 保证
        # 之后的 view/matmul 等操作在内存布局上是安全的。
        return x.view(bsz, num_tokens, num_heads, head_dim).transpose(1, 2).contiguous()

    def forward(self, x, use_cache=False):
        """MLA 前向传播。

        参数：
            x: 输入张量，形状 (b, T, d_in)；当 use_cache=True 且处于
               增量解码阶段时，T 通常等于 1(只输入新生成的那个 token)。
            use_cache (bool): 是否启用 KV 缓存(潜在向量缓存)。
               - True: 增量式解码模式，会把新 token 的潜在向量拼接进
                 已缓存的潜在向量序列中，并更新位置指针。
               - False: 一次性(非缓存)前向，通常用于训练或不使用缓存
                 的完整前向计算。

        返回：
            context_vec: 形状 (b, T, d_out) 的上下文向量，即该层
            注意力机制的输出，会被送入后续的残差连接和前馈网络。
        """
        b, num_tokens, _ = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        # 1) Project to queries (per-token, per-head) and new latent chunk
        # 中文：第一步——分别计算当前输入 x 对应的 Query，以及
        # 需要下投影得到的"新"潜在向量片段(latent_new)。
        queries_all = self.W_query(x)  # (b, T, d_out)
        latent_new = self.W_DKV(x)  # (b, T, latent_dim)

        # 2) Update latent cache and choose latent sequence to up-project
        # 中文：第二步——维护潜在向量缓存。如果启用缓存，就把这次新算出来的
        # latent_new 拼接到历史缓存 cache_c_kv 后面，得到覆盖"全部历史 + 当前"
        # token 的潜在序列 latent_total，并更新缓存；否则(不使用缓存)，
        # latent_total 就只是当前这次输入对应的潜在向量。
        if use_cache:
            if self.cache_c_kv is None:
                # 第一次调用(比如处理 prompt)，缓存为空，直接用新算出的部分
                latent_total = latent_new
            else:
                # 沿序列长度维(dim=1)拼接历史缓存与新 token 的潜在向量
                latent_total = torch.cat([self.cache_c_kv, latent_new], dim=1)
            self.cache_c_kv = latent_total
        else:
            latent_total = latent_new

        # 3) Up-project latent to per-head keys/values (then split into heads)
        # 中文：第三步——把(可能拼接了历史的)潜在向量 latent_total
        # 分别上投影还原成所有头的 K 和 V。注意这里是对"全部历史 token"
        # 的潜在向量做上投影，而不是只缓存/复用历史的 K、V 本身——
        # 这正是 MLA 与传统 KV 缓存的核心差异：缓存的是压缩后的 C，
        # 每次用的时候临时重新展开成 K、V。
        keys_all = self.W_UK(latent_total)   # (b, T_k_total, d_out)
        values_all = self.W_UV(latent_total)   # (b, T_k_total, d_out)

        # 4) Reshape to heads
        # 中文：第四步——把 Q、K、V 都拆分成多头形式，方便做
        # 逐头的注意力计算：(b, T, d_out) -> (b, num_heads, T, head_dim)
        queries = self._reshape_to_heads(queries_all, num_heads, head_dim)
        keys = self._reshape_to_heads(keys_all, num_heads, head_dim)
        values = self._reshape_to_heads(values_all, num_heads, head_dim)

        # 5) Scaled dot-product attention with causal mask
        # 中文：第五步——计算缩放点积注意力，并应用因果掩码(causal mask)，
        # 保证每个 query 位置只能看到"当前位置及之前"的 key 位置，
        # 不能看到未来的信息。
        # attn_scores 形状: (b, num_heads, T_q, T_k_total)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))

        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            # 中文：在增量解码时，Query 对应的绝对位置并不是从 0 开始，
            # 而是从当前已经处理过的位置 ptr_current_pos 继续往后数，
            # 这样才能和 Key 的绝对位置(0..T_k_total-1，因为 K 覆盖全部历史)
            # 正确比较，构造出跨越"历史缓存 + 新 token"的因果掩码。
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q
        else:
            # 不使用缓存时，Query 位置就是 0..T-1
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        # mask_bool[i, j] = True 表示 "query 位置 i 早于 key 位置 j"，
        # 即 j 是未来位置，需要被掩盖(不能被 attend 到)。
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        # Use the mask to fill attention scores
        # 中文：把被掩盖的位置的注意力分数设为 -inf，softmax 后这些位置
        # 的权重会趋近于 0，从而实现"看不到未来"的因果约束。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放点积注意力：除以 sqrt(head_dim) 防止数值过大导致 softmax
        # 梯度消失/饱和；沿最后一维(key 维度)做 softmax 得到注意力权重。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：用注意力权重对 V 做加权求和，得到每个头的上下文向量，
        # 再把 "头" 这一维转置回 "序列长度" 之前，方便下一步合并多头。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多头的输出重新拼接回 d_out 维(即 num_heads * head_dim)，
        # 恢复成 (b, T, d_out) 的形状。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：最后做一次线性投影(out_proj)，融合各头信息，得到该层
        # 注意力模块的最终输出。

        return context_vec


class LayerNorm(nn.Module):
    """层归一化(Layer Normalization)模块。

    对最后一维(特征维/embedding 维)做归一化，使其均值为 0、方差为 1，
    再通过可学习的缩放(scale)和平移(shift)参数恢复模型需要的表达能力。
    这是 Transformer 中稳定训练、加速收敛的关键组件之一。

    参数：
        emb_dim (int): 归一化的特征维度(通常等于 embedding 维度)。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # eps 防止除以 0 导致数值不稳定
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """对输入 x 的最后一维做归一化。

        参数:
            x: 形状 (..., emb_dim) 的张量。
        返回:
            与输入同形状的归一化后张量。
        """
        mean = x.mean(dim=-1, keepdim=True)
        # 使用有偏方差估计(unbiased=False)，这是 GPT-2 原始实现的做法
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 用可学习参数 scale、shift 对归一化结果做仿射变换
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数(高斯误差线性单元)的 tanh 近似实现。

    GPT-2 等模型在前馈网络中使用 GELU 而非 ReLU，因为它在 0 附近
    更平滑，通常能带来更好的训练效果。这里实现的是其常见的
    tanh 近似公式，而非精确的高斯误差函数形式。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """计算 GELU(x) 的 tanh 近似值，形状与输入 x 相同。"""
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer 中的逐位置前馈网络(Position-wise Feed-Forward Network)。

    结构为：先把特征维放大 4 倍(升维)，经过 GELU 非线性激活，
    再投影回原始维度(降维)。它对序列中每个位置独立地做同样的
    非线性变换，用于增强模型的表达能力。

    参数：
        cfg (dict): 配置字典，需要包含 "emb_dim" 键。
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

        参数:
            x: 形状 (b, T, emb_dim) 的张量。
        返回:
            形状同样为 (b, T, emb_dim) 的张量。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """一个标准的 Transformer 解码器块(此处为使用 MLA 的变体)。

    结构：
        x -> LayerNorm -> MLA 注意力 -> Dropout -> 残差相加
          -> LayerNorm -> 前馈网络 -> Dropout -> 残差相加

    这里采用的是 Pre-LayerNorm 结构(先归一化再进入子层)，
    是 GPT-2 及后续多数大模型采用的做法，有利于训练稳定性。

    参数：
        cfg (dict): 配置字典，需包含 emb_dim、n_heads、drop_rate、
            qkv_bias、latent_dim 等键。
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadLatentAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            latent_dim=cfg["latent_dim"])

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        """Transformer 块的前向传播。

        参数：
            x: 输入张量，形状 (b, T, emb_dim)。
            use_cache (bool): 是否启用 KV(潜在向量)缓存，透传给内部
                的 MultiHeadLatentAttention。
        返回：
            形状同样为 (b, T, emb_dim) 的张量。
        """
        # Shortcut connection for attention block
        # 中文：注意力子层的残差连接——先保存输入，供之后相加
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        # 中文：把 use_cache 参数透传给 MLA 注意力层，决定是否走
        # 增量式(带缓存)的推理路径。
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差相加，缓解深层网络的梯度消失问题，
        # 也让模型可以更容易地"学习恒等映射"

        # Shortcut connection for feed-forward block
        # 中文：前馈网络子层的残差连接，逻辑与上面对称
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的 GPT 风格解码器模型(此处的注意力层使用 MLA 实现)。

    结构：
        token embedding + position embedding -> dropout
        -> N 层 TransformerBlock -> 最终 LayerNorm -> 输出线性层(词表大小)

    支持两种前向模式：
      - use_cache=False：标准的一次性前向(如训练时，或不使用缓存推理)。
      - use_cache=True：增量式推理，配合每层 MLA 内部的潜在向量缓存，
        实现"只计算新 token"的高效自回归生成。

    参数：
        cfg (dict): 配置字典，需包含 vocab_size、emb_dim、context_length、
            drop_rate、n_layers 等键(详见 main() 中的 GPT_CONFIG_124M)。
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 词嵌入：把 token id 映射为 emb_dim 维的向量
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 位置嵌入：为每个绝对位置学习一个 emb_dim 维的向量，
        # 用于给模型注入序列顺序信息(GPT-2 风格的可学习绝对位置编码)
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache-related
        # 中文：这里从 nn.Sequential 改为 nn.ModuleList，是因为使用
        # KV 缓存时，forward 需要手动地把 use_cache 参数逐层传递下去
        # (nn.Sequential 的 forward 只支持单一输入，无法透传额外参数)。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0
        # current_pos：记录模型级别当前已经生成/处理到的绝对位置，
        # 用于在 use_cache=True 时给新 token 分配正确的位置编码。
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 输出头：把最后的隐藏状态投影回词表大小，得到每个 token 的 logits

    def forward(self, in_idx, use_cache=False):
        """GPT 模型前向传播。

        参数：
            in_idx: 输入 token id 张量，形状 (batch_size, seq_len)。
            use_cache (bool): 是否启用 KV(潜在向量)缓存的增量推理模式。
        返回：
            logits: 形状 (batch_size, seq_len, vocab_size) 的张量，
            表示每个位置对词表中每个 token 的预测分数(未归一化)。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        # 中文：与不带缓存的版本相比，这里需要根据 use_cache 状态
        # 计算正确的"绝对位置"来查位置嵌入表，而不是总是从 0 开始。
        if use_cache:
            # 增量推理：位置编码从 current_pos 开始，接着上次结束的地方继续
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            # 非增量模式：位置编码固定从 0 开始编号
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        # unsqueeze(0) 在最前面加一个 batch 维，方便与 tok_embeds 相加时广播
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：词嵌入 + 位置嵌入，得到每个 token 融合了内容与位置信息的表示
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        #  KV cache-related
        # 中文：手动逐层遍历 ModuleList，并把 use_cache 透传给每个
        # TransformerBlock(进而透传给内部的 MLA 注意力层)。
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    ####################################################
    #  KV cache-related
    def reset_kv_cache(self):
        """重置整个模型的 KV(潜在向量)缓存状态。

        会依次调用每一层注意力模块的 reset_cache()，并把模型级别的
        位置指针 current_pos 归零。通常在开始新一轮生成前调用，
        避免不同生成任务/序列之间相互污染缓存。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用贪心解码(greedy decoding)自回归生成文本，支持 KV 缓存加速。

    参数：
        model: GPTModel 实例。
        idx: 初始输入 token id 张量，形状 (batch_size, seq_len)，
            通常是编码后的提示词(prompt)。
        max_new_tokens (int): 需要新生成的 token 数量。
        context_size (int, optional): 模型能接受的最大上下文长度，
            默认使用模型位置嵌入表的大小(model.pos_emb.num_embeddings)。
        use_cache (bool): 是否启用 KV 缓存加速生成。
            - True：只需在最初把完整 prompt 过一遍模型建立缓存，
              之后每一步只需把"新生成的单个 token"喂给模型，
              大幅减少重复计算。
            - False：朴素做法，每一步都把当前完整序列(受 context_size
              截断)重新完整地过一遍模型，计算量随生成长度增长而增大。

    返回：
        idx: 形状 (batch_size, seq_len + max_new_tokens) 的张量，
            即原始输入拼接上新生成的 token id 序列。
    """
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        # 推理阶段不需要计算梯度，节省显存/加速
        if use_cache:
            # Init cache with full prompt
            # 中文：先清空历史缓存，再用完整的 prompt(截断到 ctx_len 以内)
            # 做一次前向，这一步会把 prompt 中每个 token 的潜在向量都
            # 写入缓存，为后续逐 token 生成做准备。
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                # 中文：贪心采样——直接取概率(或者说 logits)最大的那个 token，
                # 不做随机采样，因此生成结果是确定性的。
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                # 中文：把新生成的 token 拼接到输出序列末尾
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                # 中文：得益于 KV 缓存，这里只需要把"刚生成的这一个新 token"
                # 喂给模型，模型内部会自动结合缓存中的历史潜在向量信息，
                # 不需要重新计算之前所有 token 的 K/V，这正是缓存加速的核心。
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                # 中文：不使用缓存时，每一步都要把(截断后的)完整历史序列
                # 重新完整地喂给模型，计算量会随生成长度线性增长，
                # 用于和上面 use_cache=True 的效率做对比。
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    """脚本入口：解析命令行参数，构建 GPT 模型并运行一次文本生成，
    最后打印生成结果、耗时、生成速度(tokens/sec)以及(若使用 GPU)
    显存占用情况，用于直观感受 MLA + KV 缓存带来的效果。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run GPT with standard multi-head attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--latent_dim", type=int, default=None,
        help="Latent dim for MLA")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    # 中文：使用 GPT-2 的 BPE 分词器把起始文本编码成 token id 列表
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        # 中文：context_length 设置为"待生成 token 数 + 初始提示词长度"，
        # 保证位置嵌入表足够覆盖整个生成过程中的所有绝对位置。
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
        "latent_dim": args.latent_dim,
    }
    torch.manual_seed(123)
    # 中文：固定随机种子，保证模型参数初始化可复现
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    # 中文：使用 bfloat16 精度加载模型，减少显存占用并(在支持的硬件上)加速推理
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    # unsqueeze(0)：加上 batch 维，形状变为 (1, prompt_len)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 中文：GPU 上的操作是异步的，计时前先同步，确保之前的操作都已完成
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 中文：同理，计时结束前也要同步，保证生成过程真正执行完毕
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    # 中文：把生成的 token id 序列解码回可读文本

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
        # 中文：打印 GPU 上记录到的峰值显存占用，方便直观比较 MLA
        # 在显存效率上相对标准 MHA 的优势。


if __name__ == "__main__":
    main()
