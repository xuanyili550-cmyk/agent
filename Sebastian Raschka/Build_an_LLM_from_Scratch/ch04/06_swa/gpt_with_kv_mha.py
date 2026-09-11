# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

# ============================================================
# 中文说明（模块级 docstring 补充）
# ------------------------------------------------------------
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
# 第 3~4 章代码的汇总版本,实现了一个"标准多头自注意力"(Multi-Head Attention,
# 简称 MHA)版本的 GPT 模型,并在此基础上加入了 **KV 缓存(KV Cache)** 机制,
# 用于加速自回归(autoregressive)文本生成。
#
# 核心内容包括:
#   1. MultiHeadAttention:带因果掩码(causal mask)的多头自注意力层,
#      支持 KV 缓存,即推理时把历史 token 的 Key/Value 张量缓存下来,
#      避免每生成一个新 token 都要对整个序列重新计算 K、V。
#   2. LayerNorm / GELU / FeedForward:Transformer 块中的归一化层、
#      激活函数和前馈网络(FFN)。
#   3. TransformerBlock:把注意力子层和前馈子层用残差连接(shortcut)
#      和 LayerNorm 组合起来的标准 Transformer 层。
#   4. GPTModel:堆叠多个 TransformerBlock,加上词嵌入(token embedding)
#      和位置嵌入(position embedding),构成完整的 GPT 模型。
#   5. generate_text_simple_cached:演示如何利用 KV 缓存做贪婪解码
#      (greedy decoding),对比"有缓存"和"无缓存"两种生成方式。
#   6. main:命令行入口,构建一个 GPT-124M 规模的模型并测量生成速度。
#
# 本文件所在目录 06_swa(Sliding Window Attention,滑动窗口注意力)是
# 该书配套代码仓库中用于对比"标准多头注意力 + KV 缓存"与"滑动窗口注意力"
# 等不同注意力实现的性能/内存基准脚本之一。这里的版本是最基础的对照组:
# 使用完整的因果注意力(不做窗口裁剪),但已经加入 KV 缓存优化。
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
    """多头自注意力(Multi-Head Attention)模块,支持因果掩码与 KV 缓存。

    这是标准的"分头"实现:先用三个线性层把输入投影为 Query/Key/Value,
    再把最后一维拆分成 (num_heads, head_dim),让每个头在较低维度上
    独立计算注意力,最后再拼接起来通过输出投影层。

    参数:
        d_in (int): 输入特征维度(即每个 token 向量的维度)。
        d_out (int): 输出特征维度,同时也是所有注意力头拼接后的总维度。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头的数量,要求 d_out 能被 num_heads 整除。
        qkv_bias (bool): 是否在 Q/K/V 的线性层中使用偏置项。

    关键张量形状(设 batch=b, 序列长度=num_tokens):
        输入 x: (b, num_tokens, d_in)
        输出 context_vec: (b, num_tokens, d_out)
    """
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文:每个注意力头分到的维度 = 总输出维度 / 头数,
        # 这样多头拼接后维度依然等于 d_out,不会增加参数量或计算量的量级。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # KV cache-related code
        # 中文:KV 缓存相关代码。
        # register_buffer 注册的是"非参数"张量,不会被优化器更新,
        # 但会随模型一起 .to(device) 移动,persistent=False 表示
        # 不会被保存进 state_dict(因为它只是运行时的临时状态)。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0  # 中文:记录当前已经处理到序列的第几个位置,用于因果掩码的位置对齐
        ####################################################

    def forward(self, x, use_cache=False):
        """前向传播:计算多头自注意力输出。

        参数:
            x (Tensor): 输入张量,形状 (b, num_tokens, d_in)。
                - 若 use_cache=True 且缓存已存在,x 通常只是"新增的
                  一个或几个 token"(例如自回归生成时每步只喂入 1 个新 token)。
            use_cache (bool): 是否启用 KV 缓存。True 时会把本次算出的
                K、V 追加到缓存中,并用完整的历史 K、V 计算注意力;
                False 时每次都是"从零"计算,不保留历史。

        返回:
            Tensor: 注意力输出,形状 (b, num_tokens, d_out)。
                注意这里的 num_tokens 是本次前向传播输入 x 的 token 数,
                不是缓存后的总长度。
        """
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)
        # 中文:queries/keys_new/values_new 形状均为 (b, num_tokens, d_out)，
        # 分别是本次输入 token 对应的 Q、K、V。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文:把 d_out 这一维"重塑"拆分成 (num_heads, head_dim)，
        # 相当于把一个大的线性投影结果切成 num_heads 份，让每个头独立
        # 关注不同的表示子空间，这就是"多头"的由来。
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache-related
        # 中文:KV 缓存的核心逻辑。
        # 生成第一个 token 时（或不使用缓存时）cache_k/cache_v 为 None，
        # 直接把新算出的 K、V 当作缓存；后续每一步只算"新 token"的 K、V，
        # 再沿着序列长度维（dim=1）拼接到历史缓存后面，这样注意力计算时
        # 就能"看到"完整的历史上下文，而不需要重新计算历史 token 的 K、V，
        # 这正是 KV 缓存能显著加速自回归生成的原因。
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文:把 num_heads 提到序列长度前面，这样后面的矩阵乘法
        # 可以把 (num_heads) 当作"批次"维度，对每个头并行做注意力计算。
        # 注意：这里的 keys/values 的 num_tokens 可能是缓存后的"总长度"，
        # 而 queries 的 num_tokens 只是"本次新增"的长度，两者可以不同。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文:queries 形状 (b, num_heads, num_tokens_Q, head_dim)
        # keys.transpose(2,3) 形状 (b, num_heads, head_dim, num_tokens_K)
        # 相乘后 attn_scores 形状为 (b, num_heads, num_tokens_Q, num_tokens_K)，
        # 即"每个新 query token 对每个历史 key token"的注意力打分。

        ####################################################
        # causal mask
        # 中文:构造因果掩码（causal mask），保证每个 query 位置只能
        # "看到"它自己及之前的 key 位置，不能看到未来的位置，这是
        # GPT 这类自回归语言模型的核心约束。
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            # 中文:使用缓存时，本次 query 对应的"绝对位置"不是从 0 开始，
            # 而是接着上一次处理到的位置 ptr_current_pos 继续往后编号，
            # 这样才能保证掩码在"跨多次前向调用"时依然正确对齐。
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q
        else:
            # 中文:不使用缓存时，每次都是完整序列从头算起，
            # query 的绝对位置就是 0..num_tokens_Q-1，并把指针清零。
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        # 中文:mask_bool[i, j] = True 表示"query 位置 i 小于 key 位置 j"，
        # 即 key 在未来，需要被掩盖（禁止看到），形状通过广播得到
        # (num_tokens_Q, num_tokens_K)。
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        # Use the mask to fill attention scores
        # 中文:把未来位置对应的注意力分数填成 -inf，
        # 这样经过 softmax 后这些位置的权重会变成 0，
        # 从而实现"看不到未来"的因果约束。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 中文:除以 sqrt(head_dim) 做缩放（scaled dot-product），
        # 防止点积数值过大导致 softmax 梯度消失，这就是
        # "Scaled Dot-Product Attention"名字的由来。
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文:attn_weights (b, num_heads, num_tokens_Q, num_tokens_K) 与
        # values (b, num_heads, num_tokens_K, head_dim) 相乘，得到
        # 每个 query 位置的加权上下文向量，形状
        # (b, num_heads, num_tokens_Q, head_dim)，再 transpose(1,2)
        # 换回 (b, num_tokens_Q, num_heads, head_dim)。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文:把多个头的输出重新拼接（reshape）回 (b, num_tokens, d_out)，
        # 相当于把各头独立学到的子空间表示拼在一起。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文:再经过一次线性投影，让模型有机会把拼接后的多头信息
        # 重新融合、混合，是标准 Transformer 注意力层的最后一步。

        return context_vec

    def reset_cache(self):
        """清空 KV 缓存并重置位置指针。

        用于每次开始一段新的生成（新的 prompt）之前调用，
        避免把上一次生成残留的历史 K/V 错误地拼接到本次序列上。
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化(Layer Normalization)。

    对每个 token 的特征向量（最后一维）做归一化，使其均值为 0、方差为 1，
    再通过可学习的缩放(scale)和平移(shift)参数恢复模型需要的表达能力。
    这是 Transformer 中常见的稳定训练的手段。

    参数:
        emb_dim (int): 特征维度大小，即需要归一化的最后一维长度。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文:极小值，防止除以 0 导致数值不稳定
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 中文:可学习缩放参数 gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 中文:可学习平移参数 beta

    def forward(self, x):
        """对输入张量最后一维做归一化。

        参数:
            x (Tensor): 形状 (..., emb_dim)，通常为 (b, num_tokens, emb_dim)。
        返回:
            Tensor: 与输入同形状的归一化结果。
        """
        mean = x.mean(dim=-1, keepdim=True)
        # 中文:unbiased=False 表示用有偏方差估计（除以 N 而不是 N-1），
        # 与大多数深度学习框架的 LayerNorm 实现保持一致。
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数（GPT-2 使用的 tanh 近似版本）。

    相比 ReLU，GELU 是平滑的非线性函数，在 GPT 系列模型的
    前馈网络（FeedForward）中被广泛使用。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """计算 GELU 激活值。

        使用的是原始 GPT-2 论文中给出的 tanh 近似公式：
            0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        而不是精确的误差函数(erf)形式，二者数值上非常接近。

        参数:
            x (Tensor): 任意形状的输入张量。
        返回:
            Tensor: 与输入同形状。
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer 中的前馈网络(Position-wise FeedForward Network)。

    结构为：线性升维(emb_dim -> 4*emb_dim) -> GELU 激活 ->
    线性降维(4*emb_dim -> emb_dim)。这个"先放大再缩小"的结构
    能让模型在更高维空间中做非线性变换，增强表达能力。

    参数:
        cfg (dict): 模型配置字典，需要包含 "emb_dim" 键。
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
            x (Tensor): 形状 (b, num_tokens, emb_dim)。
        返回:
            Tensor: 形状 (b, num_tokens, emb_dim)，与输入相同（用于残差相加）。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """一个标准的 Transformer 解码器层（Pre-LayerNorm 结构）。

    包含两个子层：
        1. 多头自注意力子层（带残差连接 + LayerNorm）。
        2. 前馈网络子层（带残差连接 + LayerNorm）。
    每个子层都遵循 "LayerNorm -> 子层 -> Dropout -> 残差相加" 的顺序，
    即 Pre-Norm 结构，这有助于训练更深的网络时保持梯度稳定。

    参数:
        cfg (dict): 模型配置字典，需要包含 emb_dim、n_heads、drop_rate、qkv_bias 等键。
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
        """前向传播。

        参数:
            x (Tensor): 输入张量，形状 (b, num_tokens, emb_dim)。
            use_cache (bool): 是否启用 KV 缓存，会透传给内部的注意力层。
        返回:
            Tensor: 输出张量，形状与输入相同 (b, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        # 中文:注意力子层的残差连接（shortcut），先保存输入 x 的副本，
        # 之后与经过注意力处理的结果相加，缓解深层网络的梯度消失问题。
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        # 中文:把 use_cache 参数透传给注意力层，由它决定是否读写 KV 缓存。
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        # 中文:前馈网络子层同样使用残差连接。
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的 GPT 模型（解码器架构，仅有 Transformer 的 Decoder 部分）。

    组成部分：
        - 词嵌入(token embedding)：把 token id 映射为向量。
        - 位置嵌入(position embedding)：为每个位置提供可学习的位置编码
          （这是 GPT-2 风格的绝对位置编码，而不是 RoPE 等相对位置编码）。
        - 多层 TransformerBlock 堆叠。
        - 最终 LayerNorm 和输出线性层（把隐藏状态映射回词表大小的 logits）。

    参数:
        cfg (dict): 模型配置字典，需要包含 vocab_size、emb_dim、
            context_length、drop_rate、n_layers 等键。
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
        # 中文:这里从 nn.Sequential 改成 nn.ModuleList，是因为
        # nn.Sequential 的 forward 只能接收单一位置参数，
        # 而我们需要在每一层之间传递 use_cache 这个额外参数，
        # 所以改用 ModuleList 并在下面手动写循环调用每一层。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0  # 中文:记录整个模型当前已经生成到的绝对位置，用于位置嵌入的正确对齐
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        """前向传播，计算下一个 token 的预测 logits。

        参数:
            in_idx (Tensor): 输入 token id 序列，形状 (batch_size, seq_len)，dtype 为 long。
            use_cache (bool): 是否启用 KV 缓存。
                - True 且已有缓存时，通常 in_idx 只包含"新增的" token
                  （例如自回归生成时每步只传入最新生成的 1 个 token）。
                - False 时，每次都对传入的完整序列从头计算。

        返回:
            Tensor: logits，形状 (batch_size, seq_len, vocab_size)，
                表示词表中每个 token 的未归一化预测分数。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文:tok_embeds 形状 (batch_size, seq_len, emb_dim)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        # 中文:位置编码也需要感知 KV 缓存带来的"绝对位置偏移"。
        # 如果不使用缓存，每次都是从位置 0 开始编号；
        # 如果使用缓存，本次输入的 token 是接着上一次的位置继续编号的，
        # 否则模型会误以为每次生成的 token 都处于序列开头，位置编码会错乱。
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        # 中文:pos_embeds 形状 (1, seq_len, emb_dim)，unsqueeze(0) 是为了
        # 能够以广播的方式与 (batch_size, seq_len, emb_dim) 的 tok_embeds 相加。
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文:词嵌入与位置嵌入直接相加，融合"token 是什么"和"token 在哪里"两种信息。
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        # 中文:依次调用每一层 TransformerBlock，并把 use_cache 逐层透传下去，
        # 让每一层各自维护自己的 KV 缓存。
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        # 中文:logits 形状 (batch_size, seq_len, vocab_size)
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        """重置模型所有层的 KV 缓存以及位置计数器。

        在开始生成一段新文本（新的 prompt）之前必须调用，
        否则会把上一次生成残留的缓存和位置计数错误地延续到本次生成中，
        导致注意力掩码和位置编码出错。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用贪婪解码(greedy decoding)生成文本，演示 KV 缓存的加速效果。

    参数:
        model (GPTModel): 已训练（或随机初始化）的 GPT 模型。
        idx (Tensor): 起始 token id 序列（prompt），形状 (batch_size, prompt_len)。
        max_new_tokens (int): 要新生成的 token 数量。
        context_size (int, optional): 最大上下文窗口长度，超出部分会被截断
            （仅在不使用缓存或首次喂入 prompt 时用于截取 idx 的末尾片段）。
            默认使用模型位置嵌入支持的最大长度。
        use_cache (bool): 是否启用 KV 缓存来加速生成。
            - True：只需在第一次前向传播时处理完整 prompt 并建立缓存，
              之后每一步只需要把"新生成的 1 个 token"喂给模型，
              模型内部会自动拼接历史 K/V，从而把每步的计算量从
              O(序列长度) 降到 O(1)（相对单步而言）。
            - False：每一步都要把"当前完整序列（可能被截断到 context_size）"
              重新喂给模型，完全重新计算注意力，速度较慢，但实现更简单、
              便于验证正确性（可作为与 use_cache=True 结果对比的基准）。

    返回:
        Tensor: 拼接了 prompt 和新生成 token 的完整序列，
            形状 (batch_size, prompt_len + max_new_tokens)。
    """
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            # 中文:先重置缓存，再用完整 prompt 做一次前向传播，
            # 这一步会把 prompt 中所有 token 的 K、V 都存入缓存。
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                # 中文:贪婪采样——直接取概率（logits）最大的 token，不做随机采样。
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                # 中文:关键优化点——后续每一步只把"刚生成的这 1 个新 token"
                # 传给模型，而不是整个历史序列，因为历史的 K、V 已经缓存好了，
                # 模型内部会自动把新 token 的 K、V 拼接到缓存后面继续计算注意力。
                logits = model(next_idx, use_cache=True)
        else:
            # 中文:不使用缓存的朴素做法——每一步都把（截断后的）完整序列
            # 重新喂给模型，模型需要重新计算所有历史 token 的 K、V，
            # 计算量随序列变长而线性增长，是性能对比的基准（baseline）。
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    """命令行入口：构建一个 GPT-124M 规模的模型，测试带 KV 缓存的文本生成速度。

    流程：
        1. 解析命令行参数（模型规模、生成长度等）。
        2. 用 tiktoken 的 GPT-2 分词器对起始文本编码。
        3. 构建 GPTModel（使用 bfloat16 精度，若有 GPU 则使用 GPU）。
        4. 调用 generate_text_simple_cached 生成文本并计时。
        5. 打印生成结果、耗时、每秒生成 token 数以及显存占用（若有 GPU）。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run GPT with standard multi-head attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        # 中文:上下文长度设置为"新生成 token 数 + prompt 长度"，
        # 保证位置嵌入表 pos_emb 足够覆盖整个生成过程用到的所有位置。
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
    }
    torch.manual_seed(123)  # 中文:固定随机种子，保证模型初始化权重可复现
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    # 中文:使用 bfloat16 半精度可以减少显存占用、提升计算速度，
    # 是大模型推理时常见的做法。
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    # 中文:unsqueeze(0) 增加 batch 维，形状变为 (1, prompt_len)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 中文:同步 CUDA 流，确保之前的 GPU 操作全部完成后再开始计时，
        # 避免异步执行导致计时不准确。
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 中文:同样，在结束计时前先同步，确保所有 GPU 计算真正完成。
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


if __name__ == "__main__":
    main()
