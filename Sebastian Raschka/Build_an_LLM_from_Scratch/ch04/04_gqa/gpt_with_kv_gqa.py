# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4, adapted to use Grouped-Query Attention (GQA).
# This file can be run as a standalone script.

"""
【中文模块说明】
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书中
第 3~4 章代码的汇总与扩展版本，属于 ch04/04_gqa 附加材料。

它在原始 GPT 模型（第 4 章）的基础上做了两处关键改造：
1. 用「分组查询注意力」(Grouped-Query Attention, GQA) 替换了标准的多头自注意力
   (Multi-Head Attention, MHA)。GQA 让多个 Query 头共享同一组 Key/Value 头，
   从而大幅减少 KV 的显存占用和计算量，是 LLaMA-2/3、Mistral 等现代大模型
   常用的注意力变体。
2. 引入了「KV 缓存」(KV Cache) 机制：在自回归生成（一个 token 一个 token 地生成）
   时，历史 token 的 Key/Value 向量可以缓存下来，新生成的 token 只需要计算自己
   的 Key/Value 并拼接到缓存后面，而不必每次都对整个序列重新计算注意力，从而
   显著加速推理。

整体结构（自底向上）：
    GroupedQueryAttention  -> 实现 GQA + KV 缓存的注意力层
    LayerNorm               -> 层归一化
    GELU / FeedForward      -> 前馈网络（MLP）模块
    TransformerBlock        -> 由注意力 + 前馈网络组成的一个 Transformer 块
    GPTModel                -> 堆叠多个 TransformerBlock 构成的完整 GPT 模型
    generate_text_simple_cached -> 使用（或不使用）KV 缓存进行自回归文本生成
    main                    -> 命令行入口，构建模型并跑一次文本生成，打印耗时/显存等信息

本文件可以直接作为脚本运行：`python gpt_with_kv_gqa.py`。
"""

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# NEW: GQA instead of MHA
#####################################
class GroupedQueryAttention(nn.Module):
    """分组查询注意力（GQA）层，并内置 KV 缓存支持。

    与标准多头注意力（MHA）不同，GQA 让 `num_heads` 个 Query 头共享
    `num_kv_groups` 组 Key/Value（其中 `num_kv_groups` < `num_heads`），
    每组 KV 被 `group_size = num_heads // num_kv_groups` 个 Query 头共享。
    这样可以显著减少 K、V 投影矩阵的参数量，以及推理时 KV 缓存占用的显存。

    当 `num_kv_groups == num_heads` 时，GQA 退化为普通的 MHA；
    当 `num_kv_groups == 1` 时，GQA 退化为「多查询注意力」(MQA)。

    参数说明：
        d_in (int): 输入特征维度（即 embedding 维度）。
        d_out (int): 输出特征维度，同时也是所有 Query 头拼接后的总维度，
            必须能被 num_heads 整除。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): Query 头的数量。
        num_kv_groups (int): Key/Value 的分组数量，必须能整除 num_heads。
        dtype: 线性层参数的数据类型（如 torch.bfloat16），可选。
        qkv_bias (bool): Q/K/V 线性投影是否使用偏置项。
    """

    def __init__(
            self, d_in, d_out, dropout, num_heads, num_kv_groups, dtype=None, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        # 上面两个断言保证：
        # 1) 输出维度可以被平均分给每个 Query 头（每个头维度 = d_out / num_heads）；
        # 2) Query 头数量可以被 KV 组数整除，这样每组 KV 恰好对应整数个 Query 头。

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个注意力头的维度

        # 注意：K、V 的投影输出维度是 num_kv_groups * head_dim，而不是
        # num_heads * head_dim —— 这正是 GQA 相比 MHA 节省参数/显存的关键所在。
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups  # 每组 KV 被多少个 Query 头共享

        # Query 的投影维度仍然是 d_out = num_heads * head_dim，与标准 MHA 一致。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        # register_buffer 将 KV 缓存注册为模型的 buffer（而非可训练参数），
        # persistent=False 表示它不会被保存进 state_dict（因为它只是运行时的临时状态）。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0  # 记录当前已经处理到序列的第几个位置，用于生成因果掩码

    def forward(self, x, use_cache=False):
        """前向传播：计算 GQA 自注意力输出。

        参数：
            x (Tensor): 输入张量，形状 (b, num_tokens, d_in)。
                - 首次调用（或非缓存模式）时 num_tokens 通常是整段 prompt 的长度；
                - 使用 KV 缓存做增量解码时，num_tokens 通常为 1（只传入新 token）。
            use_cache (bool): 是否启用 KV 缓存增量解码模式。

        返回：
            Tensor: 注意力输出，形状 (b, num_tokens, d_out)，与输入的 num_tokens 一致
                （即只返回当前这次前向传播对应的那部分 token 的结果，
                历史 token 的表示不会被重复返回）。
        """
        b, num_tokens, _ = x.shape

        # Apply projections
        # 分别用三个线性层将输入映射为 Query / Key / Value
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        # 把最后一维拆分成 (头数, 每头维度)，再把「头」维度换到第 2 维，方便按头做矩阵乘法。
        # queries: (b, num_heads, num_tokens, head_dim)
        # keys_new/values_new: (b, num_kv_groups, num_tokens, head_dim)  —— 注意头数是 num_kv_groups，比 Query 少
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if use_cache:
            # 增量解码模式：把这一步新算出来的 K/V 拼接到历史缓存后面（沿 num_tokens 所在的维度，即 dim=2）。
            if self.cache_k is None:
                # 首次调用（通常是喂入整个 prompt），直接把缓存初始化为当前的 K/V
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                # 后续调用（每次只喂 1 个新 token），把新 K/V 追加到已有缓存之后
                # 拼接后形状: (b, num_kv_groups, 历史长度+num_tokens, head_dim)
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
            keys_base, values_base = self.cache_k, self.cache_v
        else:
            # 非缓存模式（如训练时的一次性前向传播）：直接用当前算出的 K/V，
            # 不使用、也不保留历史缓存。
            keys_base, values_base = keys_new, values_new
            if self.cache_k is not None or self.cache_v is not None:
                # 如果之前处于缓存模式、现在切换回非缓存模式，需要清空旧缓存，避免脏数据。
                self.cache_k, self.cache_v = None, None
                self.ptr_current_pos = 0

        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)
        # 核心 GQA 操作：把 num_kv_groups 组 K/V 沿「头」维度复制扩展成 num_heads 份，
        # 这样每个 Query 头都能找到与之对应的 K/V（同一组内的 Query 头会用到相同的 K/V）。
        # repeat_interleave 是「逐元素重复」，保证同一组内的头是相邻的。
        keys = keys_base.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values_base.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]
        # 中文解释：假设 num_kv_groups=2、group_size=2（即 num_heads=4）：
        #   repeat_interleave 得到 [K1, K1, K2, K2] —— 第 0、1 个 Query 头用 K1，第 2、3 个 Query 头用 K2；
        #   如果误用普通的 repeat，会得到 [K1, K2, K1, K2]，导致头与 KV 组的对应关系完全错乱。
        #   这也是本实现选用 repeat_interleave 而非 repeat 的原因。

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (b, num_heads, num_tokens, num_tokens)
        # 计算缩放点积注意力分数：Q @ K^T，对最后两维做矩阵乘法（每个 batch、每个头独立计算）。
        # 这里的 keys 已经是扩展后的 (b, num_heads, K序列长度, head_dim)，
        # 注意 K 的序列长度在使用缓存时是「历史长度 + 当前长度」，而 Q 的序列长度只是「当前长度」，
        # 所以 attn_scores 的形状实际上是 (b, num_heads, num_tokens_Q, num_tokens_K)。
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        ####################################################
        # causal mask
        # 因果掩码（causal mask）：保证每个位置只能看到「自己以及之前」的 token，不能看到未来的 token。
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            # 使用 KV 缓存时，Query 对应的绝对位置不是从 0 开始的，
            # 而是从 ptr_current_pos（即已经处理过的 token 数）开始的，
            # 这样才能正确判断「当前 Query 位置」相对「所有 Key 位置（含历史）」谁在前谁在后。
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q  # 更新指针，为下一次增量解码做准备
        else:
            # 非缓存模式：Query 位置就是 0..num_tokens_Q-1（标准的一次性前向传播）
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        # mask[i, j] = True 表示「Query 位置 i 在 Key 位置 j 之前」，即 j 是未来 token，需要被屏蔽。
        # 例如非缓存模式下这就是标准的上三角掩码；缓存模式下由于 q_positions 有偏移，
        # 历史 Key（位置更小）永远不会被屏蔽，只会屏蔽当前新 token 看不到的「未来」位置（这里通常为空）。
        mask = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        # Use the mask to fill attention scores
        # 把被掩码标记的位置的注意力分数设为 -inf，softmax 之后这些位置的权重会变成 0，
        # 从而实现「看不到未来」的因果约束。
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # 缩放点积注意力：除以 sqrt(head_dim) 防止点积数值过大导致 softmax 梯度消失/饱和。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 用注意力权重对 Value 做加权求和，得到每个 Query 位置的上下文向量；
        # 再把「头」维度换回第 2 维之前的位置，方便后续拼接所有头。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 把 num_heads 个头的输出在最后一维拼接（concat）回 d_out 维度。
        # contiguous() 是因为 transpose 之后张量在内存中不连续，view 需要连续内存。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection  # 输出投影，融合各头信息

        return context_vec

    def reset_cache(self):
        """清空 KV 缓存和位置指针，用于开始新一轮生成之前重置状态。"""
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）。

    对最后一个维度（通常是 embedding 维度）做归一化，使其均值为 0、方差为 1，
    再通过可学习的缩放（scale）和平移（shift）参数恢复模型需要的表达能力。
    这是 Transformer 中稳定训练、加速收敛的关键组件之一。

    参数：
        emb_dim (int): 需要归一化的特征维度大小（embedding 维度）。
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习的缩放参数 gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的平移参数 beta

    def forward(self, x):
        """对输入的最后一维做归一化。

        参数：
            x (Tensor): 形状 (..., emb_dim)，一般是 (batch, seq_len, emb_dim)。

        返回：
            Tensor: 与输入同形状，已完成归一化 + 缩放/平移。
        """
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False 表示使用有偏方差估计（分母为 N 而非 N-1），与常见深度学习框架实现一致
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数（这里使用的是 GPT-2 采用的 tanh 近似公式）。

    公式：0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    相比 ReLU，GELU 是平滑、处处可导的激活函数，被 GPT 系列模型广泛使用。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """对输入逐元素应用 GELU 激活函数，输入输出形状相同。"""
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer 中的前馈网络（FFN / MLP）子层。

    结构：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)
    先升维再降维（升维倍数通常为 4），为模型提供逐位置（position-wise）的非线性变换能力。

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
        """前向传播，输入输出形状均为 (batch, seq_len, emb_dim)。"""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """一个标准的 Transformer 块：GQA 自注意力 + 前馈网络，均带残差连接和前置归一化（Pre-LN）。

    结构（Pre-LN 风格）：
        x -> LayerNorm -> GQA 注意力 -> Dropout -> 残差相加
          -> LayerNorm -> 前馈网络   -> Dropout -> 残差相加

    参数：
        cfg (dict): 配置字典，需要包含 emb_dim、n_heads、n_kv_groups、drop_rate、qkv_bias 等键。
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        """前向传播。

        参数：
            x (Tensor): 输入张量，形状 (batch_size, num_tokens, emb_dim)。
            use_cache (bool): 是否启用 KV 缓存（透传给内部的 GroupedQueryAttention）。

        返回：
            Tensor: 与输入同形状 (batch_size, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        # 注意力子层的残差连接：先保存输入作为 shortcut，归一化后送入注意力层
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        # 中文：这里把 use_cache 参数透传给注意力层，由注意力层内部决定是否读写 KV 缓存
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back  # 残差相加，缓解深层网络梯度消失问题

        # Shortcut connection for feed-forward block
        # 前馈网络子层的残差连接，结构与上面完全对称
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的 GPT 模型：词嵌入 + 位置嵌入 + 多层 TransformerBlock + 输出层，支持 KV 缓存推理。

    参数：
        cfg (dict): 配置字典，需要包含：
            vocab_size, context_length, emb_dim, n_heads, n_layers,
            drop_rate, qkv_bias, n_kv_groups。
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])       # token 嵌入表
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])   # 可学习的绝对位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache-related
        # 中文：原本用 nn.Sequential 堆叠即可，但 Sequential 的 forward 只支持单一输入，
        # 无法传递 use_cache 这个额外参数；因此这里改用 nn.ModuleList，
        # 在下面的 forward 中手动写循环、逐层传参。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0  # 记录整个模型当前已经处理到序列的第几个位置（用于位置嵌入的增量计算）
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出投影到词表大小，得到 logits

    def forward(self, in_idx, use_cache=False):
        """前向传播。

        参数：
            in_idx (Tensor): 输入 token id 序列，形状 (batch_size, seq_len)，dtype 为整型。
            use_cache (bool): 是否启用 KV 缓存的增量解码模式。

        返回：
            Tensor: logits，形状 (batch_size, seq_len, vocab_size)。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # (batch_size, seq_len, emb_dim)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        # 中文：非缓存模式下位置永远从 0 开始；但增量解码时，每次只喂入新 token，
        # 其真实位置应该接着上一次的位置继续编号，因此需要用 self.current_pos 做偏移。
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len  # 更新全局位置指针
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)  # (1, seq_len, emb_dim)，会广播到 batch 维
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]  # token 嵌入 + 位置嵌入
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        # 中文：手动遍历每一层 TransformerBlock，并把 use_cache 透传下去，
        # 这样每一层内部的注意力模块都能各自维护自己的 KV 缓存。
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)  # (batch_size, seq_len, vocab_size)
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        """重置模型所有层的 KV 缓存以及位置指针，通常在开始生成新序列之前调用。"""
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用贪心解码（greedy decoding）自回归生成文本，可选择是否启用 KV 缓存加速。

    参数：
        model (GPTModel): 已训练（或随机初始化）的 GPT 模型。
        idx (Tensor): 起始 token 序列（prompt），形状 (batch_size, seq_len)。
        max_new_tokens (int): 要新生成的 token 数量。
        context_size (int, optional): 模型支持的最大上下文长度，超出部分会被截断；
            默认使用模型位置嵌入表的大小。
        use_cache (bool): 是否启用 KV 缓存。
            - True：只在第一次前向传播时喂入完整 prompt 建立缓存，
              之后每步只喂入新生成的 1 个 token，速度更快。
            - False：每一步都把「最近 context_size 个 token」整体重新喂入模型，
              不复用任何缓存，速度较慢但实现最简单、最容易验证正确性。

    返回：
        Tensor: 拼接了原始 prompt 和新生成 token 的完整序列，
            形状 (batch_size, seq_len + max_new_tokens)。
    """
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            # 中文：先清空历史缓存，再用完整的 prompt 做一次前向传播，
            # 这一步会把 prompt 中每个 token 的 K/V 都写入缓存。
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                # 贪心采样：直接选概率（logit）最大的下一个 token，不做随机采样
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                # 中文：因为 K/V 已经缓存，这里只需要把新生成的这 1 个 token 喂给模型，
                # 模型内部会自动把它的 K/V 拼接到缓存后面，避免重复计算历史 token 的注意力。
                logits = model(next_idx, use_cache=True)
        else:
            # 不使用缓存：每一步都重新喂入「最近 ctx_len 个 token」的完整序列，
            # 模型需要重新计算所有历史 token 的 K/V 和注意力，计算量随生成长度增长而增长。
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    """命令行入口：解析参数、构建带 GQA 的 GPT 模型，并跑一次文本生成用于演示和简单的性能测量。"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run GPT with grouped-query attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--n_kv_groups", type=int, default=2, help="Number of key/value groups.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")  # 使用 GPT-2 的 BPE 分词器
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),  # 上下文长度需要能容纳 prompt + 新生成的所有 token
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate  # 推理阶段不需要 dropout，这里直接设为 0
        "qkv_bias": False,          # Query-Key-Value bias
        "n_kv_groups": args.n_kv_groups  # GQA 的 KV 分组数
    }
    torch.manual_seed(123)  # 固定随机种子，保证模型初始化权重可复现
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)  # 使用 bfloat16 精度以节省显存、加快计算（若硬件支持）
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)  # 增加 batch 维度 -> (1, seq_len)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 同步 GPU，确保之前的操作都已完成，计时更准确
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 生成结束后再次同步，保证计时包含所有 GPU 计算
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
