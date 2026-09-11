# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4, adapted to use DeepSeek Sparse Attention (DSA).
# This file can be run as a standalone script.

# DSA is introduced in DeepSeek-V3.2:
#   https://huggingface.co/deepseek-ai/DeepSeek-V3.2
# Technical report:
#   https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf

"""
【中文说明】

本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书中
第 3-4 章内容的汇总实现，并在此基础上扩展了两项内容：

1. KV 缓存 (KV Cache)：在自回归生成（一个 token 一个 token 地往后预测）时，
   避免每一步都把已经算过的历史 token 重新跑一遍注意力，而是把历史的 Key/Value
   张量缓存下来，新来的 token 只需要和缓存拼接后计算注意力，从而大幅提升推理速度。

2. DeepSeek 稀疏注意力 (DeepSeek Sparse Attention, DSA)：这是 DeepSeek-V3.2 中
   提出的一种注意力机制，核心思想是——对于每一个 query token，并不需要和历史上
   所有的 token 都做注意力计算，而是先用一个轻量级的“打分器”(Lightning Indexer)
   给每个历史 token 打分，然后只保留分数最高的 top-K 个 token 参与真正的注意力
   计算，其余全部掩码掉（设为 -inf）。这样可以将注意力的计算复杂度从 O(L^2)
   降低到理论上的 O(L*K)（本文件为了教学目的，仍然先算出完整的 (T,S) 注意力
   矩阵再做掩码，并没有实现真正省算力的稀疏 kernel，性能收益主要体现在“可读性
   等价于 DSA 的选择逻辑”而不是速度上）。

   DSA 由两部分组成：
     a) Lightning Indexer（轻量索引器）：用一个很小的头数/头维度的注意力打分，
        为每个 (query, 历史 key) 对打一个分数。
     b) Token Selector（token 选择器）：对每个 query，选出分数最高的 K 个历史
        token，其余全部丢弃（掩码为 -inf，不参与 softmax）。

   本文件把这套逻辑套用到标准的多头自注意力上：先算出稠密的注意力分数矩阵，
   再叠加“因果掩码 + 稀疏掩码”后做 softmax，效果上等价于“只对 top-K 个位置
   做注意力”，但实现上仍然是稠密矩阵运算，便于教学阅读。

   在本书的知识体系中，这个文件属于“进阶/前沿注意力变体”的教学示例，帮助读者
   在已经掌握了第 3-4 章标准多头注意力、Transformer Block、GPT 整体结构之后，
   理解工业界最新的稀疏注意力机制是如何工作的，以及如何与 KV 缓存结合来加速
   自回归推理。
"""

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# DeepSeek Sparse Attention (DSA)
#####################################
# DSA combines two components:
#   1. A Lightning Indexer that scores all past tokens for each query
#      using a lightweight gated sum of ReLU(q · k) dot products.
#   2. A Token Selector that picks the top-K highest-scoring past tokens
#      and masks out the rest.
#
# This teaching implementation applies a dense mask to standard attention.
# It reproduces the DSA selection logic but does not include a fused sparse
# attention kernel that would reduce attention compute from O(L^2) to O(L*k).
#
# Reference implementation inspired by:
#   https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py
# 【中文】DSA 由两部分组成：1）Lightning Indexer 轻量索引器，用 ReLU(q·k) 点积的
# 加权和给每个历史 token 打分；2）Token Selector，挑出分数最高的 top-K 个历史
# token，其余全部掩码丢弃。本教学实现是把稀疏选择的“结果”应用到稠密注意力矩阵上
# （先算全量分数，再掩码），没有实现能真正节省算力的稀疏 kernel，重点在于讲清楚
# DSA 的选择逻辑，而不是追求速度。


class LightningIndexer(nn.Module):
    """Lightweight module that scores every past token for each incoming query.

    For each query token t and each candidate past token s, the score is a
    gated sum of per-head dot products:
        I_{t,s} = sum_j [ (w_{t,j} / sqrt(H_I)) * ReLU((q_{t,j} · k_s) / sqrt(d_I)) ]

    where w_{t,j} is a learned per-head scalar weight derived from the input,
    H_I is the number of index heads, d_I is the index head dimension, and j
    indexes over the index heads.

    Args:
        d_model:       model dimension (same as emb_dim).
        index_n_heads: number of lightweight index heads (H_I in the paper).
        index_head_dim: dimension of each index head.

    【中文】轻量级“打分器”（Lightning Indexer）。

    作用：对于当前的每一个 query token，给“候选的历史 token”逐个打分，分数越高
    表示该历史 token 对当前 query 越重要，后续会依据这个分数挑出 top-K 个 token
    参与真正的注意力计算。

    打分公式（对每个 query t、每个历史 token s）：
        I_{t,s} = sum_j [ (w_{t,j} / sqrt(H_I)) * ReLU((q_{t,j} · k_s) / sqrt(d_I)) ]
    也就是：先对每个“索引头” j 分别计算 query 和 key 的点积（并用 ReLU 截断负值、
    用 sqrt(d_I) 做缩放），再乘以一个由输入学出来的“每头权重” w_{t,j}（并用
    sqrt(H_I) 做缩放），最后把所有索引头的结果加起来，得到一个标量分数。

    注意：这个索引器自己有一套独立的、很小的 Q/K 投影（index_n_heads 个头，
    每头维度 index_head_dim），跟外面主注意力的 Q/K/V 是完全分开、参数不共享的，
    目的是用很小的计算开销来做“筛选”，避免用完整的大头注意力去做这件事。

    参数说明：
        d_model:        模型的隐藏维度（等同于 emb_dim）。
        index_n_heads:  索引器使用的头数（论文中记为 H_I），通常远小于主注意力头数。
        index_head_dim: 索引器每个头的维度（论文中记为 d_I）。
    """

    def __init__(self, d_model: int, index_n_heads: int, index_head_dim: int):
        super().__init__()
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim

        # Project input to indexer query vectors: (d_model -> index_n_heads * index_head_dim)
        # 【中文】把输入 x 投影成索引器专用的 query 向量，之后会 reshape 成
        # (index_n_heads, index_head_dim) 多头形式；无 bias，与主注意力的 Q/K/V 完全独立。
        self.W_q_index = nn.Linear(d_model, index_n_heads * index_head_dim, bias=False)
        # Project input to shared key vectors: (d_model -> index_head_dim)
        # 【中文】索引器的 key 投影：注意这里只输出 index_head_dim 维（不区分多头），
        # 即所有索引头共享同一份 key，这是 DSA 设计中降低计算量的一个关键点。
        self.W_k_index = nn.Linear(d_model, index_head_dim, bias=False)
        # Learn a per-head weight scalar: (d_model -> index_n_heads), as in the V3.2 paper
        # 【中文】学习一个“每个索引头的门控权重”标量，用来在最后加权合并各索引头的打分结果，
        # 相当于让模型自己学习“该更信任哪个索引头的判断”。
        self.W_weights = nn.Linear(d_model, index_n_heads, bias=False)

        self.scale = index_head_dim ** -0.5  # 【中文】1/sqrt(d_I)，用于缩放点积，防止数值过大

    def forward(
        self,
        x: torch.Tensor,         # (b, T, d_model)  current token(s)
        x_ctx: torch.Tensor,     # (b, S, d_model)  all past + current tokens
        topk: int,
        causal_mask: torch.Tensor | None = None,  # (T, S) float mask
    ) -> torch.Tensor:
        """Return top-K token indices shape (b, T, topk).

        【中文】前向计算：给每个 query token 对所有候选历史 token 打分，
        然后返回分数最高的 top-K 个历史 token 的下标。

        参数：
            x:            当前这一批要计算注意力的 query token，形状 (b, T, d_model)。
                          T 通常是当前新增的 token 数（比如用 KV 缓存时，T 可能只是 1）。
            x_ctx:        全部候选的历史 token（含当前 token），形状 (b, S, d_model)，
                          S 是历史+当前的总长度（>= T）。
            topk:         每个 query 最多保留多少个历史 token 参与真正的注意力计算。
            causal_mask:  因果掩码，形状 (T, S)，未来位置为 -inf，用于打分阶段就排除
                          掉“看到未来”的可能性，避免 topk 选出因果上不可见的位置。

        返回：
            topk_indices: 形状 (b, T, topk)，每个 query 对应的 top-K 历史 token 下标。
        """
        b, T, _ = x.shape
        _, S, _ = x_ctx.shape

        # Indexer queries: (b, T, H_I, head_dim)
        # 【中文】把 x 投影后 reshape 成多头形式，得到索引器专用的 query： (b, T, H_I, head_dim)
        q = self.W_q_index(x).view(b, T, self.index_n_heads, self.index_head_dim)
        # Indexer keys: (b, S, head_dim)
        # 【中文】索引器的 key：所有索引头共享，形状仍是 (b, S, head_dim)，没有头维度
        k = self.W_k_index(x_ctx)  # (b, S, head_dim)

        # ReLU(q · k^T) for each head: (b, T, H_I, S)
        # k: (b, S, head_dim) -> (b, 1, S, head_dim) for broadcast
        # 【中文】用 einsum 一次性计算所有索引头、所有 query-key 对的点积：
        # "bthd,bsd->bths" 表示对最后一维 d(head_dim) 做内积求和，
        # 得到形状 (b, T, H_I, S)，即每个 batch、每个 query、每个索引头、
        # 对每个候选 key 位置都有一个点积分数，然后乘以 self.scale 做缩放。
        raw = torch.einsum("bthd,bsd->bths", q, k) * self.scale  # (b, T, H_I, S)
        raw = torch.relu(raw)  # 【中文】ReLU 截断负分数，只保留“正相关”的信号，是 DSA 打分公式的一部分

        # Per-head learned gates: (b, T, H_I)
        # Reference implementations use raw learned gates scaled by sqrt(H_I),
        # not a probability distribution over index heads.
        # 【中文】注意：这里的 gate 不是 softmax 归一化后的概率分布，而是直接学出来的
        # 原始标量权重（可正可负），只是除以 sqrt(H_I) 做了个缩放，这是参考官方实现的做法。
        w = self.W_weights(x)  # (b, T, H_I)
        w = w * (self.index_n_heads ** -0.5)

        # Weighted sum over heads -> index scores (b, T, S)
        # 【中文】用每头的门控权重 w 对 raw 在头维度 H_I 上做加权求和，
        # "bth,bths->bts" 表示消去 h 维度，得到最终的索引分数 (b, T, S)：
        # 每个 query 对每个候选历史 token 一个标量分数。
        index_scores = torch.einsum("bth,bths->bts", w, raw)  # (b, T, S)

        if causal_mask is not None:
            # 【中文】叠加因果掩码：未来位置的分数被加上 -inf，
            # 这样后面 topk 选择时绝不会选到“未来”的 token，保证因果性。
            index_scores = index_scores + causal_mask  # broadcast over batch

        # Select top-K positions. topk is capped at available context length S.
        # 【中文】topk 不能超过实际可用的历史长度 S（比如生成刚开始时历史很短），
        # 所以这里做了个 min 截断保护。
        k_val = min(topk, S)
        # 【中文】沿最后一维（候选 token 维度）取分数最高的 k_val 个下标，
        # 得到 (b, T, k) 的索引张量——这就是每个 query 应该关注的历史 token 位置。
        topk_indices = index_scores.topk(k_val, dim=-1).indices  # (b, T, k)
        return topk_indices


class MultiHeadAttentionWithDSA(nn.Module):
    """Multi-head causal self-attention with DeepSeek Sparse Attention (DSA).

    After computing full attention scores, the Lightning Indexer selects the
    top-K most relevant past tokens for each query, and all other positions
    are masked to -inf before softmax. This keeps the selected-token behavior
    of DSA while using a dense attention matrix for clarity.

    Args:
        d_in:           input dimension.
        d_out:          output dimension (must be divisible by num_heads).
        dropout:        dropout probability.
        num_heads:      number of standard attention heads.
        qkv_bias:       whether to use bias in Q/K/V projections.
        index_n_heads:  number of lightweight index heads (H_I).
        index_head_dim: dimension per index head.
        topk:           number of tokens each query attends to (k).

    【中文】带 DeepSeek 稀疏注意力 (DSA) 的多头因果自注意力模块。

    整体流程：
        1. 像标准多头注意力一样，先把输入投影成 Q/K/V，并支持 KV 缓存
           （把历史的 K/V 缓存下来，新 token 只需要计算自己的 K/V 并拼接）。
        2. 计算完整的 (T_q, T_k) 稠密注意力分数矩阵（这一步仍是 O(L^2)，
           是本教学实现为了代码清晰而做的简化，真正的稀疏 kernel 会跳过
           未被选中的位置以节省算力）。
        3. 用 LightningIndexer 对每个 query 选出 top-K 个最相关的历史 token。
        4. 把“因果掩码”和“稀疏掩码”叠加到注意力分数上（未被选中/未来的位置
           都设为 -inf），再做 softmax，效果上等价于“只对被选中的 K 个 token
           做注意力”。

    参数说明：
        d_in:           输入特征维度。
        d_out:          输出特征维度（必须能被 num_heads 整除）。
        dropout:        注意力权重的 dropout 概率。
        num_heads:      标准多头注意力的头数。
        qkv_bias:       Q/K/V 线性层是否使用偏置项。
        index_n_heads:  Lightning Indexer 使用的索引头数 H_I。
        index_head_dim: Lightning Indexer 每个索引头的维度。
        topk:           每个 query 最终保留参与注意力计算的历史 token 数 k。
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
        index_n_heads: int = 4,
        index_head_dim: int = 64,
        topk: int = 64,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 【中文】每个注意力头的维度 = 总输出维度 / 头数
        self.topk = topk

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 【中文】多头拼接后的输出投影层
        self.dropout = nn.Dropout(dropout)

        # 【中文】实例化前面定义的轻量索引器，注意它的输入维度是 d_in（即原始 x 的维度），
        # 而不是 d_out，因为索引器直接吃“上一层的输出 x”，不依赖主注意力的 Q/K/V 投影。
        self.indexer = LightningIndexer(d_in, index_n_heads, index_head_dim)

        ####################################################
        # KV cache-related code
        # 【中文】KV 缓存相关：用 register_buffer 注册（非持久化，不会被存进 state_dict），
        # 初始为 None，第一次前向时才会被真正赋值成张量。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        # Keep raw input tokens for the indexer key projection
        # 【中文】除了缓存 K/V，还要额外缓存原始输入 x（cache_x），
        # 因为 LightningIndexer 需要对“全部历史 token 的原始输入”重新做一次
        # W_k_index 投影来算索引 key，所以必须把历史的 x 也存下来，而不能只存 K/V。
        self.register_buffer("cache_x", None, persistent=False)
        self.ptr_current_pos = 0  # 【中文】记录当前已经处理到的绝对位置，用于构造因果掩码的位置编号
        ####################################################

    def reset_cache(self):
        """重置 KV 缓存（以及索引器用的历史输入缓存和位置指针）。
        通常在开始一段新的生成序列之前调用，避免用到上一次生成残留的缓存。"""
        self.cache_k = None
        self.cache_v = None
        self.cache_x = None
        self.ptr_current_pos = 0

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """前向传播：计算带 DSA 稀疏掩码的因果自注意力。

        参数：
            x:         输入张量，形状 (b, num_tokens, d_in)。
                      若 use_cache=True 且缓存已存在，这里的 num_tokens 通常只是
                      新增的 token 数（例如逐 token 生成时为 1）。
            use_cache: 是否启用 KV 缓存（推理/生成时开启可大幅加速）。

        返回：
            context_vec: 注意力输出，形状 (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys_new = self.W_key(x)
        values_new = self.W_value(x)

        # Reshape to (b, T, num_heads, head_dim)
        # 【中文】把最后一维 d_out 拆分成 (num_heads, head_dim)，为多头注意力做准备
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache-related
        # 【中文】KV 缓存核心逻辑：
        # - 如果还没有缓存（第一次调用，比如输入的是完整 prompt），直接把这一批的
        #   K/V/x 当作缓存的初始内容；
        # - 如果已经有缓存（比如后续逐 token 生成），就把新算出来的 K/V/x
        #   沿着序列长度维度 (dim=1) 拼接到旧缓存后面，实现“只算新 token、
        #   复用旧 token 的 K/V”，避免重复计算历史注意力。
        if use_cache:
            if self.cache_k is None:
                keys = keys_new
                values = values_new
                x_ctx = x
            else:
                keys = torch.cat([self.cache_k, keys_new], dim=1)
                values = torch.cat([self.cache_v, values_new], dim=1)
                x_ctx = torch.cat([self.cache_x, x], dim=1)
            self.cache_k = keys
            self.cache_v = values
            self.cache_x = x_ctx
            q_start = self.ptr_current_pos  # 【中文】当前这批 query 在整体序列中的起始绝对位置
            k_start = 0  # 【中文】key 的起始位置永远是 0（缓存里包含了从头开始的全部历史）
            self.ptr_current_pos += num_tokens  # 【中文】推进位置指针，供下一次调用使用
        else:
            # 【中文】不使用缓存时（比如训练阶段，或一次性喂入完整序列做前向），
            # 每次都是全新计算，没有历史可拼接。
            keys = keys_new
            values = values_new
            x_ctx = x
            q_start = 0
            k_start = 0
        ####################################################

        # Transpose: (b, T, num_heads, head_dim) -> (b, num_heads, T, head_dim)
        # 【中文】把头维度换到序列维度前面，方便下面对每个头独立做批量矩阵乘法
        queries_t = queries.transpose(1, 2)
        keys_t = keys.transpose(1, 2)
        values_t = values.transpose(1, 2)

        # Full scaled dot-product attention scores: (b, num_heads, T_q, T_k)
        # 【中文】计算标准的缩放点积注意力分数（这里先不除以 sqrt(head_dim)，
        # 缩放放在后面 softmax 之前统一做）：Q @ K^T，得到每个 query 对每个 key
        # 的原始相似度分数，形状 (b, num_heads, T_q, T_k)。
        # 注意：这一步仍然是稠密的 O(T_q * T_k) 计算，DSA 的“省算力”体现在真正的
        # 稀疏 kernel 实现里会跳过被掩码的位置，本教学版本只是事后掩码。
        attn_scores = queries_t @ keys_t.transpose(2, 3)

        num_tokens_Q = queries_t.shape[-2]
        num_tokens_K = keys_t.shape[-2]
        device = x.device

        # ---- Build causal mask (float, -inf for masked positions) ----
        # 【中文】构造因果掩码：用绝对位置编号比较，凡是 query 位置 < key 位置
        # （即 key 在 query“未来”），就要被掩掉，保证模型看不到未来的 token。
        # 这里用绝对位置（加上 q_start/k_start 偏移）而不是相对下标，是因为
        # 启用 KV 缓存时，当前这批 query 的下标并不是从 0 开始的。
        q_positions = torch.arange(q_start, q_start + num_tokens_Q, device=device, dtype=torch.long)
        k_positions = torch.arange(k_start, k_start + num_tokens_K, device=device, dtype=torch.long)
        causal_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)  # (T_q, T_k)  # 【中文】True 表示该位置需要被掩码（未来）
        causal_float = torch.zeros(num_tokens_Q, num_tokens_K, device=device, dtype=attn_scores.dtype)
        causal_float.masked_fill_(causal_bool, float("-inf"))  # 【中文】把未来位置填成 -inf，softmax 后权重趋近于 0

        # ---- DSA: Lightning Indexer → sparse mask ----
        # The indexer receives the current queries (x) and all context tokens (x_ctx).
        # causal_float is passed so future tokens are excluded from index selection.
        # 【中文】调用轻量索引器，对每个 query 从全部历史 token 中选出 top-K 个最相关的，
        # 并且把因果掩码传进去，确保 topk 选择阶段也不会选中“未来”的 token
        # （即使某个未来 token 打分很高，也因为加了 -inf 而必然被排除）。
        topk_indices = self.indexer(x, x_ctx, self.topk, causal_mask=causal_float)
        # topk_indices: (b, T_q, k)

        # Build sparse mask: -inf everywhere, 0 at selected positions
        # 【中文】构造稀疏掩码：默认所有位置都是 -inf（不可见），
        # 然后用 scatter_ 把 topk_indices 指定的位置改写为 0（可见），
        # 这样最终只有被 Lightning Indexer 选中的 top-K 个历史 token 才会
        # 真正参与到下面的 softmax 注意力计算中。
        sparse_mask = torch.full(
            (b, num_tokens_Q, num_tokens_K), float("-inf"), device=device, dtype=attn_scores.dtype
        )
        sparse_mask.scatter_(-1, topk_indices, 0.0)  # (b, T_q, T_k)

        # Combine causal mask and sparse mask, then broadcast over heads
        # 【中文】把因果掩码（防止看到未来）和稀疏掩码（只看 top-K 个被选中的历史 token）
        # 相加合并（两者都是 0 或 -inf，相加后只要有一个是 -inf 结果就是 -inf，
        # 相当于逻辑“与”：必须“既不是未来、又被选中”才可见），
        # 再在头维度上广播（unsqueeze(1)），加到每个头的注意力分数上。
        combined_mask = causal_float.unsqueeze(0) + sparse_mask  # (b, T_q, T_k)
        attn_scores = attn_scores + combined_mask.unsqueeze(1)  # (b, num_heads, T_q, T_k)

        # 【中文】除以 sqrt(head_dim) 做缩放（标准 Scaled Dot-Product Attention 的“Scaled”部分，
        # 防止点积数值过大导致 softmax 梯度消失），再做 softmax 得到注意力权重分布。
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 【中文】对注意力权重做 dropout 正则化

        # Shape: (b, num_heads, T_q, head_dim)
        # 【中文】用注意力权重对 V 做加权求和，得到每个 query 位置的上下文向量
        context_vec = attn_weights @ values_t
        # Transpose and reshape: (b, T_q, d_out)
        # 【中文】把多头维度换回去并拼接（contiguous 保证内存连续后才能安全 view），
        # 还原成 (b, T_q, d_out) 的形状
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 【中文】最后做一次线性投影，融合各头信息
        return context_vec


#####################################
# Chapter 4
#####################################
# 【中文】以下是第 4 章介绍过的标准组件：LayerNorm、GELU 激活函数、前馈网络 FeedForward、
# Transformer Block、以及完整的 GPTModel，这里的实现和第 4 章基本一致，
# 只是把注意力模块换成了上面带 DSA 的 MultiHeadAttentionWithDSA，并加入了 KV 缓存支持。
class LayerNorm(nn.Module):
    """层归一化 (Layer Normalization)。

    对最后一维（特征维度）做归一化：减去均值、除以标准差，再用可学习的
    缩放参数 scale 和偏移参数 shift 做仿射变换，帮助稳定训练、加速收敛。

    参数：
        emb_dim: 特征维度大小，scale/shift 都是这个维度的可学习向量。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 【中文】防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 【中文】可学习缩放系数 γ，初始化为全 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 【中文】可学习偏移量 β，初始化为全 0

    def forward(self, x):
        """输入 x 形状 (..., emb_dim)，对最后一维做归一化，输出形状不变。"""
        mean = x.mean(dim=-1, keepdim=True)  # 【中文】沿特征维度求均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 【中文】沿特征维度求方差（有偏估计，与论文实现一致）
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 【中文】标准化：减均值除以标准差
        return self.scale * norm_x + self.shift  # 【中文】仿射变换：可学习地缩放和平移


class GELU(nn.Module):
    """GELU 激活函数的 tanh 近似实现（与 GPT-2 论文中使用的近似公式一致）。"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """输入输出形状相同，逐元素应用 GELU 激活。"""
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer 中的逐位置前馈网络 (Position-wise Feed-Forward Network)。

    结构：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)，
    先升维再降维，中间隐藏层维度是输入的 4 倍（GPT 系列的常见设计）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 【中文】升维到 4 倍，增加模型容量
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 【中文】再降回原始维度，便于残差相加
        )

    def forward(self, x):
        """输入输出形状均为 (b, num_tokens, emb_dim)。"""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """一个完整的 Transformer 块：多头（DSA）自注意力子层 + 前馈网络子层，
    每个子层都配有前置 LayerNorm（Pre-LN 结构）、残差连接（shortcut）和 dropout。
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionWithDSA(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            index_n_heads=cfg["index_n_heads"],
            index_head_dim=cfg["index_head_dim"],
            topk=cfg["topk"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 【中文】注意力子层之前的归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 【中文】前馈子层之前的归一化
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        """输入 x 形状 (b, num_tokens, emb_dim)，输出形状相同。

        use_cache: 是否启用 KV 缓存，透传给内部的注意力模块。
        """
        # Shortcut connection for attention block
        # 【中文】Pre-LN 结构：先归一化，再进注意力，最后与归一化前的原始输入做残差相加
        shortcut = x
        x = self.norm1(x)

        ####################################################
        # KV cache-related
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back  # 【中文】残差连接，缓解深层网络梯度消失问题

        # Shortcut connection for feed-forward block
        # 【中文】前馈子层同样是 Pre-LN + 残差的结构
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的 GPT 模型：token 嵌入 + 位置嵌入 -> N 个 TransformerBlock -> 最终归一化
    -> 输出投影得到词表 logits。支持 KV 缓存以加速自回归生成。

    参数 cfg（字典）需要包含：
        vocab_size:     词表大小
        context_length: 最大上下文长度（位置嵌入表的大小）
        emb_dim:        嵌入/隐藏维度
        n_heads:        注意力头数
        n_layers:       Transformer 块的层数
        drop_rate:      dropout 概率
        qkv_bias:       Q/K/V 投影是否使用 bias
        index_n_heads/index_head_dim/topk: 传给 DSA 注意力模块的相关超参数
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])       # 【中文】词元(token)嵌入表
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 【中文】可学习的绝对位置嵌入表
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        ####################################################
        # KV cache-related
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0  # 【中文】记录当前已经生成/处理到的绝对位置，供位置嵌入和各层 KV 缓存对齐使用
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 【中文】把隐藏状态映射回词表大小的 logits

    def forward(self, in_idx, use_cache=False):
        """前向传播。

        参数：
            in_idx:    输入 token id，形状 (batch_size, seq_len)。
            use_cache: 是否启用 KV 缓存（生成时开启以加速）。

        返回：
            logits: 形状 (batch_size, seq_len, vocab_size)，每个位置对下一个 token 的预测分数。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 【中文】(batch_size, seq_len, emb_dim)

        ####################################################
        # KV cache-related
        # 【中文】位置编码的位置 id 需要考虑是否使用缓存：
        # - 若使用缓存，位置要从上次结束的 current_pos 继续往后编号（因为这批 token
        #   是接着之前已经处理过的序列往后延伸的，比如新生成的 1 个 token）；
        # - 若不使用缓存，说明是从头开始的一次性完整前向，位置从 0 开始编号。
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len  # 【中文】推进全局位置指针
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)  # 【中文】(1, seq_len, emb_dim)，广播加到每个 batch 样本上
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]  # 【中文】词嵌入 + 位置嵌入
        x = self.drop_emb(x)

        ####################################################
        # KV cache-related
        # 【中文】依次通过每一层 TransformerBlock，use_cache 透传给每层的注意力模块，
        # 各层各自维护自己独立的 KV 缓存（缓存挂在每层的 self.att 上，见 TransformerBlock.att）。
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)  # 【中文】(batch_size, seq_len, vocab_size)
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        """重置所有层的 KV 缓存以及全局位置指针，通常在开始处理一个全新的
        输入序列（比如新的一轮生成）之前调用，避免残留上一次生成的缓存。"""
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用（可选）KV 缓存的贪心自回归文本生成函数。

    参数：
        model:          GPTModel 实例。
        idx:            初始 token id 序列，形状 (batch_size, seq_len)。
        max_new_tokens: 要新生成的 token 数量。
        context_size:   截断的上下文窗口大小；默认使用模型位置嵌入表的容量。
        use_cache:      是否启用 KV 缓存加速生成。

    返回：
        idx: 拼接了新生成 token 之后的完整序列，形状 (batch_size, seq_len + max_new_tokens)。
    """
    model.eval()  # 【中文】切换到评估模式，关闭 dropout 等训练专用行为
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():  # 【中文】生成阶段不需要梯度，节省显存和计算
        if use_cache:
            # Init cache with full prompt
            # 【中文】先重置缓存，然后把完整的 prompt（截断到 ctx_len 以内）一次性喂进去，
            # 这一步会把 prompt 中每个 token 的 K/V 都算好并存入缓存。
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                # 【中文】贪心采样：直接取概率（logits）最大的那个 token，不做随机采样
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                # 【中文】把新生成的 token 拼接到序列末尾
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                # 【中文】关键优化点：这里只把“刚生成的这一个新 token”喂给模型，
                # 而不是把整个序列重新算一遍——因为历史 token 的 K/V 已经缓存好了，
                # 新 token 的注意力只需要和缓存拼接即可，这就是 KV 缓存带来的加速。
                logits = model(next_idx, use_cache=True)
        else:
            # 【中文】不使用缓存的朴素版本：每一步都把（截断后的）完整序列重新喂给模型，
            # 计算量随生成长度增长是重复且低效的，仅作对比参考。
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    """命令行入口：解析参数、构建一个小型 GPT + DSA 模型、生成一段文本并打印耗时/吞吐/显存统计。"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run GPT with DeepSeek Sparse Attention (DSA)."
    )
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--index_n_heads", type=int, default=4,
                        help="Number of lightweight indexer heads (H_I in the DSA paper).")
    parser.add_argument("--index_head_dim", type=int, default=64,
                        help="Dimension of each indexer head.")
    parser.add_argument("--topk", type=int, default=64,
                        help="Number of tokens each query attends to (k). "
                             "For short sequences this is capped at sequence length.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")  # 【中文】使用 GPT-2 的 BPE 分词器
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),  # 【中文】上下文长度设为“待生成 token 数 + prompt 长度”，刚好够用
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate  # 【中文】推理时不需要 dropout，设为 0
        "qkv_bias": False,          # Query-Key-Value bias
        "index_n_heads": args.index_n_heads,
        "index_head_dim": args.index_head_dim,
        "topk": args.topk,
    }

    torch.manual_seed(123)  # 【中文】固定随机种子，保证参数初始化可复现
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)  # 【中文】使用 bfloat16 精度以节省显存、加快计算（若硬件支持）
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)  # 【中文】增加 batch 维度，形状变为 (1, seq_len)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 【中文】同步 CUDA 流，确保计时准确（GPU 运算是异步的）
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
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")  # 【中文】每秒生成的 token 数，衡量推理吞吐量
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")  # 【中文】峰值显存占用，便于评估 KV 缓存等机制的显存开销


if __name__ == "__main__":
    main()
