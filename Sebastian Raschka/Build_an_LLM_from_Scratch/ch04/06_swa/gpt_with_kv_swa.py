# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

# ============================================================
# 中文说明（模块级 docstring）
# ============================================================
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
# 第 3-4 章代码的一个"加强版"整合脚本，展示了如何在一个最基础的 GPT 架构上
# 叠加两项工程优化：
#   1. KV 缓存（KV Cache）：在自回归生成（逐 token 解码）时，把历史 token
#      的 Key/Value 张量缓存下来，避免每生成一个新 token 就要把整个序列重新
#      跑一遍注意力计算，从而大幅提升推理速度。
#   2. 滑动窗口注意力（Sliding Window Attention, SWA）：不同于标准的"因果
#      注意力"（每个 query 可以看到它之前的所有 token），SWA 只允许每个
#      query 关注最近 W 个 token（一个滑动窗口内），这样可以把注意力的时间/
#      显存开销从 O(seq_len^2) 降到 O(seq_len * W)，非常适合长序列场景
#      （例如 Mistral 等模型就采用了类似机制）。
#
# 文件结构对应书中章节：
#   - "Chapter 3" 部分：多头自注意力（Multi-Head Attention），这里扩展为
#     支持 KV 缓存 + 滑动窗口的版本 MultiHeadAttentionWithSWA。
#   - "Chapter 4" 部分：LayerNorm、GELU 激活、FeedForward 前馈网络、
#     TransformerBlock（一个完整的 Transformer 层）、GPTModel（把词嵌入、
#     位置嵌入、多层 TransformerBlock 以及输出头组装成完整的 GPT 模型）。
#   - 脚本末尾提供了一个使用 KV 缓存做贪心解码的生成函数
#     generate_text_simple_cached，以及一个可以从命令行运行的 main()
#     函数，用于测量生成速度和显存占用。
#
# 阅读建议：先理解标准的因果自注意力和标准 GPT 结构（对应更早章节的代码），
# 再来看本文件，会更容易抓住"KV 缓存"和"滑动窗口"这两个增量改动点。
# ============================================================

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttentionWithSWA(nn.Module):
    """多头自注意力模块（支持 KV 缓存 + 滑动窗口注意力）。

    这是标准多头自注意力（Multi-Head Self-Attention）的增强版本，在标准实现
    的基础上新增了两项能力：
      1. KV 缓存：当 use_cache=True 时，把历史时间步计算出的 Key/Value
         缓存在 self.cache_k / self.cache_v 中，新的前向传播只需要计算当前
         新增 token 的 Key/Value，并与缓存拼接，从而避免重复计算历史 token
         的注意力投影，显著加速自回归生成。
      2. 滑动窗口注意力（SWA）：通过 sliding_window_size 参数限制每个 query
         只能看到最近 W 个 key（而不是从序列开头到当前位置的所有 key），
         从而把长序列的注意力计算/显存开销控制在 O(seq_len * W) 量级。

    参数：
        d_in (int): 输入特征维度（即上一层的 embedding 维度）。
        d_out (int): 输出特征维度，同时也是 Q/K/V 投影后的总维度，
            要求能被 num_heads 整除。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool): Q/K/V 的线性层是否使用偏置项，默认为 False。
        sliding_window_size (int or None): 滑动窗口宽度 W。若为 None，
            则退化为标准的全因果注意力（可以看到之前所有 token）。
    """

    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, sliding_window_size=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头分到的维度 = 总输出维度 / 头数，
        # 例如 d_out=768, num_heads=12 时，每个头的维度 head_dim=64。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.sliding_window_size = sliding_window_size

        ####################################################
        # KV cache-related code
        # 中文：cache_k / cache_v 用于保存历史 token 的 Key/Value 张量，
        # 注册为 buffer（而不是 Parameter）是因为它们不需要梯度、也不参与
        # 反向传播，只是推理阶段的中间状态；persistent=False 表示它们不会
        # 被保存进 state_dict（因为这是运行时缓存，不是模型权重）。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0  # 记录当前已经处理到序列的绝对位置（用于计算位置索引）
        ####################################################

    def forward(self, x, use_cache=False):
        """前向传播：计算带因果掩码（可选滑动窗口）的多头自注意力。

        参数：
            x (Tensor): 输入张量，形状为 (batch, num_tokens, d_in)。
                在使用 KV 缓存做增量解码时，num_tokens 通常只是新增的
                token 数量（例如每步只有 1 个新 token）。
            use_cache (bool): 是否启用 KV 缓存模式。True 表示本次调用
                是在增量生成过程中，需要把历史缓存的 K/V 与新计算的 K/V
                拼接；False 表示普通的一次性前向（例如训练，或不使用缓存
                的推理）。

        返回：
            Tensor: 注意力输出，形状为 (batch, num_tokens, d_out)，
                与输入的 num_tokens 维度一致（只返回新增 token 对应的输出）。
        """
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)
        # 中文：注意这里只对"新输入"x 做 Q/K/V 投影，K/V 会在下面与历史缓存拼接；
        # 而 Q 只需要新 token 的（因为历史 token 的输出在之前的调用中已经算过了）。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆成 (num_heads, head_dim)，
        # 这样就能让每个头独立地在自己的子空间里计算注意力。
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache-related
        if use_cache:
            old_cache_k, old_cache_v = self.cache_k, self.cache_v
            old_len = 0 if old_cache_k is None else old_cache_k.size(1)
            if old_cache_k is None:
                # 中文：第一次调用（例如处理 prompt 的初始前向），缓存为空，
                # 直接把新计算的 K/V 当作全部的 K/V。
                combined_k, combined_v = keys_new, values_new
            else:
                # 中文：把历史缓存的 K/V 和新计算的 K/V 在序列维度（dim=1）拼接，
                # 得到"迄今为止全部 token"的 K/V。
                combined_k = torch.cat([old_cache_k, keys_new], dim=1)
                combined_v = torch.cat([old_cache_v, values_new], dim=1)

            keys, values = combined_k, combined_v
            if self.sliding_window_size is not None:
                # During chunked prefill we need up to W-1 older keys plus the whole
                # current chunk (so the earliest queries in the chunk keep their full
                # sliding-window context)
                # 中文：滑动窗口注意力下，参与本次注意力计算的 K/V 不需要全部历史，
                # 只需要保留窗口大小 W 相关的那一部分。这里之所以是
                # `sliding_window_size + num_tokens - 1` 而不是简单的
                # `sliding_window_size`，是因为分块预填充（chunked prefill）时
                # 一个 chunk 内最早的那个 query 也需要向前看满 W-1 个历史 token
                # 才能拥有完整的窗口上下文，所以要多保留 (num_tokens - 1) 个。
                attn_keep = min(keys.size(1), self.sliding_window_size + num_tokens - 1)
                keys = keys[:, -attn_keep:, :, :]
                values = values[:, -attn_keep:, :, :]

                # 中文：而缓存本身只需要保留最近 sliding_window_size 个 token 的 K/V
                # 即可（因为再往前的 token 不可能再被任何未来的 query 看到），
                # 这样可以让缓存本身也保持有限大小，不随序列长度无限增长，
                # 这正是 SWA 相比全量 KV 缓存节省显存的关键所在。
                cache_keep = min(combined_k.size(1), self.sliding_window_size)
                self.cache_k = combined_k[:, -cache_keep:, :, :]
                self.cache_v = combined_v[:, -cache_keep:, :, :]
            else:
                # 中文：非滑动窗口（标准因果注意力）情况下，缓存需要保留全部历史 K/V。
                self.cache_k, self.cache_v = combined_k, combined_v

            # 中文：dropped 表示因为窗口裁剪，本次注意力计算相比完整历史丢弃了多少个
            # 最老的 K/V；用它来推算参与注意力计算的 key 的"绝对起始位置"，
            # 这样才能在下面构造正确的位置索引（用于因果 + 窗口掩码判断）。
            dropped = combined_k.size(1) - keys.size(1)
            k_start_pos_abs = (self.ptr_current_pos - old_len) + dropped
            q_start_pos_abs = self.ptr_current_pos
        else:
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度换到 batch 之后，这样可以把 num_heads 也当作一个
        # "批量"维度，用一次矩阵乘法并行算出所有头的注意力分数。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文：queries 形状 (b, num_heads, num_tokens_Q, head_dim)，
        # keys.transpose(2,3) 形状 (b, num_heads, head_dim, num_tokens_K)，
        # 相乘后 attn_scores 形状为 (b, num_heads, num_tokens_Q, num_tokens_K)，
        # 即每个 query 对每个 key 的原始点积相似度分数。

        ####################################################
        # causal + sliding-window mask
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        # Determine absolute positions for q and k
        # 中文：因为使用 KV 缓存后，本次前向里的 query/key 张量的"局部下标"
        # 并不等于它们在整个序列中的"绝对位置"（例如只处理第 5 个 token 时，
        # 局部下标是 0，但绝对位置是 4），所以需要单独维护绝对位置，
        # 才能正确判断因果关系（谁在谁之前）和滑动窗口范围。
        if use_cache:
            q_start = q_start_pos_abs
            k_start = k_start_pos_abs
        else:
            q_start = 0
            k_start = 0
        q_positions = torch.arange(q_start, q_start + num_tokens_Q, device=device, dtype=torch.long)
        k_positions = torch.arange(k_start, k_start + num_tokens_K, device=device, dtype=torch.long)
        # Sliding window width
        # 中文：若没有设置滑动窗口，则窗口宽度设为 num_tokens_K + 1，
        # 这个值大于任何可能的 diff，等价于"不限制窗口"，从而退化为标准因果注意力。
        W = num_tokens_K + 1 if self.sliding_window_size is None else int(self.sliding_window_size)
        # 中文：diff[i, j] = 第 i 个 query 的绝对位置 - 第 j 个 key 的绝对位置。
        # diff >= 0 表示该 key 不晚于该 query（满足因果性，可以被看到）；
        # diff < W 表示该 key 落在最近 W 个 token 的窗口内。
        diff = q_positions.unsqueeze(-1) - k_positions.unsqueeze(0)
        # 中文：需要被屏蔽（设为 -inf）的位置：diff<0（key 在 query 之后，违反因果性）
        # 或者 diff>=W（key 太老，超出了滑动窗口范围）。
        mask_bool = (diff < 0) | (diff >= W)
        if use_cache:
            # 中文：更新指针，指向下一次调用时新 token 的起始绝对位置。
            self.ptr_current_pos += num_tokens_Q
        else:
            self.ptr_current_pos = 0

        # Use the mask to fill attention scores
        # 中文：把被屏蔽位置的注意力分数设为负无穷，这样经过 softmax 后
        # 这些位置的注意力权重会变成 0，即该 query 完全看不到该 key。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 中文：除以 sqrt(head_dim) 做缩放（"scaled" dot-product attention），
        # 防止点积数值过大导致 softmax 梯度消失。
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 中文：attn_weights (b, num_heads, num_tokens_Q, num_tokens_K) 与
        # values (b, num_heads, num_tokens_K, head_dim) 相乘，得到
        # (b, num_heads, num_tokens_Q, head_dim)，再转置回
        # (b, num_tokens_Q, num_heads, head_dim)，即每个 token、每个头的
        # 加权上下文向量。

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 中文：把 num_heads 和 head_dim 两个维度合并回 d_out，
        # 即把多个头的输出重新拼接成一个大向量。
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：最后再经过一个线性层做"头间信息融合"的输出投影。

        return context_vec

    def reset_cache(self):
        """清空 KV 缓存与位置指针。

        在开始一段新的生成（新的 prompt）之前调用，避免把上一次生成残留的
        历史 Key/Value 或位置信息错误地带入本次计算。
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）模块。

    对每个 token 的特征向量（最后一维）做零均值、单位方差的归一化，
    再用可学习的缩放（scale）和平移（shift）参数进行仿射变换，
    从而稳定深层网络的训练。
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除以 0 的极小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习的缩放参数 gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的平移参数 beta

    def forward(self, x):
        """对输入张量最后一维做归一化。

        参数：
            x (Tensor): 形状为 (..., emb_dim) 的张量，通常是
                (batch, num_tokens, emb_dim)。

        返回：
            Tensor: 与输入形状相同的归一化后张量。
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 中文：unbiased=False 表示使用有偏方差估计（除以 N 而不是 N-1），
        # 与常见深度学习框架中 LayerNorm 的默认实现保持一致。
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数（这里使用的是其 tanh 近似公式）。

    GELU（Gaussian Error Linear Unit）比 ReLU 更平滑，在 Transformer 类
    模型的前馈网络中被广泛使用。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 中文：GELU 的 tanh 近似公式：
        # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        # 这是 GPT-2 等模型中常用的近似实现，避免直接计算高斯误差函数 erf。
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer 中的前馈网络（Position-wise Feed-Forward Network）。

    结构为：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)，
    即先把维度放大 4 倍再压缩回来，中间加非线性激活，用于对每个位置的
    表示做独立的非线性变换（不同位置之间不交互，交互由注意力层完成）。
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
            x (Tensor): 形状为 (batch, num_tokens, emb_dim)。

        返回：
            Tensor: 形状为 (batch, num_tokens, emb_dim)，与输入相同。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """一个完整的 Transformer 块（层）。

    结构为：
        x -> LayerNorm -> 多头自注意力(带 KV 缓存/SWA) -> Dropout -> 残差相加
          -> LayerNorm -> 前馈网络 -> Dropout -> 残差相加
    即经典的 Pre-LN Transformer 结构，两个子层（注意力、前馈）都配有
    残差连接（shortcut）和层归一化。
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionWithSWA(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            sliding_window_size=cfg["sliding_window_size"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        """前向传播。

        参数：
            x (Tensor): 输入张量，形状为 (batch, num_tokens, emb_dim)。
            use_cache (bool): 是否启用 KV 缓存（透传给内部的注意力模块）。

        返回：
            Tensor: 输出张量，形状为 (batch, num_tokens, emb_dim)，
                与输入形状一致（Transformer 块不改变张量形状）。
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        x = self.att(x, use_cache=use_cache)  # 中文：把 use_cache 透传给注意力层，控制是否走缓存路径
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差连接——把注意力子层的输出加回归一化前的原始输入，
        # 有助于梯度传播、缓解深层网络训练困难。

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：同样地，前馈子层也使用残差连接。

        return x


class GPTModel(nn.Module):
    """完整的 GPT 模型（支持 KV 缓存与滑动窗口注意力）。

    整体结构：
        token embedding + position embedding -> Dropout
        -> N 层 TransformerBlock（其中部分层可配置为滑动窗口注意力）
        -> 最终 LayerNorm -> 输出线性层（映射到词表大小的 logits）

    与标准 GPT 实现的关键差异：
      - 每个 TransformerBlock 内部的注意力层可以是"全局注意力"或
        "滑动窗口注意力"，具体由 sliding_window_stride（K:1 调度）决定；
      - 支持 KV 缓存的增量前向，配合 current_pos 记录当前生成到的绝对位置，
        以便正确计算位置编码和注意力掩码。
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])   # token 嵌入表
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 可学习的绝对位置嵌入表
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache-related
        # 中文：这里没有直接用 nn.Sequential 堆叠完全相同的层，而是逐层构建，
        # 因为需要按照 K:1 的调度策略，给不同层分别指定是否启用滑动窗口注意力。
        blocks = []
        window_stride = cfg["sliding_window_stride"]
        window_size = cfg["sliding_window_size"] if "sliding_window_size" in cfg else None
        for i in range(cfg["n_layers"]):
            blk = TransformerBlock(cfg)
            # K:1 schedule meaning that K SWA layers are followed by 1 regular layer
            # 中文：K:1 调度——每 (K+1) 层为一组，其中前 K 层使用滑动窗口注意力（SWA），
            # 第 K+1 层使用普通的全局因果注意力。这种"周期性插入全局注意力层"的做法
            # 类似 Mistral 等模型的设计，既能享受 SWA 节省显存/计算的好处，
            # 又能通过少量全局层让信息能传播到更远的历史 token。
            K = int(window_stride)
            if K <= 0:
                # 0 => all regular; negative => all SWA
                # 中文：K=0 表示完全不用滑动窗口（全部为标准全局注意力层）；
                # K<0 表示全部层都使用滑动窗口注意力。
                use_swa = False if K == 0 else True
            else:
                group = K + 1
                use_swa = (i % group) < K
            blk.att.sliding_window_size = window_size if use_swa else None
            blocks.append(blk)
        self.trf_blocks = nn.ModuleList(blocks)

        self.current_pos = 0  # 中文：记录整个模型层面上，当前已生成/处理到的绝对 token 位置
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 中文：输出头把最后的隐藏状态映射回词表大小的 logits，用于预测下一个 token。

    def forward(self, in_idx, use_cache=False):
        """前向传播。

        参数：
            in_idx (Tensor): 输入 token id 序列，形状为 (batch_size, seq_len)。
                当 use_cache=True 且处于增量解码阶段时，seq_len 通常等于
                新增 token 的数量（例如每步为 1）。
            use_cache (bool): 是否启用 KV 缓存的增量前向模式。

        返回：
            Tensor: logits，形状为 (batch_size, seq_len, vocab_size)，
                表示对应位置上每个词表 token 的预测分数（未归一化）。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # (batch_size, seq_len, emb_dim)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        if use_cache:
            # 中文：增量解码时，位置编码要用"绝对位置"，而不能每次都从 0 开始，
            # 否则新 token 会被错误地当成序列的第一个 token。
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)  # (1, seq_len, emb_dim)，会广播到 batch 维度
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)  # 中文：逐层前向，并把 use_cache 透传给每个 TransformerBlock
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)  # (batch_size, seq_len, vocab_size)
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        """重置所有层的 KV 缓存以及模型的位置计数器。

        在开始处理一个新的 prompt / 新的生成序列之前调用，
        避免残留上一次生成的缓存状态影响本次结果。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用（可选）KV 缓存进行贪心解码的简单文本生成函数。

    参数：
        model (GPTModel): 已实例化的 GPT 模型。
        idx (Tensor): 起始 token id 序列（prompt），形状为 (batch, prompt_len)。
        max_new_tokens (int): 需要新生成的 token 数量。
        context_size (int or None): 模型能处理的最大上下文长度，用于在
            不使用缓存时对输入做截断；默认使用模型位置嵌入表的大小。
        use_cache (bool): 是否启用 KV 缓存加速生成。
            - True：先用完整 prompt 做一次前向填充缓存（prefill），
              之后每步只需要把"新生成的单个 token"喂给模型，
              大幅减少重复计算。
            - False：不使用缓存，每一步都要把（截断后的）完整历史序列
              重新输入模型，计算量更大但实现更简单，可用于验证 KV 缓存
              版本结果的正确性。

    返回：
        Tensor: 拼接了 prompt 和新生成 token 的完整序列，
            形状为 (batch, prompt_len + max_new_tokens)。
    """
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            # 中文：先清空缓存，然后用完整 prompt 做一次"预填充"（prefill）前向，
            # 这一步会把 prompt 中所有 token 的 K/V 都算出来并存入缓存。
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                # 中文：因为历史 token 的 K/V 已经缓存，所以这里只需要把
                # 刚生成的这 1 个新 token 喂给模型，模型内部会自动把它的 K/V
                # 与缓存拼接，这正是 KV 缓存能加速生成的核心原因。
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                # 中文：不使用缓存时，每一步都要重新输入（截断到 ctx_len 的）
                # 完整历史序列，模型会重新计算所有历史 token 的注意力，
                # 计算量随生成长度增加而增加，速度明显慢于使用缓存的版本。
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    """命令行入口：构建一个小型 GPT 模型，用 KV 缓存生成文本并统计耗时/显存。

    可通过命令行参数配置模型规模（emb_dim/n_heads/n_layers）、
    生成长度（max_new_tokens）以及滑动窗口注意力的相关超参数
    （sliding_window_size / sliding_window_stride）。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run GPT with standard multi-head attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--sliding_window_size", type=int, default=1024, help="Window size for sliding window attention.")
    parser.add_argument("--sliding_window_stride", type=int, default=2, help="K:1 frequency sliding window attention is applied. K=5 means 5 sliding window layers follows by a regular layer.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
        "sliding_window_size": args.sliding_window_size,
        "sliding_window_stride": args.sliding_window_stride
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)  # 中文：使用 bfloat16 精度以节省显存、加快推理
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
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
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
