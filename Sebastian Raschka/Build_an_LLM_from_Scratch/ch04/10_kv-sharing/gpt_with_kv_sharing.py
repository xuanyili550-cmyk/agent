# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4, adapted to use cross-layer KV sharing.
# This file can be run as a standalone script.

"""
【中文模块说明】
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 3、4 章代码的整合版本，并在此基础上演示了一种推理加速/显存节省技巧——
「跨层 KV 共享」(Cross-Layer KV Sharing)。

背景知识：
- 在标准的 Transformer 解码器（如 GPT）中，每一层（TransformerBlock）都会各自
  用自己的权重矩阵 W_key、W_value，把输入 x 投影成该层专属的 Key（K）和 Value（V）张量。
  在自回归生成（一个 token 一个 token 地生成）时，为了避免重复计算历史 token 的 K/V，
  通常会使用「KV 缓存」(KV Cache) 把每一层已经算过的 K/V 保存下来，新 token 来了只需要
  计算新 token 的 K/V 并拼接到缓存后面即可，这样可以显著加速自回归生成。
- 但是，每一层都要维护一份自己的 KV 缓存，层数越多、上下文越长，显存占用就越大。
- 「跨层 KV 共享」的核心思想是：只让模型最前面的若干层（n_kv_producing_layers 层）
  真正计算并缓存自己的 K/V，后面的层则直接复用（共享）前面某一层算好的 K/V，
  不再重复计算，从而节省显存和计算量。这是一种简化版的 GQA/MQA 思路的变体，
  只不过共享发生在「层与层之间」，而不是「注意力头与头之间」。

本文件核心组件：
1. MultiHeadAttentionWithKVSharing：多头自注意力模块，支持普通 KV 缓存，也支持
   直接接收外部传入的共享 K/V（shared_kv）而不自己计算。
2. LayerNorm / GELU / FeedForward：Transformer 中标准的层归一化、激活函数、前馈网络。
3. TransformerBlock：把注意力子层和前馈子层组合起来的标准 Transformer 块。
4. GPTModel：完整的类 GPT 解码器模型，负责词嵌入、位置嵌入、堆叠若干 TransformerBlock，
   并在前向传播时按 n_kv_producing_layers 的设置决定哪些层自己算 K/V、哪些层共享 K/V。
5. generate_text_simple_cached：使用（或不使用）KV 缓存进行自回归文本生成的简单示例函数。
6. main：命令行入口，构建模型、生成文本并打印耗时/吞吐量/显存占用等统计信息。

阅读本文件时，请重点关注被 "#### KV cache-related ####" 包裹起来的代码块，
它们就是在原始 Chapter 3-4 代码基础上，为实现 KV 缓存和跨层共享而新增/修改的部分。
"""

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttentionWithKVSharing(nn.Module):
    """多头自注意力模块（支持 KV 缓存 与 跨层 KV 共享）。

    这是第 3 章标准多头自注意力（Multi-Head Attention）实现的增强版本：
    - 保留了标准多头注意力的核心逻辑：Q/K/V 线性投影 -> 拆分成多头 -> 缩放点积注意力
      （带因果掩码，即只能看到当前及之前的 token）-> 拼回多头 -> 输出投影。
    - 新增了 KV 缓存机制：在自回归生成时，可以把历史 token 的 K/V 缓存下来，
      避免每次都要重新计算全部历史 token 的 K/V。
    - 新增了「跨层 KV 共享」接口：forward 方法可以接收外部传入的 shared_kv
      （即别的层已经算好的 K、V），此时本层就不再自己计算 K/V，而是直接复用，
      从而节省该层的计算量和显存占用。

    参数：
        d_in (int): 输入特征维度（一般等于 emb_dim）。
        d_out (int): 输出特征维度，必须能被 num_heads 整除。
        dropout (float): 注意力权重上使用的 dropout 概率。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool): Q/K/V 的线性层是否使用 bias，默认为 False。
    """
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头分到的维度 = 总输出维度 / 头数，
        # 例如 d_out=768, num_heads=12 时，head_dim=64。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # 中文：out_proj 用于把多头拼接后的向量再做一次线性变换（混合各头信息）。
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # KV cache-related code
        # 中文：KV 缓存相关代码。
        # 用 register_buffer 注册 cache_k / cache_v，这样它们会随模型一起被 .to(device) 移动，
        # 但 persistent=False 表示不会被保存进 state_dict（因为缓存是运行时状态，不是可训练参数）。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0  # 中文：记录当前已经处理到序列中的哪个位置（用于构造因果掩码的位置索引）。
        ####################################################

    def forward(self, x, use_cache=False, shared_kv=None):
        """前向传播：计算多头自注意力输出。

        参数：
            x (Tensor): 输入张量，形状 (b, num_tokens, d_in)，
                        b=batch_size, num_tokens=当前这次前向传播输入的 token 数
                        （注意：使用缓存时，num_tokens 通常只是新增的 token 数，而不是完整序列长度）。
            use_cache (bool): 是否使用/维护 KV 缓存。
            shared_kv (tuple | None): 若不为 None，则是形如 (keys, values) 的元组，
                        表示由更早的层算好并共享过来的 K/V，本层将直接复用它们，不再自己计算。

        返回：
            context_vec (Tensor): 注意力输出，形状 (b, num_tokens, d_out)。
            new_shared_kv (tuple | None): 若本层自己计算了新的 K/V（即 shared_kv 为 None 的情况），
                        则返回 (keys, values) 供后续层共享使用；否则返回 None。
        """
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把线性投影后的 Q 张量最后一维拆成 (num_heads, head_dim)，
        # 这样每个头拥有自己独立的一段特征子空间，从而实现「多头」并行关注不同信息。
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache-related
        # 中文：以下是 KV 缓存 / 跨层共享的核心分支逻辑。
        if shared_kv is None:
            # 中文：本层需要自己计算 K/V（要么是模型最前面的 n_kv_producing_layers 层，
            # 要么是没有开启跨层共享的普通场景）。
            keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
            values_new = self.W_value(x)

            keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
            values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)

            if use_cache:
                # 中文：如果开启了 KV 缓存：
                # - 第一次调用（cache_k 为 None）时，直接把新算出的 K/V 作为缓存的起点；
                # - 之后每次调用，把新 token 的 K/V 沿 seq_len 维度（dim=1）拼接到已有缓存后面，
                #   这样缓存里始终保存着「从序列开始到当前」所有 token 的 K/V，
                #   而不需要重新计算历史 token 的 K/V，从而加速自回归生成。
                if self.cache_k is None:
                    self.cache_k, self.cache_v = keys_new, values_new
                else:
                    self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                    self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
                keys, values = self.cache_k, self.cache_v
            else:
                # 中文：不使用缓存时，keys/values 就是本次前向传播里刚算出来的（可能是整段 prompt）。
                keys, values = keys_new, values_new
                if self.cache_k is not None or self.cache_v is not None:
                    # 中文：如果之前用过缓存、现在又切换回非缓存模式，需要清空旧缓存，避免状态污染。
                    self.reset_cache()

            new_shared_kv = (keys, values)  # 中文：把本层算好的完整 K/V 打包返回，供后续层共享复用。
        else:
            # 中文：本层不需要自己算 K/V，直接使用外部（更早的层）传入的共享 K/V。
            # 这就是「跨层 KV 共享」节省计算量和显存的关键所在：
            # 后面的层跳过了 W_key/W_value 的计算以及维护自己 KV 缓存的开销。
            keys, values = shared_kv
            new_shared_kv = None  # 中文：既然是复用别人的 K/V，本层就不再产生新的共享结果向下传递。
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度换到前面，方便后续对每个头独立做矩阵乘法（批量矩阵乘法会广播 batch 和 head 维度）。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 中文：计算缩放点积注意力分数。queries @ keys^T
        # queries: (b, num_heads, num_tokens_Q, head_dim)
        # keys^T:  (b, num_heads, head_dim, num_tokens_K)
        # attn_scores: (b, num_heads, num_tokens_Q, num_tokens_K)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        ####################################################
        # causal mask
        # 中文：因果掩码（causal mask）相关代码。
        # 因为使用了 KV 缓存后，Q 的位置和 K 的位置不再是简单的「同一段序列」的两个视角
        # （Q 可能只是新增的最后一个 token，而 K 是从序列开头到现在的所有 token），
        # 所以这里显式地用「绝对位置索引」来构造掩码，而不是简单地用上三角矩阵。
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            # 中文：使用缓存时，本次输入的 Q 对应的是序列中从 ptr_current_pos 开始的
            # num_tokens_Q 个位置（例如生成阶段每次只有 1 个新 token，
            # 其绝对位置就是 ptr_current_pos）。
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q  # 中文：更新指针，为下一次调用做准备。
        else:
            # 中文：不使用缓存时，Q 就是从位置 0 开始的完整序列。
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        # 中文：因果掩码的本质——第 i 个 query 位置只能看到 <= i 的 key 位置。
        # q_positions.unsqueeze(-1) < k_positions.unsqueeze(0) 得到形状
        # (num_tokens_Q, num_tokens_K) 的布尔矩阵，True 表示「该 key 位置在该 query 之后」，
        # 即需要被屏蔽（不允许看到未来信息）。
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        # Use the mask to fill attention scores
        # 中文：把被掩码标记为 True 的位置填成 -inf，这样经过 softmax 后这些位置的注意力权重会趋近于 0。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：对注意力分数按 sqrt(head_dim) 缩放后做 softmax，得到归一化的注意力权重。
        # 缩放是为了防止点积结果过大导致 softmax 梯度消失（Scaled Dot-Product Attention 的标准做法）。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：用注意力权重对 values 做加权求和，得到每个 query 位置的上下文向量；
        # 再把 (b, num_heads, num_tokens, head_dim) transpose 回 (b, num_tokens, num_heads, head_dim)。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多头的输出重新拼接成一个整体向量 (b, num_tokens, d_out)。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：out_proj 对拼接后的多头输出做一次线性变换，融合各头信息。

        return context_vec, new_shared_kv

    def reset_cache(self):
        """清空该注意力层的 KV 缓存，并把位置指针重置为 0。

        通常在开始一次新的生成（新的 prompt）之前调用，避免上一次生成残留的缓存
        污染这一次的计算。
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）模块。

    对每个 token 的特征向量（最后一维）做归一化：减均值、除以标准差，
    再用可学习的 scale（缩放）和 shift（平移）参数做仿射变换。
    这是 Transformer 中稳定训练、加速收敛的常用组件。

    参数：
        emb_dim (int): 特征维度，即需要归一化的最后一维大小。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除以 0 的极小常数。
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 中文：可学习的缩放参数 gamma，初始化为全 1。
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 中文：可学习的平移参数 beta，初始化为全 0。

    def forward(self, x):
        """对输入 x 的最后一维做层归一化。

        参数：
            x (Tensor): 形状 (..., emb_dim)，通常是 (batch, seq_len, emb_dim)。
        返回：
            Tensor: 与输入同形状，已完成归一化 + 仿射变换。
        """
        mean = x.mean(dim=-1, keepdim=True)  # 中文：沿特征维求均值，keepdim 保持维度便于广播。
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 中文：沿特征维求方差（有偏估计，与常见实现一致）。
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 中文：标准化：减均值除以标准差。
        return self.scale * norm_x + self.shift  # 中文：仿射变换，恢复模型的表达能力。


class GELU(nn.Module):
    """GELU 激活函数（此处使用其 tanh 近似公式，与 GPT-2 原始实现一致）。"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """对输入逐元素应用 GELU 激活函数（tanh 近似版本）。

        公式：0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

        参数：
            x (Tensor): 任意形状的输入张量。
        返回：
            Tensor: 与输入同形状。
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer 中的前馈网络（FFN / MLP）子层。

    结构：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)。
    先把维度放大 4 倍再压缩回来，是 GPT 系列模型常见的设计，用于增强模型的非线性表达能力。

    参数：
        cfg (dict): 配置字典，需包含 "emb_dim" 键。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """前向传播：依次通过升维线性层、GELU 激活、降维线性层。

        参数：
            x (Tensor): 形状 (batch, seq_len, emb_dim)。
        返回：
            Tensor: 形状 (batch, seq_len, emb_dim)，与输入相同。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """标准 Transformer 解码器块（含 KV 缓存 / 跨层共享支持）。

    结构：LayerNorm -> 多头自注意力（支持 KV 缓存/共享） -> 残差连接
         -> LayerNorm -> 前馈网络 -> 残差连接
    这是「Pre-LN」结构（LayerNorm 放在子层之前），有利于深层网络的训练稳定性。

    参数：
        cfg (dict): 配置字典，需包含 emb_dim, n_heads, drop_rate, qkv_bias 等键。
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionWithKVSharing(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False, shared_kv=None):
        """前向传播：注意力子层 + 前馈子层，均带残差连接。

        参数：
            x (Tensor): 输入张量，形状 (batch, num_tokens, emb_dim)。
            use_cache (bool): 是否启用 KV 缓存，透传给内部的注意力模块。
            shared_kv (tuple | None): 若不为 None，表示复用外部（更早层）算好的 K/V；
                        若为 None，则本层的注意力模块会自己计算 K/V（是否缓存取决于 use_cache）。

        返回：
            x (Tensor): 该 Transformer 块的输出，形状 (batch, num_tokens, emb_dim)。
            new_shared_kv (tuple | None): 若本层自己产生了新的 K/V，则返回给上层（GPTModel）
                        以便传递给后续需要共享 KV 的层；否则为 None。
        """
        # Shortcut connection for attention block
        # 中文：保存注意力子层的残差连接（shortcut）输入。
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        # 中文：调用支持 KV 缓存/共享的多头注意力模块，并把它返回的 new_shared_kv
        # 继续向上传递给 GPTModel，供后续层判断是否要复用。
        x, new_shared_kv = self.att(x, use_cache=use_cache, shared_kv=shared_kv)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差连接，把归一化+注意力之前的原始输入加回来，缓解深层网络的梯度消失问题。

        # Shortcut connection for feed-forward block
        # 中文：保存前馈子层的残差连接输入。
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：同样地，前馈子层也使用残差连接。

        return x, new_shared_kv


class GPTModel(nn.Module):
    """完整的类 GPT 解码器模型（支持 KV 缓存 与 跨层 KV 共享）。

    结构：Token Embedding + Position Embedding -> Dropout
         -> 堆叠 n_layers 个 TransformerBlock（其中只有前 n_kv_producing_layers 层
            自己计算并（可选）缓存 K/V，其余层直接复用第一批层产生的 K/V）
         -> 最终 LayerNorm -> 输出线性层（映射到词表大小，得到 logits）

    参数：
        cfg (dict): 模型配置字典，需包含：
            vocab_size (int): 词表大小。
            context_length (int): 支持的最大序列长度（用于位置嵌入表大小）。
            emb_dim (int): 嵌入/隐藏层维度。
            n_heads (int): 注意力头数。
            n_layers (int): Transformer 块的总层数。
            drop_rate (float): dropout 概率。
            qkv_bias (bool): Q/K/V 线性层是否带 bias。
            n_kv_producing_layers (int): 前多少层自己计算/缓存 K/V，其余层共享这些层的 K/V，
                        取值范围必须在 [1, n_layers] 之间。
    """
    def __init__(self, cfg):
        super().__init__()
        if not 1 <= cfg["n_kv_producing_layers"] <= cfg["n_layers"]:
            raise ValueError("n_kv_producing_layers must be between 1 and n_layers.")
        # 中文：校验参数合法性——至少要有 1 层产生 K/V（否则没人能共享），
        # 最多也不能超过总层数。

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache-related
        # 中文：原本用 nn.Sequential 顺序堆叠即可（因为标准 forward 只需要输入输出一一对应）。
        # 但由于现在每层的 forward 需要多传入/传出 use_cache、shared_kv 等参数，
        # nn.Sequential 无法灵活处理多输入多输出，所以改用 nn.ModuleList，
        # 在 forward 中手动 for 循环遍历每一层，自行控制参数传递。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.n_kv_producing_layers = cfg["n_kv_producing_layers"]  # 中文：记录「产生 K/V 的层数」这一超参数。
        self.current_pos = 0  # 中文：记录当前已经处理到序列的哪个绝对位置（用于生成位置嵌入的位置 id）。
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 中文：输出投影层，把隐藏状态映射回词表大小的 logits，用于预测下一个 token 的概率分布。

    def forward(self, in_idx, use_cache=False):
        """前向传播：把 token id 序列转换为下一个 token 的 logits。

        参数：
            in_idx (Tensor): 输入 token id，形状 (batch_size, seq_len)，
                        使用 KV 缓存时，seq_len 通常只是「新增」的 token 数
                        （例如生成阶段每步只传入 1 个新 token）。
            use_cache (bool): 是否启用 KV 缓存（影响位置编码的起始位置、以及各层是否维护缓存）。

        返回：
            logits (Tensor): 形状 (batch_size, seq_len, vocab_size)，
                        每个位置对词表中每个 token 的打分（未归一化的概率，即 softmax 前的值）。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 中文：查表得到 token 的词嵌入，形状 (batch, seq_len, emb_dim)。

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        # 中文：位置编码也需要感知「当前已经生成到哪个位置」，
        # 否则每次调用都会从位置 0 开始编码，导致生成阶段新 token 的位置编码出错。
        if use_cache:
            # 中文：使用缓存时，本次输入对应序列中从 current_pos 开始的 seq_len 个绝对位置。
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len  # 中文：更新全局位置指针，为下一次调用（比如生成下一个 token）做准备。
        else:
            # 中文：不使用缓存时，默认这次输入就是从头开始的完整序列，位置从 0 开始编号。
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)  # 中文：增加 batch 维度以便与 tok_embeds 广播相加，形状变为 (1, seq_len, emb_dim)。
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：词嵌入 + 位置嵌入，得到每个 token 的初始表示（利用广播机制，pos_embeds 会广播到每个 batch）。
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        # 中文：这里是「跨层 KV 共享」真正生效的地方。
        # shared_kv 初始为 None；遍历每一层时：
        #   - 若当前层下标 layer_idx < n_kv_producing_layers，说明它属于「前面负责产生 K/V 的层」，
        #     调用时传 shared_kv=None，让该层自己计算 K/V（并根据 use_cache 决定是否缓存），
        #     然后把该层返回的 new_shared_kv 保存起来，供后面层复用。
        #     注意：这里每次循环 shared_kv 都会被最新产生 K/V 的层覆盖，
        #     所以实际共享的是「最后一个产生 K/V 的层」（即第 n_kv_producing_layers-1 层）的 K/V。
        #   - 若 layer_idx >= n_kv_producing_layers，说明该层不需要自己算 K/V，
        #     直接把 shared_kv 传进去复用，该层返回的 new_shared_kv 被忽略（用 _ 接收），
        #     因为后续层只需要复用同一份共享 K/V，不需要再产生新的。
        shared_kv = None
        for layer_idx, blk in enumerate(self.trf_blocks):
            if layer_idx < self.n_kv_producing_layers:
                x, shared_kv = blk(x, use_cache=use_cache, shared_kv=None)
            else:
                x, _ = blk(x, use_cache=use_cache, shared_kv=shared_kv)
        ####################################################

        x = self.final_norm(x)  # 中文：最终的层归一化，稳定输出分布。
        logits = self.out_head(x)  # 中文：线性层把隐藏状态映射到词表维度，得到每个位置的下一个 token 打分。
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        """重置整个模型所有层的 KV 缓存和位置指针。

        在开始处理一个全新的 prompt / 一次新的生成任务之前应当调用，
        避免上一次生成残留的缓存状态干扰这一次的计算结果。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()  # 中文：逐层清空各自的 cache_k/cache_v 以及 ptr_current_pos。
        self.current_pos = 0  # 中文：同时重置模型级别的位置指针。
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用（或不使用）KV 缓存的简单自回归贪心解码（greedy decoding）生成函数。

    参数：
        model (GPTModel): 已训练好的 GPT 模型实例。
        idx (Tensor): 初始 prompt 对应的 token id，形状 (batch_size, prompt_len)。
        max_new_tokens (int): 需要新生成的 token 数量。
        context_size (int | None): 模型能处理的最大上下文长度；若为 None，
                    则从 model.pos_emb.num_embeddings 读取（即位置嵌入表的容量）。
        use_cache (bool): 是否启用 KV 缓存来加速生成。
                    - True: 先用完整 prompt 做一次前向传播并建立缓存，
                      之后每步只需要把新生成的 1 个 token 喂给模型（增量推理），速度更快、
                      重复计算更少。
                    - False: 每一步都把「当前完整序列的最后 context_size 个 token」
                      重新完整地喂给模型做一次前向传播，没有复用之前的计算结果，速度较慢，
                      但实现简单、便于对比验证正确性。

    返回：
        idx (Tensor): 形状 (batch_size, prompt_len + max_new_tokens)，
                    即原始 prompt 拼接上新生成的 token 序列。
    """
    model.eval()  # 中文：切换到评估模式，关闭 dropout 等训练专用行为。
    ctx_len = context_size or model.pos_emb.num_embeddings  # 中文：确定允许使用的最大上下文窗口长度。

    with torch.no_grad():  # 中文：生成阶段不需要反向传播，关闭梯度计算以节省显存、加速推理。
        if use_cache:
            # Init cache with full prompt
            # 中文：先清空可能残留的旧缓存，再用完整 prompt 做一次前向传播，
            # 这一步会把 prompt 中每个 token 的 K/V 全部计算并存入缓存。
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                # 中文：贪心采样——直接选取概率（logits）最大的 token，不做随机采样。
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                # 中文：把新生成的 token 拼接到已生成序列末尾。
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                # 中文：由于历史 token 的 K/V 已经在缓存中，这里只需要把新生成的这 1 个 token
                # 喂给模型即可，模型内部会自动把它的 K/V 追加到缓存后面，大幅减少重复计算。
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                # 中文：不使用缓存时，每一步都要把「最近 ctx_len 个 token」整体重新算一遍，
                # 计算量随生成长度增长而显著增加（不复用任何历史计算结果）。
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    """命令行入口：解析参数、构建模型、生成文本，并打印耗时/吞吐量/显存统计信息。

    可通过命令行参数自定义模型规模（emb_dim, n_heads, n_layers）以及
    跨层 KV 共享的层数（n_kv_producing_layers），从而观察不同共享比例对
    生成速度和显存占用的影响。
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run GPT with cross-layer KV sharing.",
    )
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--n_kv_producing_layers", type=int, default=6,
                        help="Number of early layers that compute and cache their own K/V tensors.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")  # 中文：使用 GPT-2 的 BPE 分词器。
    encoded = tokenizer.encode(start_context)  # 中文：把起始文本编码成 token id 列表。

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        # 中文：context_length 设置为「待生成的新 token 数 + 初始 prompt 长度」，
        # 刚好覆盖整个生成过程中会用到的最大位置索引，避免位置嵌入表越界。
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
        "n_kv_producing_layers": args.n_kv_producing_layers,
        # 中文：跨层共享的关键超参数——只有前面这么多层会真正计算/缓存 K/V。
    }
    torch.manual_seed(123)  # 中文：固定随机种子，保证模型参数初始化可复现。
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)  # 中文：使用 bfloat16 半精度，减少显存占用、加快计算（若硬件支持）。
    model.eval()  # disable dropout
    # 中文：eval 模式下 dropout 层会变为恒等映射（不丢弃任何神经元），保证生成结果确定性更好。

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    # 中文：把 token id 列表转换为张量，并增加 batch 维度，形状变为 (1, prompt_len)。
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 中文：等待 GPU 上所有已排队的操作执行完毕，保证计时准确。
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 中文：生成结束后再次同步，确保计时包含全部 GPU 计算时间。
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    # 中文：把生成的 token id 序列（去掉 batch 维）解码回可读文本。

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    # 中文：打印吞吐量（每秒生成 token 数），用于评估 KV 缓存/跨层共享带来的性能提升。
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")
        # 中文：打印 GPU 峰值显存占用，用于评估跨层 KV 共享节省显存的效果。


if __name__ == "__main__":
    main()
