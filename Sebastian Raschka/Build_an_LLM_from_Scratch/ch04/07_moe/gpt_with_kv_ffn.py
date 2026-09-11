# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

"""
【中文模块说明】
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书中
第 3-4 章内容的汇总实现，用于演示一个"迷你 GPT"模型的完整搭建过程，并额外加入了：

1. KV 缓存（KV cache）：在自回归生成（逐 token 生成）时，把历史 token 的 Key/Value
   向量缓存下来，避免每生成一个新 token 都要重新计算全部历史位置的 Key/Value，
   从而大幅提升推理速度。
2. FeedForward（前馈网络）的耗时与显存统计：用于对比"标准 FFN"与后续章节中
   "MoE（专家混合，Mixture-of-Experts）前馈网络"在推理时的时间/显存开销，
   为本章 07_moe 小节的对比实验做铺垫。

模型结构遵循经典 GPT 架构：
    Token Embedding + Position Embedding
        -> N 层 TransformerBlock（每层：多头自注意力 + 前馈网络，均带残差连接与 LayerNorm）
        -> 最终 LayerNorm
        -> 输出线性层（映射到词表大小，得到每个位置的下一个 token 的 logits）

本文件既可以被其他脚本 import 使用（如 07_moe 目录下用于做 MoE 对比实验的脚本），
也可以直接作为独立脚本运行（见文件末尾的 `if __name__ == "__main__":`），
运行时会用一个简单的 prompt 做贪婪解码（greedy decoding）生成文本，并打印生成速度、
显存占用等统计信息。
"""

import argparse
import time
import tiktoken
import torch
import torch.nn as nn

# 用于记录每次调用 FeedForward（前馈网络）所耗费的时间（毫秒）和显存增量（字节）。
# 这是全局列表，在 generate_text_simple_cached 生成开始前会被清空，
# 生成结束后统计平均值，方便和 MoE 版本的 FFN 做性能对比。
FFN_TIME_MS = []
FFN_MEM_BYTES = []


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """
    多头自注意力（Multi-Head Self-Attention）模块。

    这是 Transformer 的核心组件之一：把输入序列的每个 token 表示投影成
    Query（查询）、Key（键）、Value（值）三组向量，并拆分成多个"头"（heads）
    并行计算注意力，最后再把各头的输出拼接、投影回原始维度。

    本实现额外集成了 KV 缓存（KV cache）机制，用于自回归生成时避免重复计算
    历史位置的 Key/Value，从而加速推理。

    参数：
        d_in (int): 输入特征维度。
        d_out (int): 输出特征维度（同时也是 Q/K/V 投影后的总维度），
            要求能被 num_heads 整除。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool): Q/K/V 线性层是否使用偏置项，默认为 False。
    """

    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头的维度 = 总输出维度 / 头数，
        # 这样多头拼接回去后总维度依然等于 d_out。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # KV cache-related code
        # 中文：KV 缓存相关代码。
        # cache_k / cache_v 用 register_buffer 注册为非持久化 buffer（不会被保存进 state_dict），
        # 初始为 None，表示还没有缓存任何历史 Key/Value。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        # ptr_current_pos 记录当前已经处理到序列的第几个位置，
        # 用于在使用缓存时给新的 Query token 分配正确的绝对位置索引（构造因果掩码时要用到）。
        self.ptr_current_pos = 0
        ####################################################

    def forward(self, x, use_cache=False):
        """
        前向传播。

        参数：
            x (Tensor): 输入张量，形状 (b, num_tokens, d_in)，
                即 (batch_size, 当前输入的 token 数, 输入维度)。
                注意：当 use_cache=True 且已有缓存时，num_tokens 通常只是"新增的" token 数
                （比如逐 token 生成时每次只有 1 个新 token），而不是完整序列长度。
            use_cache (bool): 是否启用 KV 缓存。True 时会把本次算出的 K/V
                拼接到历史缓存中，并对完整的历史 K/V 做注意力计算。

        返回：
            context_vec (Tensor): 注意力输出，形状 (b, num_tokens, d_out)，
                num_tokens 与输入 x 的 token 数一致（即只对本次输入的 Query 计算输出）。
        """
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)
        # 中文：分别对输入做线性投影，得到本次新增 token 的 Key/Value/Query，
        # 此时形状均为 (b, num_tokens, d_out)。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆分成 (num_heads, head_dim)，
        # 相当于把一个大的线性投影"隐式地"切分成多个头各自独立的子空间。
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache-related
        # 中文：KV 缓存的核心逻辑——
        # 如果是第一次调用（cache_k 为 None），直接把本次的 K/V 作为缓存；
        # 否则把新算出的 K/V 沿着 token 维度（dim=1，即序列长度维）拼接到已有缓存后面，
        # 这样就得到了"历史全部 token + 本次新 token"的完整 Key/Value 序列，
        # 而不需要重新计算历史 token 的 K/V 投影，节省了大量重复计算。
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            # 中文：不使用缓存时（如训练阶段，或一次性对完整 prompt 做前向），
            # 直接用本次算出的 K/V，不做任何拼接。
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度换到 batch 维度之后，方便对每个头独立做矩阵乘法（批量矩阵乘）。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文：Query 与 Key 做点积，得到注意力分数（未缩放、未加掩码）。
        # queries 形状 (b, num_heads, num_tokens_Q, head_dim)
        # keys.transpose(2,3) 形状 (b, num_heads, head_dim, num_tokens_K)
        # 相乘后 attn_scores 形状为 (b, num_heads, num_tokens_Q, num_tokens_K)，
        # 表示每个 Query 位置对每个 Key 位置的"相关性"打分。

        ####################################################
        # causal mask
        # 中文：因果掩码（causal mask）——保证每个位置只能"看到"它自己以及它之前的 token，
        # 不能看到未来的 token，这是自回归语言模型的核心约束。
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            # 中文：使用缓存时，本次的 Query 对应的是序列中"绝对位置"从
            # ptr_current_pos 到 ptr_current_pos + num_tokens_Q - 1 的那些 token
            # （而不是从 0 开始），因此要用 ptr_current_pos 来偏移，
            # 才能保证掩码基于正确的绝对位置计算。
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q
        else:
            # 中文：不使用缓存时，本次输入就是从头开始的完整序列，
            # Query 的绝对位置就是 0..num_tokens_Q-1，并重置位置指针。
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        # 中文：mask_bool[i, j] = True 表示 Query 位置 i 严格早于 Key 位置 j
        # （即 j 是"未来"的 token），此处应当被屏蔽（不允许看到）。
        # 通过广播 (num_tokens_Q, 1) < (1, num_tokens_K) 得到形状 (num_tokens_Q, num_tokens_K) 的布尔矩阵。
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        # Use the mask to fill attention scores
        # 中文：把被掩盖的位置（未来 token）填成负无穷，这样经过 softmax 后其权重趋近于 0，
        # 相当于彻底阻止了"偷看未来"。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：对注意力分数做缩放（除以 sqrt(head_dim)，即 keys.shape[-1] 开根号），
        # 这是经典的"缩放点积注意力"(scaled dot-product attention)，
        # 缩放的目的是防止 head_dim 较大时点积数值过大导致 softmax 梯度消失。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：用注意力权重对 Value 做加权求和，得到每个 Query 位置的上下文向量（context vector）。
        # attn_weights: (b, num_heads, num_tokens_Q, num_tokens_K)
        # values:       (b, num_heads, num_tokens_K, head_dim)
        # 相乘结果:      (b, num_heads, num_tokens_Q, head_dim)
        # 再 transpose(1, 2) 换回 (b, num_tokens_Q, num_heads, head_dim)。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多头的输出重新拼接（flatten）回单一维度 d_out，
        # 形状变为 (b, num_tokens, d_out)。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：out_proj 是一个额外的线性层，用于混合各头信息（多头拼接后再做一次线性变换）。

        return context_vec

    def reset_cache(self):
        """
        重置 KV 缓存。

        将缓存的 Key/Value 清空为 None，并将位置指针归零。
        通常在开始一段新的生成序列（新的 prompt）之前调用，
        避免上一次生成残留的缓存污染本次计算。
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）模块。

    对最后一维（特征维度）做均值方差归一化，然后用可学习的缩放（scale）和
    偏移（shift）参数进行仿射变换。作用是稳定训练过程中每一层输入的分布，
    缓解梯度消失/爆炸问题。

    参数：
        emb_dim (int): 归一化的特征维度大小（即嵌入维度）。
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除以 0 的极小数值稳定项。
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习缩放参数，初始化为全 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习偏移参数，初始化为全 0

    def forward(self, x):
        """
        参数：
            x (Tensor): 形状 (..., emb_dim)，通常为 (batch, num_tokens, emb_dim)。
        返回：
            Tensor: 与输入同形状，最后一维已做归一化+仿射变换。
        """
        mean = x.mean(dim=-1, keepdim=True)
        # 中文：unbiased=False 表示使用有偏方差估计（除以 N 而非 N-1），
        # 这是深度学习框架里 LayerNorm 的标准做法。
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    GELU（Gaussian Error Linear Unit）激活函数的近似实现（tanh 近似版本）。

    GELU 是 GPT 系列模型中常用的激活函数，相比 ReLU 更平滑，
    在原点附近对负值也有一定的非零响应，有助于模型学习更细腻的非线性关系。
    注：本文件中实际的 FeedForward 已经改用 SwiGLU（见下方 FeedForward 类），
    这里保留 GELU 类定义以兼容/参考经典实现。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        参数：
            x (Tensor): 任意形状的输入张量。
        返回：
            Tensor: 与输入同形状，逐元素应用 GELU 激活后的结果。
        """
        # 中文：GELU 的 tanh 近似公式：
        # GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# class FeedForward(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
#             GELU(),
#             nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
#         )

#     def forward(self, x):
#         return self.layers(x)

# 中文：以上被注释掉的代码是"经典版"FeedForward（Linear -> GELU -> Linear），
# 即原书第 4 章标准 GPT 的前馈网络实现，这里保留作为对照，未删除、未启用。

# Uses SwiGLU instead of GeLU to make it more comparable to MoE
# 中文：为了让"稠密 FFN"与后续章节的 MoE（专家混合）前馈网络在结构上更具可比性，
# 这里改用 SwiGLU（Swish-Gated Linear Unit）门控结构，
# 这也是 LLaMA 等现代 LLM 中常用的前馈网络形式。
class FeedForward(nn.Module):
    """
    基于 SwiGLU 的前馈网络（Feed-Forward Network）。

    结构：out = fc3( SiLU(fc1(x)) * fc2(x) )
    其中 fc1、fc2 把输入从 emb_dim 投影到 hidden_dim（两路并行的门控分支），
    fc3 再把 hidden_dim 投影回 emb_dim。这种"门控 + 逐元素相乘"的结构
    相比单一的 Linear->激活->Linear，表达能力更强，也是与 MoE 中单个专家
    结构保持一致，便于后续做"稠密 FFN vs. MoE"的性能与效果对比。

    参数：
        cfg (dict): 配置字典，需包含：
            - "emb_dim": 输入/输出的嵌入维度。
            - "hidden_dim": 中间隐藏层维度。
    """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)

    def forward(self, x):
        """
        参数：
            x (Tensor): 形状 (batch, num_tokens, emb_dim)。
        返回：
            Tensor: 形状 (batch, num_tokens, emb_dim)，与输入形状一致。
        """
        # 中文：SiLU(fc1(x)) 起"门控"作用（类似开关，决定每个隐藏单元的激活强度），
        # 与 fc2(x) 逐元素相乘后再通过 fc3 投影回原始维度。
        # 中间张量 fc1(x)/fc2(x) 形状均为 (batch, num_tokens, hidden_dim)。
        return self.fc3(torch.nn.functional.silu(self.fc1(x)) * self.fc2(x))


class TransformerBlock(nn.Module):
    """
    Transformer 基本模块（一层 Transformer Block）。

    标准的 Pre-LayerNorm 结构：
        x -> LayerNorm -> 多头自注意力 -> Dropout -> 残差相加
          -> LayerNorm -> 前馈网络（FFN）-> Dropout -> 残差相加

    参数：
        cfg (dict): 配置字典，需包含 "emb_dim"、"n_heads"、"drop_rate"、
            "qkv_bias" 等键（详见 GPTModel 中的配置说明）。
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 中文：注意力子层前的归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 中文：前馈子层前的归一化
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        """
        参数：
            x (Tensor): 形状 (batch, num_tokens, emb_dim)。
            use_cache (bool): 是否启用 KV 缓存（透传给内部的注意力模块）。
        返回：
            Tensor: 形状 (batch, num_tokens, emb_dim)，与输入形状一致。
        """
        # Shortcut connection for attention block
        # 中文：残差连接（shortcut/skip connection）——先保存输入，
        # 后面会把子层输出加回这个原始输入，缓解深层网络的梯度消失问题。
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        # 中文：把 use_cache 参数透传给多头注意力模块，以便复用/更新 KV 缓存。
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        # 中文：以下这段是为了统计 FeedForward（FFN）单次调用的耗时与显存占用，
        # 目的是与后续 MoE 版本的 FFN 做性能对比（MoE 通常每个 token 只激活部分专家，
        # 理论上可以在保持模型容量的同时降低单次前向的计算量）。
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()          # 中文：同步 CUDA 流，确保之前的操作都已完成，计时才准确
            torch.cuda.reset_peak_memory_stats()  # 中文：重置显存峰值统计，便于单独测量本次 FFN 调用的峰值显存
            base_mem = torch.cuda.memory_allocated()  # 中文：记录 FFN 调用前已分配的显存基线
        start = time.perf_counter()
        x = self.ff(x)
        if use_cuda:
            torch.cuda.synchronize()          # 中文：等待 FFN 计算真正完成后再计时/统计显存
            peak_mem = torch.cuda.max_memory_allocated()
            FFN_MEM_BYTES.append(peak_mem - base_mem)  # 中文：记录本次 FFN 调用相对基线的显存增量
        FFN_TIME_MS.append((time.perf_counter() - start) * 1000.0)  # 中文：记录本次 FFN 调用耗时（毫秒）
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """
    完整的 GPT 语言模型。

    结构：
        Token Embedding + Position Embedding
        -> Dropout
        -> N 层 TransformerBlock（堆叠）
        -> 最终 LayerNorm
        -> 输出线性层（Linear，映射到词表大小，得到 logits）

    额外支持 KV 缓存机制，用于加速自回归生成。

    参数：
        cfg (dict): 配置字典，需包含：
            - "vocab_size": 词表大小。
            - "context_length": 支持的最大上下文长度（用于位置编码表大小）。
            - "emb_dim": 嵌入维度。
            - "hidden_dim": FFN 中间隐藏层维度。
            - "n_heads": 注意力头数。
            - "n_layers": Transformer 层数。
            - "drop_rate": dropout 概率。
            - "qkv_bias": Q/K/V 投影是否使用偏置。
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
        # 中文：原本用 nn.Sequential 依次串联各层即可（因为默认前向不需要额外参数），
        # 但由于 TransformerBlock.forward 现在多了一个 use_cache 参数，
        # nn.Sequential 无法方便地把额外参数透传给每一层，
        # 因此改用 nn.ModuleList，在下面的 forward 中手动 for 循环调用每一层。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 中文：current_pos 记录整个模型当前已经生成/处理到的绝对位置，
        # 用于在使用 KV 缓存时给位置编码（position embedding）分配正确的索引。
        self.current_pos = 0
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        """
        参数：
            in_idx (Tensor): 输入的 token id 序列，形状 (batch_size, seq_len)，
                dtype 为整型。当 use_cache=True 且已有缓存时，seq_len 通常只是
                "新增"的 token 数（如逐 token 生成时 seq_len=1）。
            use_cache (bool): 是否启用 KV 缓存。

        返回：
            logits (Tensor): 形状 (batch_size, seq_len, vocab_size)，
                表示每个输入位置预测"下一个 token"的未归一化对数概率分布。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文：tok_embeds 形状 (batch_size, seq_len, emb_dim)，
        # 把每个 token id 查表映射为对应的嵌入向量。

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        # 中文：位置编码需要根据"绝对位置"而不是"本次输入内的相对位置"来查表，
        # 否则使用缓存时，每次新 token 的位置编码都会被错误地重置为从 0 开始。
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len  # 中文：更新位置指针，为下一次调用做准备
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        # 中文：pos_embeds 原形状 (seq_len, emb_dim)，unsqueeze(0) 后变为
        # (1, seq_len, emb_dim)，便于与 tok_embeds 做广播相加。
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：词嵌入 + 位置嵌入，得到融合了"语义信息"和"位置信息"的输入表示。
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        # 中文：手动逐层调用 TransformerBlock，并把 use_cache 透传下去，
        # 这样每一层内部的注意力模块都能各自维护自己的 KV 缓存。
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        # 中文：logits 形状 (batch_size, seq_len, vocab_size)，
        # 对最后一维做 softmax/argmax 即可得到下一个 token 的概率分布/预测结果。
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        """
        重置整个模型的 KV 缓存。

        遍历所有 TransformerBlock，调用其内部注意力模块的 reset_cache，
        并把模型级别的位置指针 current_pos 归零。
        通常在开始处理一个全新的 prompt / 新的生成序列之前调用，
        避免复用上一次生成残留的缓存和位置信息。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """
    使用（可选）KV 缓存进行贪婪解码（greedy decoding）的文本生成函数。

    每一步都选取当前 logits 中概率最大（argmax）的 token 作为下一个 token，
    不做任何采样随机性（因此生成结果是确定性的）。

    参数：
        model (GPTModel): 已训练好的 GPT 模型实例。
        idx (Tensor): 初始 prompt 的 token id，形状 (batch_size, base_len)。
        max_new_tokens (int): 要生成的新 token 数量。
        context_size (int, optional): 模型能处理的最大上下文长度；不传则使用
            model.pos_emb.num_embeddings（即位置编码表的大小）。
        use_cache (bool): 是否启用 KV 缓存来加速生成。True 时每步只需把
            "刚生成的 1 个新 token"喂给模型；False 时每步都要把
            "最近 context_size 个 token"整体重新喂给模型做前向计算。

    返回：
        Tensor: 形状 (batch_size, base_len + max_new_tokens)，
            即 prompt 与新生成 token 拼接后的完整序列
            （若中途因为 cur_len 达到 total_len 上限提前结束，则返回已生成部分，
            实际返回的是 generated[:, :cur_len]）。
    """
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings
    batch_size, base_len = idx.shape
    total_len = base_len + max_new_tokens
    # 中文：预先分配好整段序列的存储空间（prompt + 待生成部分），
    # 避免生成过程中反复 torch.cat 带来的额外开销。
    generated = torch.empty(
        batch_size, total_len, dtype=idx.dtype, device=idx.device
    )
    generated[:, :base_len] = idx
    cur_len = base_len
    use_cuda = torch.cuda.is_available()
    # 中文：每次调用生成函数前，清空上一次遗留的 FFN 耗时/显存统计数据。
    FFN_TIME_MS.clear()
    FFN_MEM_BYTES.clear()

    with torch.no_grad():
        # 中文：生成阶段不需要反向传播，用 torch.no_grad() 关闭梯度计算，
        # 节省显存并加速计算。
        if use_cache:
            # Init cache with full prompt
            # 中文：先重置缓存，然后把完整 prompt（可能截断到最近 ctx_len 个 token）
            # 一次性喂给模型，初始化 KV 缓存，得到 prompt 最后一个位置的 logits。
            model.reset_kv_cache()
            prompt_start = max(0, cur_len - ctx_len)
            logits = model(generated[:, prompt_start:cur_len], use_cache=True)

            if use_cuda:
                torch.cuda.synchronize()

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                # 中文：取最后一个位置（即"下一个待预测 token"对应位置）的 logits，
                # 在词表维度上取 argmax，得到贪婪解码选出的 token id。
                next_idx = logits[:, -1].argmax(dim=-1)
                # b) append it to the running sequence (in-place)
                generated[:, cur_len] = next_idx
                cur_len += 1
                # c) feed model only the new token
                # 中文：得益于 KV 缓存，这里只需要把"刚生成的这 1 个新 token"
                # 喂给模型，模型内部会自动把它的 K/V 拼接到缓存中，
                # 而不需要把整个历史序列重新算一遍，这正是 KV 缓存加速的关键所在。
                logits = model(generated[:, cur_len - 1 : cur_len], use_cache=True)

                if use_cuda:
                    torch.cuda.synchronize()
        else:
            # 中文：不使用缓存的对照分支——每一步都要把"最近 ctx_len 个 token"
            # 整体重新喂给模型做一次完整前向计算，计算量随生成长度增长而显著增加，
            # 用于和上面 use_cache=True 的分支做速度对比。
            if use_cuda:
                torch.cuda.synchronize()

            for _ in range(max_new_tokens):
                start_ctx = max(0, cur_len - ctx_len)
                logits = model(generated[:, start_ctx:cur_len], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1)
                generated[:, cur_len] = next_idx
                cur_len += 1

                if use_cuda:
                    torch.cuda.synchronize()

    # 中文：生成结束后，汇总打印 FFN 的平均耗时、平均显存增量、峰值显存增量，
    # 方便与 MoE 版本的 FFN 实现做对比分析。
    if FFN_TIME_MS:
        avg_ffn_time = sum(FFN_TIME_MS) / len(FFN_TIME_MS)
        print(f"Avg FFN time/call: {avg_ffn_time:.3f} ms")
    if FFN_MEM_BYTES:
        avg_ffn_mem = sum(FFN_MEM_BYTES) / len(FFN_MEM_BYTES)
        max_ffn_mem = max(FFN_MEM_BYTES)

        def to_mb(bytes_val):
            return bytes_val / (1024 ** 2)
        print(f"Avg FFN mem delta/call: {to_mb(avg_ffn_mem):.2f} MB (max {to_mb(max_ffn_mem):.2f} MB)")

    return generated[:, :cur_len]


def main():
    """
    脚本入口函数：解析命令行参数、构建模型、执行一次贪婪解码生成，
    并打印生成结果、耗时、生成速度（tokens/sec）以及显存占用等统计信息。

    支持的命令行参数（均有默认值，详见 argparse 定义）：
        --emb_dim: 模型嵌入维度。
        --hidden_dim: FFN 中间隐藏层维度。
        --n_heads: 注意力头数。
        --n_layers: Transformer 层数。
        --max_new_tokens: 生成的新 token 数量。
        --no_kv_cache: 若指定该开关，则禁用 KV 缓存（用于对比性能）。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=768*4, help="Intermediate FFN size.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV caching during generation.",
    )

    args = parser.parse_args()

    start_context = "Hello, I am"
    # 中文：使用 GPT-2 的 BPE 分词器（tiktoken）对起始文本做编码，得到 token id 列表。
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,            # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        # 中文：context_length 设为"待生成 token 数 + prompt 长度"，
        # 保证位置编码表足够覆盖本次生成会用到的所有绝对位置。
        "emb_dim": args.emb_dim,        # Embedding dimension
        "hidden_dim": args.hidden_dim,  # Intermediate size
        "n_heads": args.n_heads,        # Number of attention heads
        "n_layers": args.n_layers,      # Number of layers
        "drop_rate": 0.0,               # Dropout rate
        "qkv_bias": False,              # Query-Key-Value bias
    }
    torch.manual_seed(123)  # 中文：固定随机种子，保证模型参数初始化可复现
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    # 中文：把模型搬到可用设备（有 GPU 用 GPU，否则用 CPU），
    # 并使用 bfloat16 精度以节省显存、提升推理速度（这里是随机初始化的模型，
    # 仅用于演示生成流程与性能对比，并非真正训练好的语言模型，故生成内容不具备语义意义）。
    model.eval()  # disable dropout
    # 中文：切换到 eval 模式，关闭 dropout 等只在训练时生效的行为。

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    # 中文：unsqueeze(0) 增加 batch 维度，形状变为 (1, prompt_len)。
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
        use_cache=not args.no_kv_cache,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    # 中文：把生成的 token id 序列解码回可读文本。

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    # 中文：打印整体生成速度（每秒生成的 token 数），用于评估 KV 缓存带来的加速效果。
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
