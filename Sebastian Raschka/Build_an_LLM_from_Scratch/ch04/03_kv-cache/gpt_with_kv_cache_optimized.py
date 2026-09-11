# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

"""
【中文说明】本文件用途与在《从零构建大语言模型》(Build a Large Language Model From Scratch)
一书中的角色：

本文件整合了第 3 章（自注意力 / 多头注意力）和第 4 章（GPT 模型整体架构：LayerNorm、
GELU、FeedForward、TransformerBlock、GPTModel）已经实现过的代码，并在此基础上加入了
**KV 缓存（Key-Value Cache）** 与 **滑动窗口（sliding window）** 的优化实现，属于第 4 章
配套的进阶示例（ch04/03_kv-cache）。

背景知识——为什么需要 KV 缓存：
在自回归（autoregressive）文本生成中，模型每生成一个新 token，都需要重新对“当前已有的
全部序列”做一次前向传播来计算注意力。如果不做任何优化，每生成一个新 token 就要把之前
所有 token 的 Key、Value 重新计算一遍，计算量随序列长度增长是 O(n^2) 级别，非常浪费。

KV 缓存的核心思想是：把历史 token 已经算好的 Key/Value 张量缓存下来，后续只需要为“新
增的 token”计算一次 Key/Value，然后与缓存拼接（或写入缓存的固定窗口），从而把每一步
生成的计算量降到接近 O(n)。本文件中的 `MultiHeadAttention` 类相比标准实现新增了
`cache_k` / `cache_v` 缓冲区、`ptr_cur` 指针、滑动窗口逻辑（`window_size`），
`GPTModel` 也新增了位置编码指针 `ptr_current_pos`，用于在“只喂入新 token”的场景下，
仍然能正确地取到对应位置的位置编码（positional embedding）。

此外还实现了滑动窗口机制：当输入的 token 数量超过 `window_size` 时，缓存会自动丢弃
最旧的 token（“先进先出”），从而在生成很长文本时把显存占用限制在一个固定上限内，
代价是模型只能“看到”窗口内的历史上下文（类似有限注意力窗口/local attention 的效果）。

本文件可以直接作为脚本运行（见文件底部的 `main()` 函数），会构造一个 GPT-2 124M 规模的
模型（未训练，随机初始化权重），并使用 KV 缓存版本的生成函数 `generate_text_simple_cached`
生成文本，同时打印生成速度（tokens/sec）和显存占用，用于直观感受 KV 缓存带来的推理加速。
"""

import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """多头自注意力（Multi-Head Self-Attention）模块，带 KV 缓存与滑动窗口支持。

    这是第 3 章讲解的多头注意力机制的“优化版”实现：在标准的 Q/K/V 投影 + 缩放点积注意力
    + 因果掩码（causal mask）的基础上，新增了 KV 缓存（`cache_k`/`cache_v`）与滑动窗口
    （`window_size`）机制，用于加速自回归生成、并限制显存占用。

    参数说明：
        d_in (int): 输入特征维度（即输入张量最后一维的大小）。
        d_out (int): 输出特征维度，同时也是 Q/K/V 投影后的总维度，需能被 num_heads 整除。
        context_length (int): 模型支持的最大上下文长度，用于在未显式传入 max_seq_len 时
            作为缓存长度的默认值。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头的数量，每个头的维度 head_dim = d_out // num_heads。
        qkv_bias (bool): Q/K/V 线性层是否使用偏置项。
        max_seq_len (int, 可选): KV 缓存能容纳的最大序列长度，默认等于 context_length。
        window_size (int, 可选): 滑动窗口大小，即 KV 缓存实际保留的最近 token 数量，
            默认等于 max_seq_len（即不做窗口裁剪，相当于普通 KV 缓存）。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, max_seq_len=None, window_size=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：把总输出维度 d_out 平均切分给 num_heads 个头，每个头只负责 head_dim 维的子空间，
        # 这样多个头可以并行地从不同的“子空间视角”学习不同的注意力模式。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # NEW
        # 中文：以下是相较于“无缓存版”多头注意力新增的部分——KV 缓存相关的状态。
        self.max_seq_len = max_seq_len or context_length
        # 中文：KV 缓存缓冲区在“时间”维度上的最大容量。若未指定则退化为 context_length。
        self.window_size = window_size or self.max_seq_len
        # 中文：滑动窗口大小，缓存中最多保留这么多个最近 token 的 K/V；超出时会丢弃最旧的。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        # 中文：把 cache_k / cache_v 注册为 buffer（而非普通属性），可以让它们跟随模型的
        # .to(device) 一起搬到 GPU/CPU；persistent=False 表示它们不会被保存进 state_dict
        # （因为这是运行时的临时状态，不是需要持久化的模型参数）。
        ####################################################

    def forward(self, x, use_cache=False):
        """前向传播：计算多头自注意力输出，可选择是否使用 KV 缓存。

        参数：
            x (Tensor): 输入张量，形状 (b, num_tokens, d_in)。
                - 不使用缓存时：num_tokens 通常是完整序列长度。
                - 使用缓存时：num_tokens 通常只是“新增的”token 数量（例如生成阶段每步为 1）。
            use_cache (bool): 是否启用 KV 缓存机制。

        返回：
            Tensor: 注意力输出，形状 (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape

        if use_cache:
            # to prevent self.ptr_cur became negative
            # 中文：如果单次输入的新 token 数超过了窗口大小，后面滑动窗口逻辑里 overflow
            # 计算可能出问题（甚至指针变负），所以这里提前断言限制输入块大小。
            assert num_tokens <= self.window_size, (
                f"Input chunk size ({num_tokens}) exceeds KV cache window size ({self.window_size}). "
            )

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)
        # 中文：对当前输入块分别做线性投影得到 Q/K/V；注意这里算的 K/V 只是“新增 token”的
        # K/V（keys_new/values_new），历史 token 的 K/V 会从缓存里取，不需要重新计算，
        # 这正是 KV 缓存节省计算量的关键所在。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 中文：把最后一维 d_out 拆成 (num_heads, head_dim)，为后续把“头”维度提到前面做准备。

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)
        queries = queries.transpose(1, 2)
        # 中文：交换 num_tokens 与 num_heads 维度，使张量形状变为 (b, num_heads, num_tokens, head_dim)，
        # 这样可以把 num_heads 当作一个“批量”维度，方便后面对每个头并行地做矩阵乘法。

        ####################################################
        # NEW
        # 中文：以下是 KV 缓存的核心逻辑：把新算出的 K/V 写入缓存，并在需要时做滑动窗口“左移丢弃”。
        if use_cache:
            if self.cache_k is None or self.cache_k.size(0) != b:
                # 中文：首次调用，或者 batch size 发生变化（比如切换了不同的输入批次），
                # 需要重新分配一块全零的缓存张量。
                self.cache_k = torch.zeros(b, self.num_heads,
                                           self.window_size, self.head_dim,
                                           device=x.device)
                self.cache_v = torch.zeros_like(self.cache_k)
                self.ptr_cur = 0  # pointer to next free slot
                # 中文：ptr_cur 指向缓存中“下一个空闲位置”，也就是当前已经写入了多少个 token 的 K/V。

            # if incoming chunk would overflow discard oldest tokens
            if self.ptr_cur + num_tokens > self.window_size:
                # 中文：如果这次写入会超出窗口容量，说明缓存满了，需要腾出空间——
                # 采用滑动窗口策略：丢弃最旧的 overflow 个 token，把剩余的整体左移。
                overflow = self.ptr_cur + num_tokens - self.window_size
                # shift everything left by `overflow` (cheap view-copy)
                self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
                self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
                # 中文：把 [overflow:] 的内容拷贝到最前面，相当于把整个缓存往左“滑动”了
                # overflow 步，实现固定窗口大小的“先进先出”效果（老的 token 被彻底丢弃）。
                self.ptr_cur -= overflow  # pointer after shift
                # 中文：左移后，有效数据的末尾指针也要相应减少 overflow。

            self.cache_k[:, :, self.ptr_cur:self.ptr_cur + num_tokens, :] = keys_new
            self.cache_v[:, :, self.ptr_cur:self.ptr_cur + num_tokens, :] = values_new
            # 中文：把本次新算出的 K/V 写入缓存中从 ptr_cur 开始的空闲区间。
            self.ptr_cur += num_tokens
            # 中文：更新指针，指向下一次写入的起始位置。

            keys = self.cache_k[:, :, :self.ptr_cur, :]
            values = self.cache_v[:, :, :self.ptr_cur, :]
            # 中文：取出缓存中“当前有效”的那一段（从 0 到 ptr_cur），
            # 即历史 token + 本次新 token 的全部 K/V，用于后面和 queries 做注意力计算。
        else:
            keys, values = keys_new, values_new
            self.ptr_cur = 0  # keep pointer sane if you interleave modes
            # 中文：不使用缓存时，K/V 就是本次输入直接算出的值（相当于标准注意力，无历史记忆）；
            # 同时把指针清零，防止后续如果混用 use_cache=True/False 造成指针状态不一致。
        ####################################################
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文：queries 形状 (b, num_heads, num_tokens, head_dim)，
        # keys.transpose(2,3) 形状 (b, num_heads, head_dim, K)，
        # 矩阵乘法结果 attn_scores 形状为 (b, num_heads, num_tokens, K)，
        # 其中 K 是当前可见的 Key 总数（有缓存时 K = 历史长度 + 本次新增长度，否则 K = num_tokens）。

        ####################################################
        # NEW
        # 中文：由于引入了 KV 缓存，因果掩码（causal mask）不能再简单地用一个 num_tokens x num_tokens
        # 的下三角矩阵来生成——因为 query 的数量（num_tokens）和 key 的数量（K）可能不相等
        # （例如生成阶段每次只有 1 个新 query，但 K 包含了全部历史 token）。这里需要根据
        # 二者是否相等，分别构造合适的掩码。
        K = attn_scores.size(-1)

        if num_tokens == K:
            # No cache → use the pre‑baked triangular mask slice
            # 中文：没有历史缓存（或 query 数等于 key 数）的情况，退化为标准的上三角掩码——
            # 第 i 行只允许看到第 0..i 列，屏蔽掉 j > i 的“未来”位置。
            causal_mask = torch.triu(torch.ones(num_tokens, K, device=x.device, dtype=torch.bool), diagonal=1)
        else:
            # Cached: need to offset the diagonal by (K − num_tokens)
            # 中文：有缓存时，第 i 个 query 实际上对应的是“全局位置” i + offset
            # （offset 是缓存中已有的历史 token 数），所以掩码的对角线需要相应地整体右移 offset，
            # 即只允许 query i 看到 key 列 j <= i + offset（不能看到比自己更晚出现的 token）。
            offset = K - num_tokens  # number of tokens already in cache before this chunk
            row_idx = torch.arange(num_tokens, device=x.device).unsqueeze(1)  # (num_tokens, 1)
            col_idx = torch.arange(K, device=x.device).unsqueeze(0)           # (1, K)
            causal_mask = row_idx + offset < col_idx                          # True where j > i+offset
            # 中文：causal_mask[i, j] = True 表示“j 是未来位置，需要被屏蔽”；
            # 广播比较 row_idx + offset < col_idx 等价于逐元素判断 j > i + offset。
        ####################################################

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        # 中文：把掩码在 batch 和 num_heads 维度上广播（unsqueeze 两次），
        # 将被掩盖位置的注意力分数设为负无穷，这样后面 softmax 之后这些位置的权重会趋近于 0，
        # 从而保证模型在预测第 i 个 token 时不会“看到”未来的信息（因果性/自回归约束）。

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 中文：除以 sqrt(head_dim) 做缩放（scaled dot-product），防止点积数值过大导致 softmax
        # 梯度消失；然后在最后一维（key 维度）做 softmax，得到归一化的注意力权重分布。
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 中文：attn_weights 形状 (b, num_heads, num_tokens, K)，values 形状 (b, num_heads, K, head_dim)，
        # 相乘得到 (b, num_heads, num_tokens, head_dim)，再 transpose(1,2) 换回
        # (b, num_tokens, num_heads, head_dim)，即每个 token 位置汇总了各个头的加权上下文向量。

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 中文：把多个头的输出重新拼接（concat）回一个 d_out 维的向量；
        # .contiguous() 是因为前面 transpose 后张量在内存中不连续，view 前需要先变连续。
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：再经过一个输出线性层，融合各头信息，得到最终的多头注意力输出。

        return context_vec

    ####################################################
    # NEW
    def reset_cache(self):
        """重置 KV 缓存，将缓存清空（置为 None）。

        通常在开始一次新的独立生成序列之前调用，避免上一次生成残留的缓存状态
        污染这一次的注意力计算（因为缓存是跟 batch size 绑定的，且内容是“历史 token”的
        Key/Value，换了新的输入序列后这些历史信息就不再适用）。
        """
        self.cache_k, self.cache_v = None, None
    ####################################################


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）模块。

    对最后一维（特征维 emb_dim）做归一化：减均值、除以标准差，再用可学习的缩放
    （scale）和平移（shift）参数做仿射变换。作用是稳定训练、缓解内部协变量偏移，
    是 Transformer 结构中每个子层前/后常用的归一化手段。

    参数：
        emb_dim (int): 特征维度大小，即需要归一化的最后一维长度。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：极小值，防止除以标准差为 0 时数值不稳定（除零错误）。
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 中文：可学习的缩放参数 γ，初始化为全 1。
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        # 中文：可学习的平移参数 β，初始化为全 0。

    def forward(self, x):
        """前向传播：对输入张量最后一维做归一化。

        参数：
            x (Tensor): 输入张量，形状 (..., emb_dim)，常见为 (batch, num_tokens, emb_dim)。
        返回：
            Tensor: 归一化并做仿射变换后的张量，形状与输入相同。
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 中文：unbiased=False 表示用有偏方差估计（除以 N 而不是 N-1），与常见 LayerNorm 实现保持一致。
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 中文：标准化，使得每个 token 的特征向量均值为 0、方差为 1。
        return self.scale * norm_x + self.shift
        # 中文：再通过可学习参数做缩放和平移，让模型自己决定归一化后的分布形态。


class GELU(nn.Module):
    """GELU（Gaussian Error Linear Unit）激活函数模块，使用 tanh 近似实现。

    这是 GPT-2 等模型中常用的激活函数，相较于 ReLU 更平滑，在原点附近有更细腻的梯度，
    经验上有助于提升语言模型的训练效果。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """前向传播：对输入逐元素应用 GELU 激活函数（tanh 近似公式）。

        参数：
            x (Tensor): 任意形状的输入张量。
        返回：
            Tensor: 与输入形状相同的激活后张量。
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        # 中文：这是 GELU 的近似解析式，比直接用误差函数 erf 计算更快，
        # 是 GPT-2 原始实现中使用的版本。


class FeedForward(nn.Module):
    """前馈网络（Feed-Forward Network，FFN）模块，即 Transformer Block 中的 MLP 子层。

    结构为：Linear（升维到 4 倍）→ GELU 激活 → Linear（降维回原维度）。
    作用是在注意力子层之后，对每个 token 位置独立地做非线性特征变换，增强模型的表达能力。

    参数：
        cfg (dict): 模型配置字典，需包含键 "emb_dim"（嵌入维度）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # 中文：升维到 4 * emb_dim，这是 Transformer 论文以来的常见经验设置，
            # 让中间层有更大的容量去做非线性变换。
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            # 中文：再投影回原始的 emb_dim，保证输出可以与残差连接（shortcut）相加。
        )

    def forward(self, x):
        """前向传播：对输入逐 token 独立地做“升维-激活-降维”的非线性变换。

        参数：
            x (Tensor): 输入张量，形状 (batch, num_tokens, emb_dim)。
        返回：
            Tensor: 输出张量，形状与输入相同 (batch, num_tokens, emb_dim)。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer 块：由“多头自注意力子层”和“前馈网络子层”组成，各自带残差连接与前置 LayerNorm。

    这是 GPT 类模型的基本重复单元，`GPTModel` 会堆叠多个这样的块（层数由 cfg["n_layers"] 决定）。

    参数：
        cfg (dict): 模型配置字典，需包含 "emb_dim"、"context_length"、"n_heads"、
            "drop_rate"、"qkv_bias"，可选包含 "kv_window_size"（KV 缓存滑动窗口大小）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            window_size=cfg["kv_window_size"] if "kv_window_size" in cfg else cfg["context_length"]   # NEW
            # 中文：若配置里提供了 kv_window_size，则用它作为该层注意力的 KV 缓存滑动窗口大小；
            # 否则退化为不做窗口裁剪（等于完整的 context_length）。
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        """前向传播：依次执行“注意力子层 + 残差”和“前馈子层 + 残差”，均为 Pre-LN 结构。

        参数：
            x (Tensor): 输入张量，形状 (batch, num_tokens, emb_dim)。
            use_cache (bool): 是否让内部的注意力层使用 KV 缓存（生成阶段设为 True 可加速）。
        返回：
            Tensor: 输出张量，形状与输入相同 (batch, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        # 中文：Pre-LN（先归一化再进子层）结构，相比 Post-LN 训练更稳定，是 GPT-2 等模型采用的方式。

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        # NEW
        x = self.att(x, use_cache=use_cache)
        # 中文：把 use_cache 透传给底层的多头注意力模块，由它决定是否读写 KV 缓存。
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差连接（residual connection），把子层输出与子层输入相加，
        # 有助于缓解深层网络的梯度消失问题。

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：前馈子层同样采用 Pre-LN + 残差连接的结构。

        return x


class GPTModel(nn.Module):
    """完整的 GPT 风格语言模型：Token 嵌入 + 位置嵌入 + 多层 TransformerBlock + 输出头。

    支持 KV 缓存推理：当 use_cache=True 时，模型内部会维护一个“当前位置指针”
    （ptr_current_pos），使得每次只输入新增的 token 时，也能取到正确的位置编码；
    同时会把 use_cache 透传给每一层 TransformerBlock 及其内部的注意力模块。

    参数：
        cfg (dict): 模型配置字典，需包含：
            "vocab_size"（词表大小）、"context_length"（最大上下文长度）、
            "emb_dim"（嵌入维度）、"n_layers"（Transformer 层数）、"drop_rate"（dropout 概率）、
            以及每层 TransformerBlock 所需的其余键；可选 "kv_window_size"（KV 缓存窗口大小）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 中文：词嵌入表，把 token id 映射为 emb_dim 维的稠密向量。
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 中文：可学习的绝对位置嵌入表，最多支持 context_length 个位置。
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        # NEW
        # 中文：原本用 nn.Sequential 堆叠各层，只能顺序执行、不方便传递 use_cache 这种
        # 额外参数；这里改用 nn.ModuleList，手动在 forward 里逐层调用，
        # 就能把 use_cache 显式传给每一个 TransformerBlock。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.ptr_current_pos = 0
        # 中文：记录“当前已经处理到的位置”，用于 KV 缓存模式下计算下一批 token 应该
        # 使用哪些位置编码（例如第一次喂入长度为 5 的 prompt 后，指针变为 5，
        # 下一次生成单个新 token 时，其位置编码应取第 5 个位置，而不是从 0 开始）。
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 中文：输出投影层，把最终隐藏状态映射回词表大小的 logits，用于预测下一个 token 的概率分布。
        self.kv_window_size = cfg["kv_window_size"]  if "kv_window_size" in cfg else cfg["context_length"]
        # 中文：把 KV 缓存窗口大小保存在模型层面，方便生成函数（如 generate_text_simple_cached）
        # 读取，从而知道每次“预填充（prefill）”时最多能一次性喂入多少个 token。

    def forward(self, in_idx, use_cache=False):
        """前向传播：将 token id 序列映射为下一个 token 的 logits 分布。

        参数：
            in_idx (Tensor): 输入 token id，形状 (batch_size, seq_len)，dtype 为长整型。
            use_cache (bool): 是否启用 KV 缓存（同时决定位置编码是从 0 开始还是接着
                ptr_current_pos 继续）。
        返回：
            Tensor: 输出 logits，形状 (batch_size, seq_len, vocab_size)。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文：tok_embeds 形状 (batch_size, seq_len, emb_dim)。

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        # NEW
        # 中文：原来的写法（上面注释掉的那行）总是从位置 0 开始取位置编码，
        # 这在“非缓存”模式（每次都喂完整序列）下没问题，但在“缓存”模式下，
        # 如果每次只喂入新增的 token，就必须知道这些新 token 在整个序列中的“真实位置”，
        # 否则位置编码会重复从 0 开始，导致模型误以为每次都是在处理句子开头。

        if use_cache:
            context_length = self.pos_emb.num_embeddings
            # to prevent generate more sequence than context_length
            # since longer than context_length will cause model out of bound error when reading the position embedding
            # 中文：位置嵌入表只有 context_length 行，如果指针加上本次序列长度超过这个上限，
            # 说明生成的总长度已经超出模型能表示的最大位置，必须提前报错，
            # 否则 self.pos_emb(pos_ids) 会因为索引越界而抛异常。
            assert self.ptr_current_pos + seq_len <= context_length, (
                f"Position embedding overflow. Want to read {self.ptr_current_pos + seq_len} which excceded size of {context_length}"
            )
            pos_ids = torch.arange(self.ptr_current_pos, self.ptr_current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            # 中文：从“当前累计位置”开始，取本次输入这 seq_len 个 token 对应的位置编号。
            self.ptr_current_pos += seq_len
            # 中文：处理完之后，把指针向前推进 seq_len，为下一次调用做准备。
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
            # 中文：非缓存模式下，每次都是完整序列从头开始，位置编码固定从 0 开始编号。
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        # 中文：unsqueeze(0) 增加一个 batch 维，形状变为 (1, seq_len, emb_dim)，
        # 后面与 tok_embeds 相加时会自动在 batch 维上广播。
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：词嵌入 + 位置嵌入，得到融合了内容信息与位置信息的输入表示。
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # NEW
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
            # 中文：逐层调用 TransformerBlock，并把 use_cache 透传下去，
            # 使每一层的注意力模块都能各自维护自己的 KV 缓存（每层缓存独立存储）。
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        # 中文：logits 形状 (batch_size, seq_len, vocab_size)，每个位置对应“预测下一个 token”
        # 的未归一化分数，后续可用 softmax/argmax 等方式转换为概率或直接取最可能的 token。
        return logits

    ####################################################
    # NEW
    def reset_kv_cache(self):
        """重置整个模型所有层的 KV 缓存，并把位置指针清零。

        通常在开始生成一段全新的、与之前无关的文本序列之前调用，
        确保不会残留上一次生成过程中的历史 Key/Value 或位置指针状态。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.ptr_current_pos = 0
    ####################################################


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """不使用 KV 缓存的朴素自回归文本生成函数（贪心解码 / greedy decoding）。

    每生成一个新 token，都会把“当前完整序列的最后 context_size 个 token”重新喂给模型
    做一次完整的前向传播，计算量随生成长度增长较大，是本文件中用于对照的基准实现
    （相较于下面的 generate_text_simple_cached，没有做任何缓存优化）。

    参数：
        model (GPTModel): 已训练好（或至少已初始化）的 GPT 模型。
        idx (Tensor): 初始 token id 序列，形状 (batch, seq_len)。
        max_new_tokens (int): 需要生成的新 token 数量。
        context_size (int): 模型支持的最大上下文长度，超出部分会被截断，只保留最近的
            context_size 个 token 作为条件上下文。
    返回：
        Tensor: 拼接了新生成 token 之后的完整序列，形状 (batch, seq_len + max_new_tokens)。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        # 中文：只取最近 context_size 个 token，防止序列长度超出模型位置编码的支持范围。

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
            # 中文：注意这里没有传 use_cache，默认按“非缓存模式”对整段 idx_cond 做完整前向计算，
            # 也就是说每一步生成都要把已生成的全部上下文重新算一遍注意力，效率较低。

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]
        # 中文：只关心序列最后一个位置的输出，因为它对应的是“预测下一个 token”的分布。

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        # 中文：贪心解码——直接取概率（logits）最大的词作为下一个 token，不做采样。

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
        # 中文：把新生成的 token 拼接到序列末尾，作为下一轮生成的上下文。

    return idx


####################################################
# NEW
def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    """使用 KV 缓存加速的自回归文本生成函数（贪心解码），是本文件的核心优化演示。

    相比 generate_text_simple，本函数在 use_cache=True 时采用“预填充（prefill）+ 逐 token
    解码（decode）”的两阶段策略：
      1) 预填充阶段：把初始 prompt 按照 kv_window_size 切块，依次喂给模型，
         把每个 token 的 Key/Value 写入缓存（这一步是计算量较大的一次性开销）。
      2) 解码阶段：之后每生成一个新 token，只需要把这一个新 token 喂给模型
         （而不是整个历史序列！），配合已经缓存好的历史 Key/Value 计算注意力，
         从而把每步生成的计算量从 O(当前序列长度) 降到 O(1)（相对而言）。

    参数：
        model (GPTModel): 已初始化的 GPT 模型，需具有 pos_emb、kv_window_size、
            reset_kv_cache 等属性/方法。
        idx (Tensor): 初始 token id 序列（prompt），形状 (batch, seq_len)。
        max_new_tokens (int): 期望生成的新 token 数量上限（实际可能因位置编码上限而更少）。
        context_size (int, 可选): 模型能处理的最大上下文长度，默认使用模型位置嵌入表的大小。
        use_cache (bool): 是否启用 KV 缓存优化路径；若为 False 则退化为逐 token 的朴素解码
            （但仍是每步重新计算整个上下文，只是写法上没有用缓存分支）。
    返回：
        Tensor: 拼接了新生成 token 之后的完整序列，形状 (batch, seq_len + 实际生成的 token 数)。
    """
    model.eval()

    ctx_len = context_size or model.pos_emb.num_embeddings
    # 中文：确定上下文长度上限，默认取模型位置嵌入表能支持的最大长度。
    kv_window_size = model.kv_window_size
    # 中文：KV 缓存的滑动窗口大小，决定预填充阶段每个 chunk 的最大长度。

    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            # 中文：开始一次全新生成前，先清空所有层的 KV 缓存和位置指针，避免脏状态。

            input_tokens = idx[:, -ctx_len:]
            # 中文：如果原始 prompt 比 ctx_len 还长，只保留最近 ctx_len 个 token。
            input_tokens_length = input_tokens.size(1)

            # prefill to handle input_tokens_length > kv_window_size
            for i in range(0, input_tokens_length, kv_window_size):
                chunk = input_tokens[:, i:i+kv_window_size]
                logits = model(chunk, use_cache=True)
                # 中文：预填充（prefill）阶段——如果 prompt 长度超过了单次 KV 窗口大小，
                # 需要分块依次喂入模型，让每个 token 的 K/V 都被写入缓存；
                # 循环结束后，logits 保留的是“最后一个 chunk”前向传播的结果，
                # 其最后一个位置的 logits 就对应“prompt 结束后下一个 token”的预测分布，
                # 可以直接用来开始下面的解码循环，不用额外再算一次。

            # can't generate more than ctx_len of result
            # due to the limitation of position embedding
            max_generable = ctx_len - input_tokens_length
            max_new_tokens = min(max_new_tokens, max_generable)
            # 中文：由于位置编码表最多只有 ctx_len 个位置，prompt 已经占用了
            # input_tokens_length 个位置，所以最多还能再生成 (ctx_len - input_tokens_length) 个
            # 新 token，超出部分会因位置编码越界而报错，这里提前把 max_new_tokens 截断到安全范围。

            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # 中文：贪心解码，取上一步 logits 最后一个位置中概率最高的 token 作为下一个 token。
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True)
                # 中文：关键优化点——这里只把“刚生成的这一个新 token”喂给模型，
                # 而不是像朴素版那样把全部历史 token 都重新算一遍；
                # 模型内部会利用 KV 缓存中已经保存的历史 Key/Value，
                # 只为这个新 token 计算一次 Q/K/V 并与缓存拼接做注意力，计算量大大减少。
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                # 中文：不使用缓存的路径——每一步都把最近 ctx_len 个 token 重新完整地
                # 前向传播一次，等价于朴素解码，仅作为对照/回退选项。
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
####################################################


def main():
    """脚本入口：构建一个未训练的 GPT-2 124M 规模模型，使用 KV 缓存生成一段文本，
    并打印生成结果、耗时、生成速度（tokens/sec）以及（若有 GPU）显存占用，
    用于直观展示 KV 缓存对推理速度的提升效果。
    """
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False,       # Query-Key-Value bias
        "kv_window_size": 1024   # NEW: KV cache window size
        # 中文：这里把 KV 缓存窗口大小设为与 context_length 相同（1024），
        # 即不主动做滑动窗口裁剪；如果想演示滑动窗口丢弃旧 token 的效果，
        # 可以把这个值设得比 context_length 小。
    }

    torch.manual_seed(123)
    # 中文：固定随机种子，保证模型权重初始化可复现（本例并未加载预训练权重，纯随机初始化，
    # 因此生成的文本内容本身没有实际语义，主要用于演示推理流程和速度对比）。
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # disable dropout
    # 中文：切换到 eval 模式，关闭 Dropout，保证推理结果的确定性。

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    # 中文：unsqueeze(0) 增加 batch 维，得到形状 (1, prompt_len) 的输入张量。

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 中文：GPU 上的操作是异步执行的，计时前需要同步，等待之前所有 CUDA 操作真正完成，
        # 否则测出来的时间会不准确。
    start = time.time()

    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=encoded_tensor,
    #     max_new_tokens=200,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )

    ####################################################
    # NEW
    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )
    # 中文：使用带 KV 缓存的生成函数代替上面注释掉的朴素版本，
    # 以体现本文件的核心优化——同样生成 200 个新 token，KV 缓存版本理论上应显著更快。
    ####################################################

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
        # 中文：打印峰值显存占用，方便观察 KV 缓存（尤其是滑动窗口限制）对显存的影响。


if __name__ == "__main__":
    main()
