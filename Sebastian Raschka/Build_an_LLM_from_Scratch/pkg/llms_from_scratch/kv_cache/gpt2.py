# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明（docstring）
============================
本文件是《从零构建大语言模型》一书中 GPT-2 模型的实现，相较于普通版本，
本版本额外支持 **KV Cache（键值缓存）**，用于加速自回归（autoregressive）
文本生成时的增量推理（incremental decoding）。

KV Cache 的核心思想：
    在自回归生成时，每生成一个新 token，都需要让它与之前所有 token 做
    注意力计算。如果每次都把全部历史 token 重新过一遍 Q/K/V 投影和注意力，
    会造成大量重复计算。KV Cache 的做法是：把历史 token 的 Key（K）和
    Value（V）张量缓存下来，新的一步只需要计算“新 token”的 Q/K/V，
    然后把新 K/V 拼接（concat）到缓存的 K/V 后面，再与新 Q 做注意力，
    从而把每步计算量从 O(全部序列长度) 降到 O(新增 token 数)。

本文件中与 KV Cache 相关的关键类/逻辑：
    - MultiHeadAttention.forward：读取/写入某一层注意力的 K/V 缓存，
      并在维度 2（序列长度维）上做拼接。
    - TransformerBlock.forward：把 use_cache/start_pos/cache 透传给
      内部的注意力模块。
    - GPTModel.forward：维护 self.current_pos（当前已生成的全局位置），
      用于计算新增 token 的绝对位置编码（position embedding），并通过
      外部传入的 cache 对象（KVCache，定义在 .utils 中）按层读写缓存。
    - GPTModel.reset_kv_cache：清空/重置位置计数器，通常在开始新一轮
      生成前调用。

以下代码在原始实现基础上只新增了中文注释（docstring 与行内注释），
未对任何可执行代码（变量名、函数签名、逻辑顺序、缩进、字符串、导入语句）
做任何修改。原始英文注释全部保留。
"""

from .utils import KVCache   # noqa: F401
# 从同包的 utils 模块导入 KVCache 类（KV 缓存的具体数据结构实现）。
# noqa: F401 表示该导入虽然本文件里没有直接使用它的名字，但需要保留
# （通常是为了让外部使用者可以从本模块 `from gpt2 import KVCache` 导入，
# 属于“再导出 / re-export”用途），因此关闭 flake8 的“未使用导入”告警。

import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """
    多头自注意力（Multi-Head Self-Attention）模块，支持 KV Cache。

    这是标准的因果自注意力（causal self-attention）实现，额外增加了
    use_cache / start_pos / cache 参数以支持增量推理：
      - 训练或首次前向（prefill）时，use_cache 通常为 False，或者
        use_cache=True 但 cache=None（此时相当于把当前输入的 K/V
        作为“全部历史”，返回值 next_cache 作为下一步要用的初始缓存）。
      - 增量解码（decode）阶段，cache 会传入上一步保存的 (keys, values)
        张量，本次前向计算出的“新 token”的 K/V 会拼接到旧缓存后面，
        再与新 token 的 Q 一起做注意力计算。

    参数：
        d_in (int): 输入特征维度。
        d_out (int): 输出特征维度（同时也是 Q/K/V 投影后的总维度）。
        context_length (int): 上下文最大长度（本类中未直接使用，保留接口）。
        dropout (float): 注意力权重的 dropout 概率。
        num_heads (int): 注意力头数，d_out 必须能被其整除。
        qkv_bias (bool): 是否给 Q/K/V 的线性层加偏置项。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 每个头的维度 = 总输出维度 / 头数，保证多头拼接后维度仍为 d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 分别对输入 x 做线性投影，得到 Query / Key / Value
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_cache=False, start_pos=0, cache=None):
        """
        前向传播：计算带因果掩码的多头注意力，并支持 KV Cache 的读写。

        参数：
            x (Tensor): 输入张量，形状 (b, num_tokens, d_in)。
                在增量解码时，num_tokens 通常等于 1（只输入新生成的 token）；
                在首次 prefill 时，num_tokens 等于 prompt 的长度。
            use_cache (bool): 是否启用 KV Cache 机制。
            start_pos (int): 当前这批 token 在完整序列中的起始位置
                （本函数体内实际未直接使用该值来做位置计算，位置信息
                通过因果掩码的形状切片间接体现；start_pos 主要是为了
                和上层调用接口保持一致，供其他实现/未来扩展使用）。
            cache (tuple or None): 上一步缓存的 (keys, values)，
                形状均为 (b, num_heads, past_seq_len, head_dim)；
                如果是第一次调用（没有历史缓存），传 None。

        返回：
            context_vec (Tensor): 注意力输出，形状 (b, num_tokens, d_out)。
            next_cache (tuple or None): 若 use_cache=True，则返回拼接后的
                (keys, values)，形状均为 (b, num_heads, seq_len, head_dim)，
                其中 seq_len = 历史长度 + num_tokens，供下一步调用时传入；
                若 use_cache=False，则返回 None。
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values = self.W_value(x)
        queries = self.W_query(x)
        # 上面三行：分别得到本次输入（新 token，可能只有 1 个）的
        # Key / Value / Query，形状均为 (b, num_tokens, d_out)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 把最后一维 d_out 拆分成 (num_heads, head_dim)，为多头注意力做准备

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # 把 num_heads 维度提前，方便后续按“头”做批量矩阵乘法

        if use_cache:
            if cache is not None:
                # 关键：KV Cache 的“写入/拼接”操作。
                # cache[0]、cache[1] 分别是历史的 keys、values，
                # 形状为 (b, num_heads, past_seq_len, head_dim)；
                # 本次新算出的 keys/values 形状为 (b, num_heads, num_tokens, head_dim)；
                # 在 dim=2（序列长度维）上拼接后，得到
                # (b, num_heads, past_seq_len + num_tokens, head_dim)，
                # 即把“新 token 的 K/V”追加到“历史 K/V”后面，
                # 这样后面做注意力时新 Query 就能看到全部历史 Key/Value。
                keys = torch.cat([cache[0], keys], dim=2)
                values = torch.cat([cache[1], values], dim=2)
            next_cache = (keys, values)
            # 把拼接后（或首次生成）的完整 keys/values 作为“下一步要用的缓存”返回，
            # 调用方（GPTModel）会把它保存起来，供下一次增量前向使用。
        else:
            next_cache = None

        seq_len = keys.size(2)
        # seq_len 是“历史 + 当前”的总键值序列长度（拼接后的长度）
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        # 构造上三角布尔掩码（不含对角线），True 表示“未来位置，需要被屏蔽”
        causal_mask = causal_mask[-num_tokens:, :][None, None, :, :]
        # 只取最后 num_tokens 行——即只保留“本次新 token作为 Query”所对应的掩码行，
        # 因为本次前向只为这 num_tokens 个新 token 计算注意力输出，
        # 而它们可以看到全部 seq_len 个历史+当前 Key（不含未来的）。
        # 增加两个 None 维度，广播成 (1, 1, num_tokens, seq_len)，
        # 以便与 (b, num_heads, num_tokens, seq_len) 的注意力分数张量对齐

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # queries: (b, num_heads, num_tokens, head_dim)
        # keys.transpose(2,3): (b, num_heads, head_dim, seq_len)
        # 相乘后 attn_scores: (b, num_heads, num_tokens, seq_len)
        # —— 即“新 token”对“全部历史+当前 token”的注意力打分

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(causal_mask, -torch.inf)
        # 把掩码中 True（未来位置）对应的分数填为 -inf，softmax 后趋近于 0

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 缩放点积注意力：除以 sqrt(head_dim) 做缩放，再在最后一维（seq_len）做 softmax
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # attn_weights: (b, num_heads, num_tokens, seq_len)
        # values: (b, num_heads, seq_len, head_dim)
        # 相乘后: (b, num_heads, num_tokens, head_dim)，再转置回
        # (b, num_tokens, num_heads, head_dim)，方便下面合并多头

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 把多头拼回到一起：(b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 输出投影层，融合多头信息

        return context_vec, next_cache
        # 返回本次（仅新 token 部分的）注意力输出，以及供下一步使用的 KV 缓存


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）模块。

    对最后一维（特征维）做均值方差归一化，并通过可学习的
    scale（缩放）和 shift（平移）参数还原表达能力。
    与 KV Cache 机制无直接关系，仅作为 TransformerBlock 的组成部分。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习缩放参数 γ
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习平移参数 β

    def forward(self, x):
        """
        参数：
            x (Tensor): 任意形状，最后一维为 emb_dim。
        返回：
            归一化后的张量，形状与输入相同。
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    GELU（Gaussian Error Linear Unit）激活函数的 tanh 近似实现。
    用于前馈网络（FeedForward）中的非线性激活。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        参数：
            x (Tensor): 任意形状的输入张量。
        返回：
            对 x 逐元素应用 GELU 激活后的张量，形状与输入相同。
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Transformer 中的前馈网络（Position-wise Feed-Forward Network）。

    结构为：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)。
    对序列中每个位置独立、相同地进行变换，与 KV Cache 无关（不涉及跨 token 的注意力）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        参数：
            x (Tensor): 形状 (b, num_tokens, emb_dim)。
        返回：
            形状与输入相同的张量 (b, num_tokens, emb_dim)。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    单个 Transformer 块：由“多头自注意力 + 残差连接”和
    “前馈网络 + 残差连接”两个子层组成，均带前置 LayerNorm（Pre-LN 结构）。

    透传 use_cache / start_pos / cache 参数给内部的 MultiHeadAttention，
    使得 KV Cache 能够按层（layer-wise）独立维护。
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False, start_pos=0, cache=None):
        """
        参数：
            x (Tensor): 输入张量，形状 (b, num_tokens, emb_dim)。
            use_cache (bool): 是否启用 KV Cache，透传给内部注意力模块。
            start_pos (int): 当前 token 在完整序列中的起始位置，透传给注意力模块。
            cache (tuple or None): 本层的 (keys, values) 历史缓存，透传给注意力模块。

        返回：
            x (Tensor): 该 Transformer 块的输出，形状 (b, num_tokens, emb_dim)。
            next_cache (tuple or None): 本层注意力更新后的 KV 缓存
                （拼接了历史与当前 token 的 keys/values），供上层保存后
                在下一次增量前向时传回本层。
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, use_cache=use_cache, start_pos=start_pos, cache=cache) # Shape [batch_size, num_tokens, emb_size]
        # 注意力子层：先归一化，再做（带 KV Cache 的）多头自注意力，
        # next_cache 是本层更新后的 K/V 缓存，会被逐层向上传递保存
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 残差连接：把注意力输出加回原始输入

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 前馈子层同样使用 Pre-LN + 残差连接结构，与 KV Cache 无关

        return x, next_cache


class GPTModel(nn.Module):
    """
    完整的 GPT-2 风格语言模型，支持基于 KV Cache 的增量推理。

    整体结构：Token Embedding + 位置 Embedding -> Dropout ->
              N 层 TransformerBlock（每层可选启用 KV Cache）->
              最终 LayerNorm -> 输出线性层（映射到词表维度的 logits）。

    关键状态：
        self.current_pos (int): 记录“下一批输入 token”在完整序列中的
            起始绝对位置。每次启用缓存的前向调用后会自动累加，
            用于正确计算位置编码（position embedding）在增量解码时的偏移。
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.current_pos = 0
        # 初始化“当前生成位置”计数器为 0，表示尚未开始增量生成；
        # 该计数器仅在 use_cache=True 时才会被更新和使用

    def forward(self, in_idx, use_cache=False, cache=None):
        """
        参数：
            in_idx (LongTensor): 输入 token id 序列，形状 (batch_size, seq_len)。
                - 首次 prefill 阶段：seq_len 通常等于完整 prompt 长度。
                - 增量解码阶段：seq_len 通常等于 1（每步只输入新生成的一个 token）。
            use_cache (bool): 是否启用 KV Cache（若 cache 不为 None 也会被
                隐式视为启用）。
            cache (KVCache or None): 外部维护的、跨所有 Transformer 层的
                KV 缓存容器对象（定义见 .utils.KVCache），提供 get(i)/update(i, ...)
                接口按层存取每一层的 (keys, values)。

        返回：
            logits (Tensor): 输出的下一 token 预测分数，
                形状 (batch_size, seq_len, vocab_size)。
        """
        use_cache = use_cache or cache is not None
        # 只要传入了 cache 对象，即便 use_cache 显式传 False，也视为启用缓存模式

        batch_size, seq_len = in_idx.shape
        start_pos = self.current_pos if use_cache else 0
        # 关键：位置管理（position management）。
        # 若启用缓存，则本批 token 的绝对起始位置 = 之前已经处理过的 token 总数
        # （即 self.current_pos，随每次调用累加）；
        # 若不启用缓存（如普通训练/一次性前向），起始位置固定为 0。
        pos = torch.arange(start_pos, start_pos + seq_len, device=in_idx.device)
        # 生成本批 token 对应的绝对位置索引序列，形状 (seq_len,)，
        # 例如增量解码时 seq_len=1，则 pos 只包含当前这一个新位置
        tok_embeds = self.tok_emb(in_idx)   # 形状 (batch_size, seq_len, emb_dim)
        pos_embeds = self.pos_emb(pos)      # 形状 (seq_len, emb_dim)，广播到 batch 维
        x = self.drop_emb(tok_embeds + pos_embeds)
        # 词向量与位置向量相加后做 dropout，得到 Transformer 的输入嵌入

        if use_cache:
            self.current_pos += seq_len
            # 关键：位置计数器的“写入”。启用缓存时，把本次处理的 token 数量
            # 累加到 current_pos 上，使得下一次调用时 start_pos 能正确
            # 指向“紧接着上次末尾”的下一个绝对位置。

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            # 关键：KV Cache 的“按层读取”。从外部 KVCache 容器中取出
            # 第 i 层此前保存的 (keys, values)，若容器为空（首次调用）或
            # 该层尚无缓存，则 blk_cache 为 None。
            x, new_cache = block(x, use_cache=use_cache, start_pos=start_pos, cache=blk_cache)
            # 该 Transformer 块内部会把 blk_cache 与本次新算出的 K/V
            # 在序列长度维上拼接，返回拼接后的 new_cache
            if cache:
                cache.update(i, new_cache)
                # 关键：KV Cache 的“按层写入”。把该层更新后的
                # （历史+当前，拼接后的）K/V 写回外部 KVCache 容器，
                # 供下一次增量前向调用时通过 cache.get(i) 取出

        x = self.final_norm(x)
        logits = self.out_head(x)
        # 最终归一化后，通过输出线性层映射到词表维度，得到每个位置的下一 token 分布分数
        return logits

    def reset_kv_cache(self):
        """
        重置增量生成的位置计数器。

        通常在开始一轮全新的生成（例如换了新的 prompt）之前调用，
        将 self.current_pos 归零，避免沿用上一轮生成遗留的位置偏移。
        注意：此方法本身不清空外部传入的 KVCache 容器内容，
        清空缓存数据本身由 KVCache 对象自身的方法负责（定义在 .utils 中）。
        """
        self.current_pos = 0
