# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文文档字符串（新增，原文件无模块级 docstring，仅有上方版权注释，予以保留）。

本模块是 Qwen3 大语言模型的“批量（batched）KV 缓存”版本实现。

与非批量版本相比，本文件的核心差异在于：
1. **批量 KV 缓存（batched KV cache）**：KV 缓存以 `KVCache`（定义在 `.utils` 中）为容器，
   按 `(layer_idx, batch_idx)` 分别存取每一层、每一个 batch 样本自己的 key/value 缓存，
   而不是像单样本版本那样对整个 batch 用同一份缓存张量。
2. **每样本位置管理（per-sample position management）**：由于批量生成时，batch 内不同样本
   可能已经消耗了不同长度的 KV 缓存（例如不同样本的 prompt 长度不同，或者某些样本已提前结束），
   所以每个样本的“当前位置”`start_pos` 是一个形状为 `(batch_size,)` 的张量，而不是一个标量。
   这会影响到：RoPE 位置编码要按样本分别取用 cos/sin，以及因果注意力 mask 也要按样本单独构造。
3. **分组查询注意力（GQA, Grouped Query Attention）**：多个 query 头共享同一组 key/value 头，
   以减少 KV 缓存的显存占用；本文件中通过 `repeat_interleave` 把 KV 头“广播”到与 query 头数量一致。
4. **旋转位置编码（RoPE, Rotary Positional Embedding）**：对 query/key 张量在 head_dim 维度上
   做旋转变换来注入位置信息；批量版本中通过 `offset`（即每个样本的 `start_pos`）为每个样本单独计算
   其对应的位置索引，再取出对应的 cos/sin 值。

下面在类和函数层面、以及关键代码行上，补充了详细的中文注释，帮助理解各处张量的形状变化。
所有原始英文注释均予以保留，本次修改只新增中文注释/文档字符串，不改动任何可执行代码。
"""

from .utils import KVCache   # noqa: F401
from ..qwen3 import (   # noqa: F401
    QWEN_CONFIG_06_B, QWEN3_CONFIG_1_7B, QWEN3_CONFIG_4B,
    QWEN3_CONFIG_8B, QWEN3_CONFIG_14B, QWEN3_CONFIG_32B,
    Qwen3Tokenizer, load_weights_into_qwen,
    download_from_huggingface,
    download_from_huggingface_from_snapshots
)
# 上面从 `.utils` 导入 `KVCache`：这是本文件使用的“按 (层, batch 样本) 索引”的批量 KV 缓存容器类。
# 从上一级包的 `..qwen3` 模块导入：各种规模的 Qwen3 配置字典（如 QWEN3_CONFIG_1_7B 等）、
# 分词器 `Qwen3Tokenizer`、权重加载函数 `load_weights_into_qwen`，
# 以及从 HuggingFace 下载权重/快照的两个工具函数。这些均直接复用非批量版本中的定义，无需重复实现。

import torch
import torch.nn as nn


class Qwen3Model(nn.Module):
    """
    Qwen3 语言模型（支持批量 KV 缓存版本）。

    整体结构与标准 Transformer 解码器一致：
        token 嵌入 -> N 层 TransformerBlock（含分组查询注意力 + 前馈网络）-> 最终 RMSNorm -> 输出线性层（语言模型头）

    与非批量版本的关键区别：
        - `forward` 接受一个 `cache`（`KVCache` 实例）与 `start_pos`（形状为 `(batch_size,)` 的张量，
          表示 batch 中每个样本当前已经生成/缓存到的位置），从而支持一个 batch 内、各样本处于不同
          生成进度时仍能正确地做因果注意力与 RoPE 位置编码。
        - `current_pos` 用于在外部（如生成循环）跟踪每个样本的当前位置，是一个形状为 `(batch_size,)` 的张量。
    """

    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        # 主体模型参数
        # token 嵌入层：把输入的 token id 映射为形状 (vocab_size, emb_dim) 的向量表中的一行，
        # 输出张量形状为 (batch_size, seq_len, emb_dim)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            # 使用 ModuleList 而非 Sequential，因为每个 TransformerBlock 的 forward 需要多个输入
            # （隐藏状态 x、注意力 mask、RoPE 的 cos/sin），Sequential 只支持单一输入的链式传递。
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # 最终归一化层：在输出到语言模型头之前，对最后一层的隐藏状态做 RMSNorm
        self.final_norm = RMSNorm(cfg["emb_dim"])
        # 语言模型头（输出投影）：把隐藏状态 (batch_size, seq_len, emb_dim) 映射为
        # 词表维度的 logits，形状为 (batch_size, seq_len, vocab_size)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        # 可复用的工具（预计算好、注册为 buffer，避免每次前向都重新计算）
        if cfg["head_dim"] is None:
            # 若配置未显式指定每个注意力头的维度，则用嵌入维度除以头数得到
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            # 否则直接使用配置中指定的 head_dim（Qwen3 中 head_dim 可能与 emb_dim/n_heads 不一致）
            head_dim = cfg["head_dim"]
        # 预先计算好 RoPE 所需的 cos/sin 表，形状均为 (context_length, head_dim)
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        # 注册为非持久化 buffer：会随模型 .to(device) 移动，但不会被保存进 state_dict
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = None  # Batched version tracks positions per sample
        # 批量版本中，current_pos 用于跟踪 batch 内“每个样本”各自的当前位置（生成到第几个 token），
        # 形状为 (batch_size,)，而不是单样本版本里的一个标量。

    def forward(self, in_idx, cache=None, start_pos=None):
        """
        前向传播。

        参数：
            in_idx: 形状 (B, num_tokens) 的输入 token id 张量。
            cache: 可选的 `KVCache` 实例，按 (layer_idx, batch_idx) 存取每层每个样本的 (key, value) 缓存。
                   若为 None，表示不使用 KV 缓存（例如训练或一次性全量前向）。
            start_pos: 形状 (B,) 的张量，表示 batch 内每个样本在本次前向之前，
                       其 KV 缓存中已经累积的 token 数量（即本次新 token 的起始位置）。
                       仅在 cache 不为 None 时使用。

        返回：
            logits: 形状 (B, num_tokens, vocab_size) 的语言模型输出 logits。
        """
        B, num_tokens = in_idx.size()
        # tok_embeds 形状: (B, num_tokens, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        device = x.device

        if cache is not None:
            # 使用 KV 缓存的增量解码分支：batch 内每个样本可能已经缓存了不同长度的历史 token，
            # 因此需要按样本分别构造因果注意力 mask。
            pos_start = start_pos
            # pos_end[i] 表示样本 i 在拼接完本次新 token 之后，其序列的总长度（历史 + 新增）
            pos_end = pos_start + num_tokens
            # max_len 为 batch 中最长的“历史+新增”长度，用于构造一个足够大的全量下三角 mask
            max_len = pos_end.max().item()
            # full_mask 形状: (max_len, max_len)，上三角（不含对角线）为 True，表示这些位置需要被屏蔽
            # （即因果 mask：每个位置只能看到自己及之前的位置）
            full_mask = torch.triu(
                torch.ones(max_len, max_len, device=device, dtype=torch.bool), diagonal=1
            )
            # mask 形状: (B, 1, num_tokens, max_len)，中间的 1 是为了可以广播到多头注意力的头维度
            mask = torch.zeros(B, 1, num_tokens, max_len, device=device, dtype=torch.bool)
            for i in range(B):
                # 针对每个样本，从 full_mask 中切出对应它自己的位置区间：
                # 行取 [ps, pe) 表示本次新增的 num_tokens 个查询位置；
                # 列取 [0, pe) 表示该样本目前累计的所有 key 位置（历史缓存 + 新增）。
                # 这正是“每样本位置管理”的体现：不同样本的 (ps, pe) 可能不同。
                ps, pe = pos_start[i].item(), pos_end[i].item()
                mask[i, 0] = full_mask[ps:pe, :pe]
        else:
            # 不使用 KV 缓存的分支（例如一次性对完整序列做前向）：所有样本从位置 0 开始
            pos_start = torch.zeros(B, dtype=torch.long, device=device)
            # mask 形状: (1, 1, num_tokens, num_tokens)，标准因果 mask，会广播到 batch 维度和头维度
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=device, dtype=torch.bool), diagonal=1
            )[None, None, :, :]

        for i, block in enumerate(self.trf_blocks):
            # 从批量 KV 缓存中取出第 i 层、每个 batch 样本各自的 (key, value) 缓存列表；
            # blk_cache 是一个长度为 B 的 list，每个元素是该样本在第 i 层的缓存（可能为 None，表示还没有缓存）
            blk_cache = [cache.get(i, b_idx) for b_idx in range(B)] if cache is not None else None
            # 将当前隐藏状态、mask、RoPE 的 cos/sin、每样本起始位置、每样本缓存传入 TransformerBlock
            x, new_blk_cache = block(x, mask, self.cos, self.sin, start_pos=pos_start, cache=blk_cache)
            if cache is not None:
                # 将该层每个样本更新后的 (key, value) 缓存写回批量 KV 缓存容器中
                for b_idx in range(B):
                    cache.update(i, b_idx, new_blk_cache[b_idx])
        # x 形状: (B, num_tokens, emb_dim)
        x = self.final_norm(x)
        # logits 形状: (B, num_tokens, vocab_size)；先将隐藏状态转换回配置的目标 dtype 再做输出投影
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self, batch_size, device=None):
        """
        重置（初始化）用于跟踪 batch 内每个样本当前生成位置的 `current_pos`。

        参数：
            batch_size: 当前 batch 的样本数量。
            device: 张量所在设备；若不指定，则使用模型参数所在设备。

        返回：
            无返回值；直接设置 `self.current_pos` 为形状 (batch_size,) 的全零张量，
            表示所有样本的生成位置从 0 开始。
        """
        device = device or next(self.parameters()).device
        self.current_pos = torch.zeros(batch_size, dtype=torch.long, device=device)


class TransformerBlock(nn.Module):
    """
    单个 Transformer 解码器块，由「分组查询注意力子层」和「前馈网络子层」组成，
    每个子层都采用 Pre-Norm（先 RMSNorm 再子层）+ 残差连接的结构。
    """

    def __init__(self, cfg):
        super().__init__()
        # 分组查询注意力（GQA）子层：多个 query 头共享较少数量的 key/value 头组
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        # 前馈网络子层（SwiGLU 结构）
        self.ff = FeedForward(cfg)
        # 注意力子层前的归一化
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        # 前馈网络子层前的归一化
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        """
        参数：
            x: 输入隐藏状态，形状 (B, num_tokens, emb_dim)。
            mask: 因果注意力 mask，形状 (B, 1, num_tokens, kv_total_len) 或 (1, 1, num_tokens, num_tokens)。
            cos, sin: RoPE 预计算表，形状 (context_length, head_dim)。
            start_pos: 形状 (B,) 的张量，每个样本在 KV 缓存中的起始位置（用于 RoPE 位置偏移）。
            cache: 长度为 B 的列表，每个元素是该样本在本层的 (key, value) 缓存（或 None）。

        返回：
            x: 更新后的隐藏状态，形状 (B, num_tokens, emb_dim)。
            next_cache: 更新后的、长度为 B 的 (key, value) 缓存列表。
        """
        # Shortcut connection for attention block
        # 注意力子层的残差连接的“捷径”分支
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)  # Shape [batch_size, num_tokens, emb_size]
        # 中文：att 的输出形状为 [batch_size, num_tokens, emb_size]，与输入 x 的形状一致，可直接相加做残差
        x = x + shortcut  # Add the original input back
        # 中文：将注意力子层的输出与原始输入相加，完成残差连接

        # Shortcut connection for feed-forward block
        # 前馈网络子层的残差连接的“捷径”分支
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back
        # 中文：将前馈网络子层的输出与其输入相加，完成残差连接

        return x, next_cache


class FeedForward(nn.Module):
    """
    前馈网络（FFN）子层，采用 SwiGLU 门控结构：
        输出 = fc3( SiLU(fc1(x)) * fc2(x) )
    其中 fc1、fc2 将维度从 emb_dim 投影到 hidden_dim（两路并行，一路做门控激活，一路作为线性分支），
    fc3 再将维度从 hidden_dim 投影回 emb_dim。
    """

    def __init__(self, cfg):
        super().__init__()
        # fc1、fc2：emb_dim -> hidden_dim 的线性投影（无偏置）
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # fc3：hidden_dim -> emb_dim 的线性投影（无偏置），将门控结果映射回原始维度
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        """
        参数：
            x: 形状 (B, num_tokens, emb_dim) 的输入张量。
        返回：
            形状 (B, num_tokens, emb_dim) 的输出张量。
        """
        # x_fc1, x_fc2 形状均为 (B, num_tokens, hidden_dim)
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        # SiLU(x_fc1) 作为门控信号，与 x_fc2 逐元素相乘，结果形状仍为 (B, num_tokens, hidden_dim)
        x = nn.functional.silu(x_fc1) * x_fc2
        # 投影回 emb_dim，最终输出形状 (B, num_tokens, emb_dim)
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（Grouped Query Attention, GQA），并支持批量 KV 缓存的增量解码。

    GQA 核心思想：将 num_heads 个 query 头划分为 num_kv_groups 组，每组内的多个 query 头
    共享同一对 key/value 头，从而大幅减少 KV 缓存所需存储的 key/value 头数量
    （相比标准多头注意力 MHA，每个 query 头都有独立的 key/value）。

    本类同时负责：
        - 对 query/key 应用 RoPE 位置编码（按每样本的 start_pos 做位置偏移）；
        - 维护/更新每个 batch 样本自己的 KV 缓存（增量解码场景下，将新 key/value 与历史缓存拼接）；
        - 将 key/value 头通过 repeat_interleave 广播到与 query 头数量一致，以执行标准的缩放点积注意力。
    """

    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        # 断言：query 头数必须能被 kv 组数整除，才能均匀分组

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        # group_size：每个 kv 组对应多少个 query 头（即需要将每个 kv 头重复多少次才能与 query 头对齐）
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        # d_out：所有 query 头拼接后的总维度
        self.d_out = num_heads * head_dim

        # W_query：将输入投影为所有 query 头拼接后的向量，输出维度 num_heads * head_dim
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        # W_key、W_value：只投影出 num_kv_groups 组的 key/value（远小于 num_heads），这是 GQA 节省显存的关键
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        # 输出投影：将多头注意力拼接后的结果 (num_heads * head_dim) 投影回模型维度 d_in
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            # Qwen3 特有的 QK-Norm：对每个头的 query/key 向量（维度为 head_dim）分别做 RMSNorm，
            # 有助于训练稳定性
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        """
        参数：
            x: 输入隐藏状态，形状 (b, num_tokens, d_in)。
            mask: 因果注意力 mask。
            cos, sin: RoPE 预计算表，形状 (context_length, head_dim)。
            start_pos: 形状 (b,) 的张量，batch 内每个样本的位置偏移量（用于 RoPE 与缓存拼接）。
            cache: 长度为 b 的列表，每个元素是该样本此前缓存的 (key, value) 元组，或 None（无历史缓存）。

        返回：
            输出张量，形状 (b, num_tokens, d_in)；
            next_cache：更新后的、长度为 b 的 (key, value) 缓存列表，每个 key/value 形状为
                        (1, num_kv_groups, total_seq_len_i, head_dim)（total_seq_len_i 为该样本累计长度）。
        """
        b, num_tokens, _ = x.shape

        # Apply projections
        # 应用线性投影
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        # 重塑形状：将“头数 * 头维度”拆分为独立的头维度，并把头维度换到第 2 维，方便按头做矩阵乘法
        # queries: (b, num_tokens, num_heads, head_dim) -> transpose -> (b, num_heads, num_tokens, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # keys/values: (b, num_tokens, num_kv_groups, head_dim) -> transpose -> (b, num_kv_groups, num_tokens, head_dim)
        # 注意这里的头数是 num_kv_groups（远小于 num_heads），这正是 GQA 节省 KV 缓存显存的地方
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        # 可选的 QK-Norm：对最后一维（head_dim）做归一化，不改变张量形状
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        # 应用旋转位置编码：offset=start_pos 为形状 (b,) 的每样本位置偏移，
        # 使得批量内不同样本能够使用各自正确的绝对位置来计算 RoPE 角度
        # （详见下方 apply_rope 函数中的形状变化说明）
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys = apply_rope(keys, cos, sin, offset=start_pos)

        # KV caching
        # 批量 KV 缓存的核心逻辑：因为 batch 内每个样本可能有不同长度的历史缓存，
        # 所以无法简单地对整个 batch 做一次 torch.cat，而是要逐样本处理再拼回 batch 维度。
        next_cache = []
        for i in range(b):
            # 取出样本 i 此前的缓存（若存在），cache 是长度为 b 的列表
            prev = cache[i] if cache else None
            if prev is None:
                # 没有历史缓存（例如第一次前向）：直接用本次新计算的 key/value 作为缓存起点
                # keys[i:i+1] 形状: (1, num_kv_groups, num_tokens, head_dim)（保留 batch 维为 1，便于后续 cat）
                k_cat = keys[i:i+1]
                v_cat = values[i:i+1]
            else:
                # 有历史缓存：沿序列长度维（dim=2）将历史 key/value 与本次新 key/value 拼接
                prev_k, prev_v = prev
                # k_cat/v_cat 形状: (1, num_kv_groups, prev_len + num_tokens, head_dim)
                k_cat = torch.cat([prev_k, keys[i:i+1]], dim=2)
                v_cat = torch.cat([prev_v, values[i:i+1]], dim=2)
            next_cache.append((k_cat, v_cat))

        # 将逐样本处理后的 key/value 沿 batch 维度重新拼接回一个完整的 batch 张量
        # keys/values 形状: (b, num_kv_groups, max_total_len, head_dim)
        # 注意：这里假设本次调用中各样本拼接后的序列长度是一致的（由外部 mask 构造逻辑保证对齐），
        # 否则 torch.cat 会因为形状不一致而报错。
        keys = torch.cat([k for k, _ in next_cache], dim=0)
        values = torch.cat([v for _, v in next_cache], dim=0)

        # Expand K and V to match number of heads
        # 将 K、V 从 num_kv_groups 个头广播（重复）到 num_heads 个头，
        # 每个 kv 头被连续重复 group_size 次，以便与对应的一组 query 头对齐
        # 广播后 keys/values 形状: (b, num_heads, max_total_len, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        # 缩放点积注意力
        # attn_scores 形状: (b, num_heads, num_tokens, max_total_len)
        attn_scores = queries @ keys.transpose(2, 3)
        # 应用因果 mask：mask 中为 True 的位置（未来位置/无效位置）被置为 -inf，softmax 后趋近于 0
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        # PyTorch fails to do the implicit casting, so we have to be intentional with the types
        # 中文：PyTorch 无法自动完成隐式类型转换，因此这里显式构造一个与 queries 同 dtype/device 的缩放因子张量，
        # 避免因 dtype 不匹配（如混合精度）导致的运算错误
        scale = torch.tensor(self.head_dim**0.5, dtype=queries.dtype, device=queries.device)
        # attn_weights 形状: (b, num_heads, num_tokens, max_total_len)，最后转换为 values 的 dtype 以便后续矩阵乘法
        attn_weights = torch.softmax(attn_scores / scale, dim=-1).to(values.dtype)

        # context: (b, num_heads, num_tokens, head_dim) -> transpose(1,2) -> (b, num_tokens, num_heads, head_dim)
        # -> reshape -> (b, num_tokens, d_out)，即把多头结果重新拼接回一个向量
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        # 最终通过输出投影，将维度从 d_out 映射回 d_in；同时返回更新后的每样本 KV 缓存列表
        return self.out_proj(context), next_cache


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """
    预计算 RoPE（旋转位置编码）所需的 cos / sin 查找表。

    参数：
        head_dim: 每个注意力头的维度，必须为偶数（RoPE 需要将维度两两配对做旋转）。
        theta_base: RoPE 频率的基数（通常记为 theta 或 rope_base），控制不同维度对应的旋转频率。
        context_length: 预计算的最大位置数量（表的行数）。
        dtype: 计算所用的数据类型。

    返回：
        cos, sin: 形状均为 (context_length, head_dim) 的张量，
                  分别是每个位置、每个维度对应旋转角度的余弦值和正弦值。
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    # 计算逆频率（频率随维度指数衰减），inv_freq 形状: (head_dim // 2,)
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    # 生成位置索引序列，positions 形状: (context_length,)
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    # 计算每个位置、每个频率对应的旋转角度：外积得到 (context_length, head_dim // 2)
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    # 将角度沿最后一维复制一份并拼接，使维度从 head_dim // 2 扩展到 head_dim，
    # 这样前半部分与后半部分维度对应相同的角度，便于后续“旋转一半”的实现方式
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    # 预先计算好正弦、余弦值，避免每次前向重复计算，均为 (context_length, head_dim)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset):
    """
    对输入张量 x（query 或 key）应用旋转位置编码（RoPE），支持“每样本不同位置偏移”。

    参数：
        x: 形状 (batch_size, num_heads, seq_len, head_dim) 的 query 或 key 张量。
        cos, sin: 由 `compute_rope_params` 预计算得到的表，形状均为 (context_length, head_dim)。
        offset: 形状 (batch_size,) 的张量，表示 batch 内每个样本的位置偏移量
                （即该样本在 KV 缓存中已有多少个历史 token，本次新 token 的绝对位置
                 从 offset[i] 开始）。这是实现“每样本位置管理”的关键参数。

    返回：
        x_rotated: 形状与输入 x 相同，(batch_size, num_heads, seq_len, head_dim)，
                   是应用了旋转位置编码之后的结果。
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    bsz, n_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    assert offset.shape[0] == bsz, "Offset must have one value per batch item"
    # 中文：offset 必须与 batch 大小一致，即每个样本都要有自己对应的一个位置偏移值

    # Prepare cos/sin: (seq_len, head_dim)
    # 中文：这里的切片 cos[:cos.shape[0], :] 实际上取的是完整的 cos 表（自身长度），
    # 再增加两个大小为 1 的维度，得到 (1, 1, total_seq_len, head_dim)，方便后续按位置索引 gather
    cos = cos[:cos.shape[0], :].unsqueeze(0).unsqueeze(0)  # (1, 1, total_seq_len, head_dim)
    sin = sin[:sin.shape[0], :].unsqueeze(0).unsqueeze(0)

    # Build position indices per batch item
    # 为 batch 内每个样本构造各自的绝对位置索引：
    # torch.arange(seq_len) 形状 (seq_len,) 表示本次新增 token 在“本次序列内部”的相对位置 0..seq_len-1，
    # 加上 offset（形状 (bsz,)，广播为 (bsz, 1)）得到每个样本的绝对位置
    # position_ids 最终形状: (bsz, seq_len)，即“每样本位置管理”在 RoPE 中的具体体现
    position_ids = torch.arange(seq_len, device=offset.device).unsqueeze(0) + offset.unsqueeze(1)  # (bsz, seq_len)
    # 防止位置索引超出预计算表的最大长度（context_length - 1），做截断保护
    position_ids = position_ids.clamp(max=cos.shape[2] - 1)

    # Gather cos/sin for each position
    # 根据每个样本各自的位置索引，从 cos/sin 表中取出对应位置的值
    # 结果形状: (bsz, seq_len, head_dim)——每个样本、每个 token 位置都有自己对应的一组 cos/sin 值
    cos = cos[0, 0, position_ids, :]  # (bsz, seq_len, head_dim)
    sin = sin[0, 0, position_ids, :]

    # Expand for multi-heads
    # 增加头维度（大小为 1，后续可广播到 n_heads），使形状与 x 对齐以便逐元素相乘
    cos = cos.unsqueeze(1)  # (bsz, 1, seq_len, head_dim)
    sin = sin.unsqueeze(1)

    # 将 head_dim 一分为二：x1 为前半部分，x2 为后半部分，各自形状 (bsz, n_heads, seq_len, head_dim // 2)
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    # 构造“旋转后”的向量：将后半部分取负号放到前面，前半部分放到后面，
    # 即标准 RoPE 实现中的 rotate_half 操作，形状与 x 相同
    rotated = torch.cat((-x2, x1), dim=-1)
    # RoPE 的核心公式：x_rotated = x * cos + rotate_half(x) * sin，逐元素运算，形状保持不变
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated


class RMSNorm(nn.Module):
    """
    RMSNorm（Root Mean Square Layer Normalization）。

    与 LayerNorm 不同，RMSNorm 不做均值中心化，只用均方根（RMS）对输入做缩放归一化，
    计算量更小，是 Qwen3 等现代 LLM 中常用的归一化方式。
    """

    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        # 可学习的缩放参数，形状 (emb_dim,)，初始化为全 1（即初始时不改变归一化后的幅值）
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 可选的可学习偏置参数，形状 (emb_dim,)；若 bias=False 则不使用偏置
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        """
        参数：
            x: 任意形状，最后一维大小为 emb_dim 的张量（例如 (b, num_tokens, emb_dim)
               或 (b, num_heads, num_tokens, head_dim)）。
        返回：
            与输入形状相同的归一化结果张量。
        """
        input_dtype = x.dtype

        if self.qwen3_compatible:
            # 为与 Qwen3 官方实现保持数值一致，先转换为 float32 计算，避免低精度带来的数值误差
            x = x.to(torch.float32)

        # 沿最后一维计算均方值（不减均值），variance 形状: 与 x 相同但最后一维变为 1（keepdim=True）
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # 用均方根的倒数对 x 做缩放，实现归一化，形状与 x 相同
        norm_x = x * torch.rsqrt(variance + self.eps)
        # 应用可学习的缩放参数（在最后一维上逐元素相乘，利用广播机制）
        norm_x = norm_x * self.scale

        if self.shift is not None:
            # 若启用偏置，则在缩放之后再加上可学习偏置
            norm_x = norm_x + self.shift

        # 转换回输入原始的 dtype，保持与外部张量精度一致
        return norm_x.to(input_dtype)
