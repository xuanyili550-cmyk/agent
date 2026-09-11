# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明
================
本文件实现了带 **KV 缓存（KV Cache）** 的 Llama 3 模型（含 Llama 3.2 的 1B / 3B 配置），
以及配套的分词器（Tokenizer）和聊天格式化工具。核心技术点包括：

1. **RoPE（Rotary Position Embedding，旋转位置编码）**：不通过额外的位置向量编码位置信息，
   而是把位置信息编码成对 Query/Key 向量的"旋转角度"，通过旋转变换隐式地把相对位置信息
   注入到注意力得分中。本文件中 `compute_rope_params` 负责预计算旋转角度对应的 cos/sin
   表，`apply_rope` 负责把这套旋转变换实际应用到 Q/K 张量上。Llama 3.2 还引入了频率
   缩放（NTK-aware / YaRN 风格的分段缩放），用于扩展模型可处理的上下文长度。

2. **RMSNorm（Root Mean Square LayerNorm）**：LayerNorm 的简化版本，只做"除以均方根"的
   缩放归一化，不做均值中心化，也更省计算量。这里直接复用 PyTorch 内置的 `nn.RMSNorm`。

3. **GQA（Grouped-Query Attention，分组查询注意力）**：Query 头的数量多于 Key/Value 头的
   数量，多个 Query 头共享同一组 Key/Value，从而减少 KV 张量的显存占用与计算量（尤其在
   自回归推理阶段的 KV 缓存中收益明显）。`GroupedQueryAttention` 类中会先计算较少数量的
   K/V，再用 `repeat_interleave` 把它们"复制"到与 Query 头数一致，以执行标准的注意力计算。

4. **KV Cache（键值缓存）**：自回归生成时，每一步只有新 token 需要重新计算 Q/K/V，历史
   token 的 K/V 可以直接复用之前缓存的结果并做拼接（`torch.cat`），避免重复计算，把
   每步生成的复杂度从 O(n^2) 降低到 O(n)。本文件中 `Llama3Model.forward` 与
   `GroupedQueryAttention.forward` 都接收 `cache` 参数并维护 `current_pos`（当前已经
   生成到的位置），用于保证 RoPE 的位置编码在增量解码时依然正确对齐。

5. **SwiGLU（Swish-Gated Linear Unit）**：Llama 系列前馈网络（FeedForward）所使用的门控
   激活结构，用 SiLU（Swish）激活的一路和线性变换的另一路做逐元素相乘，再投影回原维度，
   相比普通的 ReLU-MLP 通常有更好的效果。对应 `FeedForward` 类中的
   `silu(fc1(x)) * fc2(x)` 再经 `fc3` 投影回 `emb_dim`。

此外，本文件末尾还提供了一套"高效版"实现（`GroupedQueryAttentionFast`、
`TransformerBlockFast`、`Llama3ModelFast`），使用 PyTorch 原生的
`scaled_dot_product_attention`（可在支持的硬件上自动启用 FlashAttention 等高效内核），
但该版本不支持 KV 缓存，主要用于全量前向（一次性处理整个序列）场景下的效率优化演示。

注意：本文件仅添加中文注释与文档字符串，不改动任何原始可执行代码逻辑。
"""

from .utils import KVCache   # noqa: F401  # 从同目录 utils 模块导入 KVCache 工具类（用于管理各层的 KV 缓存），noqa 是因为这里导入了但本文件不直接使用它（仅是为了暴露给外部调用者）

import os
from pathlib import Path

import torch
import torch.nn as nn
import tiktoken
from tiktoken.load import load_tiktoken_bpe


# Llama 3.2 - 1B 参数量模型的配置字典
LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,           # Vocabulary size  # 词表大小（tokenizer 能表示的 token 种类数）
    "context_length": 131_072,       # Context length that was used to train the model  # 训练时使用的最大上下文长度（也决定了 RoPE 预计算表的长度）
    "emb_dim": 2048,                 # Embedding dimension  # 词嵌入 / 隐藏层维度
    "n_heads": 32,                   # Number of attention heads  # 注意力头（Query 头）数量
    "n_layers": 16,                  # Number of layers  # Transformer block 堆叠层数
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward  # 前馈网络（SwiGLU）中间层维度
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention  # GQA 中 Key/Value 的组数（少于 n_heads，多个 Query 头共享一组 K/V）
    "rope_base": 500_000.0,          # The base in RoPE's "theta"  # RoPE 旋转角频率公式中的底数 theta_base，越大则低频分量周期越长
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage  # 使用 bfloat16 低精度类型以节省显存
    "rope_freq": {                   # RoPE frequency scaling  # RoPE 频率缩放配置（用于扩展上下文长度，Llama3.2 引入的分段缩放策略）
        "factor": 32.0,              # 整体缩放因子，用于把超出原始训练长度的高波长（低频）分量按此因子缩放频率
        "low_freq_factor": 1.0,      # 低频判定边界因子，决定哪些频率被视为"低频"从而需要缩放
        "high_freq_factor": 4.0,     # 高频判定边界因子，决定哪些频率被视为"高频"从而保持不变
        "original_context_length": 8192,  # 模型原始（未做长上下文扩展前）的训练上下文长度，用作波长的参考基准
    }
}

# Llama 3.2 - 3B 参数量模型的配置字典（结构同上，仅 emb_dim / n_heads / n_layers 更大）
LLAMA32_CONFIG_3B = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 131_072,       # Context length that was used to train the model
    "emb_dim": 3072,                 # Embedding dimension
    "n_heads": 24,                   # Number of attention heads
    "n_layers": 28,                  # Number of layers
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
    "rope_freq": {                   # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}


class Llama3Model(nn.Module):
    """Llama 3（带 KV 缓存版本）整体模型。

    结构：Token Embedding -> N 层 TransformerBlock（RMSNorm + GQA + SwiGLU 前馈）
    -> 最终 RMSNorm -> 输出线性层（映射到词表大小的 logits）。

    与"标准"实现的主要区别在于 `forward` 方法额外接收 `cache` 参数，用于支持
    自回归生成时的增量解码（每步只处理新 token，历史 Key/Value 从缓存中读取并拼接），
    并通过 `self.current_pos` 跟踪当前已生成到的绝对位置，从而保证 RoPE 的相位
    在增量解码过程中依然正确。

    Args:
        cfg (dict): 模型配置字典，例如 LLAMA32_CONFIG_1B / LLAMA32_CONFIG_3B，
            需包含 vocab_size、emb_dim、n_heads、n_layers、hidden_dim、
            n_kv_groups、rope_base、rope_freq、context_length、dtype 等字段。
    """
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        # 词嵌入层：把 token id（整数）映射为 emb_dim 维的稠密向量
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            # 使用 ModuleList 而非 Sequential，因为每个 TransformerBlock.forward 需要接收
            # 多个输入（x、mask、cos、sin、start_pos、cache），Sequential 只支持单输入透传
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最终归一化层（RMSNorm）：在输出投影之前对隐藏状态做一次归一化，稳定数值尺度
        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        # 输出投影层：把 emb_dim 维隐藏状态映射为 vocab_size 维的 logits（不使用 bias，Llama 风格）
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        # 预计算 RoPE 所需的 cos/sin 查找表（长度覆盖整个 context_length，
        # 后续每一层、每一步解码都直接从这张表里按位置切片使用，避免重复计算三角函数）
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],       # 单个注意力头的维度 = 总嵌入维度 / 头数
            theta_base=cfg["rope_base"],                      # RoPE 角频率公式的底数
            context_length=cfg["context_length"],             # 预计算表需要覆盖的最大序列长度
            freq_config=cfg["rope_freq"]                       # 频率缩放配置（用于长上下文外推）
        )
        # 注册为 buffer（非可训练参数，但会随 .to(device)/.to(dtype) 一起移动；
        # persistent=False 表示不写入 state_dict，因为这是可以重新计算得到的固定值）
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # Track current position in KV cache  # 记录 KV 缓存中已经写入的位置（即下一个新 token 的起始绝对位置）

    def forward(self, in_idx, cache=None):
        """前向传播（支持可选的 KV 缓存增量解码）。

        Args:
            in_idx (torch.LongTensor): 形状 (batch_size, num_tokens) 的 token id 序列。
                若使用 KV 缓存做增量解码，num_tokens 通常为 1（每次只喂入新 token）；
                若不使用缓存，则可以是完整的 prompt 序列。
            cache (KVCache | None): 各层 KV 缓存的容器。传入时会在每层读取上一步缓存的
                (keys, values) 并与当前新计算的 keys/values 拼接，然后写回缓存；
                不传时按普通的全量前向（无缓存）方式计算，且使用标准的因果掩码。

        Returns:
            torch.Tensor: 形状 (batch_size, num_tokens, vocab_size) 的 logits，
                数据类型为 cfg["dtype"]。
        """
        # 词嵌入：(batch_size, num_tokens) -> (batch_size, num_tokens, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        if cache is not None:
            # 增量解码模式：本次新 token 的位置区间是 [pos_start, pos_end)
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end  # 更新缓存位置指针，供下一次调用使用
            # 构造因果掩码：整体是 (pos_end, pos_end) 的上三角掩码（True 表示需要被屏蔽/看不到未来），
            # 再截取出本次新 token 对应的行 [pos_start:pos_end, :pos_end]，
            # 使得新 token 可以看到之前所有缓存位置以及自己范围内已生成的 token（因果关系）
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0  # Not strictly necessary but helps torch.compile  # 无缓存时起始位置固定为 0，这行对逻辑非必需，但有助于 torch.compile 做静态形状优化
            # 标准的下三角因果掩码：(num_tokens, num_tokens)，对角线以上为 True（禁止看到未来 token）
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        # Shape (1, 1, num_tokens, num_tokens) to broadcast across batch and heads
        # 增加 batch 和 head 两个维度（均为1，利用广播机制），使掩码可以直接与
        # 注意力得分张量 (batch, num_heads, num_tokens, num_tokens) 相加/相乘
        mask = mask[None, None, :, :]

        for i, block in enumerate(self.trf_blocks):
            # 取出第 i 层此前缓存的 (keys, values)，若未启用缓存则为 None
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,   # 告知本层当前新 token 的起始绝对位置，供 RoPE 计算正确的旋转角
                                     cache=blk_cache)
            if cache is not None:
                # 把本层拼接更新后的 (keys, values) 写回缓存容器，供下一次前向调用复用
                cache.update(i, new_blk_cache)

        # 最终归一化
        x = self.final_norm(x)
        # 输出投影到词表维度，得到每个位置对每个词的预测得分（logits）
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        """重置 KV 缓存位置指针（在开始一次全新的生成/推理会话前调用）。"""
        self.current_pos = 0


class TransformerBlock(nn.Module):
    """单个 Transformer 解码器块：Pre-Norm 结构的 GQA 自注意力 + SwiGLU 前馈网络。

    结构：
        x -> RMSNorm -> GroupedQueryAttention -> 残差相加
          -> RMSNorm -> FeedForward(SwiGLU)   -> 残差相加

    Args:
        cfg (dict): 模型配置字典（同 Llama3Model）。
    """
    def __init__(self, cfg):
        super().__init__()
        # 分组查询注意力子层
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        # SwiGLU 前馈网络子层
        self.ff = FeedForward(cfg)
        # 注意力子层前的 RMSNorm（Pre-Norm 结构）
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        # 前馈子层前的 RMSNorm
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        """前向传播。

        Args:
            x (torch.Tensor): 形状 (batch_size, num_tokens, emb_dim) 的输入隐藏状态。
            mask (torch.Tensor): 因果注意力掩码，形状可广播到 (batch, num_heads, num_tokens, total_kv_len)。
            cos, sin (torch.Tensor): RoPE 预计算的 cos/sin 表，形状 (context_length, head_dim)。
            start_pos (int): 当前这批新 token 的起始绝对位置（用于 RoPE 相位对齐及 KV 缓存位置计算）。
            cache (tuple | None): 该层此前缓存的 (keys, values)，用于增量解码时拼接。

        Returns:
            tuple[torch.Tensor, tuple]: (更新后的隐藏状态 x, 该层新的 (keys, values) 缓存)。
        """
        # Shortcut connection for attention block
        shortcut = x  # 保存残差连接的输入
        x = self.norm1(x)  # Pre-Norm：先归一化再进入注意力子层
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back  # 残差相加，缓解深层网络梯度消失问题

        # Shortcut connection for feed-forward block
        shortcut = x  # 保存前馈子层的残差输入
        x = self.norm2(x)  # Pre-Norm：先归一化再进入前馈子层
        x = self.ff(x)  # SwiGLU 前馈变换
        x = x + shortcut  # Add the original input back  # 残差相加

        return x, next_cache


class FeedForward(nn.Module):
    """SwiGLU 前馈网络（Llama 系列使用的门控前馈结构）。

    计算公式：FFN(x) = fc3( SiLU(fc1(x)) * fc2(x) )
    其中 fc1、fc2 把 emb_dim 投影到 hidden_dim（两路并行的线性变换，不共享参数），
    fc3 再把 hidden_dim 投影回 emb_dim。SiLU(fc1(x)) 起"门控"作用，
    逐元素地对 fc2(x) 的信息流进行调制，相比普通 MLP + ReLU 通常表达能力更强。

    Args:
        cfg (dict): 模型配置字典，需包含 emb_dim、hidden_dim、dtype。
    """
    def __init__(self, cfg):
        super().__init__()
        # 门控分支：emb_dim -> hidden_dim，后面接 SiLU 激活
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # 数值分支：emb_dim -> hidden_dim，不经过激活，直接与门控分支逐元素相乘
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # 输出投影：hidden_dim -> emb_dim，把门控后的结果映射回原始隐藏维度
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 形状 (batch_size, num_tokens, emb_dim)。

        Returns:
            torch.Tensor: 形状 (batch_size, num_tokens, emb_dim)。
        """
        x_fc1 = self.fc1(x)  # (b, n, hidden_dim) —— 门控分支的原始输出（未激活）
        x_fc2 = self.fc2(x)  # (b, n, hidden_dim) —— 数值分支输出
        x = nn.functional.silu(x_fc1) * x_fc2  # SwiGLU 核心：SiLU(fc1(x)) 逐元素门控 fc2(x)
        return self.fc3(x)  # 投影回 emb_dim 维度


class GroupedQueryAttention(nn.Module):
    """分组查询注意力（Grouped-Query Attention, GQA），支持 KV 缓存的增量解码。

    与标准多头注意力（MHA）的区别：Query 仍然使用 `num_heads` 个头，但 Key/Value
    只使用较少的 `num_kv_groups` 组（`num_kv_groups <= num_heads`，且
    `num_heads` 必须能整除 `num_kv_groups`）。多个相邻的 Query 头共享同一组
    Key/Value（通过 `repeat_interleave` 把 K/V 复制扩展到与 Query 头数一致后
    再做点积注意力），从而显著降低 KV 缓存所需的显存占用与相关计算量，
    是 Llama 3 等大模型在推理效率上的关键优化手段之一。

    Args:
        d_in (int): 输入特征维度（通常等于 emb_dim）。
        d_out (int): 输出特征维度（通常等于 emb_dim），必须能被 num_heads 整除。
        num_heads (int): Query 注意力头数。
        num_kv_groups (int): Key/Value 分组数，num_heads 必须是它的整数倍。
        dtype: 权重矩阵的数据类型（如 torch.bfloat16）。
    """
    def __init__(
            self, d_in, d_out, num_heads, num_kv_groups, dtype=None
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 保证每个头维度整除得到整数 head_dim
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"  # 保证每组 KV 能被相同数量的 Query 头均匀共享

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 单个注意力头的维度

        # Key/Value 投影：注意输出维度是 num_kv_groups * head_dim（远小于 d_out，因为组数少于头数），
        # 这正是 GQA 节省显存/计算的关键所在
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups  # 每一组 KV 被多少个 Query 头共享

        # Query 投影：输出维度是完整的 d_out = num_heads * head_dim
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        # 多头注意力输出后的融合投影层
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        """前向传播，支持可选的 KV 缓存拼接。

        Args:
            x (torch.Tensor): 形状 (batch_size, num_tokens, d_in) 的输入。
            mask (torch.Tensor): 因果掩码，可广播到 (b, num_heads, num_tokens, total_kv_len)。
            cos, sin (torch.Tensor): RoPE 的 cos/sin 查找表，形状 (context_length, head_dim)。
            start_pos (int): 当前新 token 批次在整个序列中的起始绝对位置，用于从 cos/sin
                表中取出正确偏移的旋转角，保证增量解码时位置编码的连续性。
            cache (tuple[torch.Tensor, torch.Tensor] | None): 该层之前缓存的
                (keys, values)，形状均为 (b, num_kv_groups, prev_len, head_dim)；
                若为 None 表示不使用缓存（全量前向）。

        Returns:
            tuple[torch.Tensor, tuple]:
                - context_vec: 形状 (batch_size, num_tokens, d_out) 的注意力输出。
                - next_cache: 拼接（或新建）后的 (keys, values)，供下一步复用。
        """
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)  # Query 投影，完整头数
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)  # Key 投影，只有少量分组
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)  # Value 投影，只有少量分组

        # Reshape
        # 拆分出多头维度并转置，使 head 维度提前到 dim=1，便于按头做批量矩阵乘法
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)      # -> (b, num_heads, num_tokens, head_dim)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)     # -> (b, num_kv_groups, num_tokens, head_dim)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2) # -> (b, num_kv_groups, num_tokens, head_dim)

        # Apply RoPE
        # 对本次新计算出的 Query / Key 应用旋转位置编码；注意 Value 不做 RoPE（RoPE 只作用于 Q/K 的点积相似度计算）。
        # offset=start_pos 确保在增量解码时，新 token 使用的是它在整个序列中的真实绝对位置对应的旋转角，而不是从 0 开始
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            # 增量解码：把历史缓存的 Key/Value 与本次新计算的 keys_new/values_new 沿序列长度维度（dim=2）拼接，
            # 得到覆盖全部历史 + 当前新 token 的完整 Key/Value
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
            next_cache = (keys, values)  # 更新后的缓存，将被写回外层的 KVCache 容器
        else:
            start_pos = 0  # reset RoPE  # 不使用缓存时不存在"历史位置偏移"的概念，重置为 0（仅用于下方局部变量，不影响上面已经应用过的 RoPE）
            keys, values = keys_new, values_new
            next_cache = (keys, values)  # 即便不使用外部缓存机制，也返回本次的 (keys, values)，调用方可自行决定是否使用

        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)
        # GQA 的关键步骤：把 num_kv_groups 组的 Key/Value 沿 head 维度（dim=1）重复扩展，
        # 使其数量与 Query 头数 num_heads 对齐，才能与所有 Query 头做逐头的点积注意力
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]
        # 中文说明：必须用 repeat_interleave（逐元素重复）而不是 repeat（整体平铺重复），
        # 这样才能保证第 0~group_size-1 个 Query 头对应第 1 组 KV，
        # 第 group_size~2*group_size-1 个 Query 头对应第 2 组 KV，以此类推，
        # 与 Query 头的排列顺序语义一致。

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (b, num_heads, num_tokens, num_tokens)
        # 计算 Q·K^T 得到注意力原始得分：(b, num_heads, num_tokens_q, total_kv_len)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head  # 每个头分别做点积

        # Use the mask to fill attention scores
        # 将掩码为 True（代表"未来位置"，不允许被看到）的位置填充为 -inf，softmax 后趋近于 0 权重
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # 缩放点积注意力：除以 sqrt(head_dim) 防止点积数值过大导致 softmax 梯度消失，再做 softmax 归一化得到注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim  # 断言确认最后一维确实是 head_dim（防御性检查）

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 用注意力权重对 Value 做加权求和，得到每个 Query 位置的上下文向量；随后转置回 (b, num_tokens, num_heads, head_dim) 便于合并多头
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 把多头拼接（reshape）回单一的 d_out 维向量
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection  # 最终的线性融合投影

        return context_vec, next_cache


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None, dtype=torch.float32):
    """预计算 RoPE（旋转位置编码）所需的 cos / sin 查找表。

    RoPE 的核心思想：不是把位置信息"加"到词向量上，而是把每一对相邻维度
    (x_{2i}, x_{2i+1}) 看作二维平面上的一个点，按照与位置 pos 成正比的角度
    对其进行旋转。这样两个位置 m、n 的 Query/Key 做点积时，其结果只依赖于
    相对位置 (m-n)，从而天然地把相对位置信息编码进了注意力得分中，且不需要
    额外的可学习位置向量。

    Llama 3.2 额外引入了"频率缩放"（NTK-aware / YaRN 风格的分段线性插值），
    使得模型在推理时可以处理远超训练时 `original_context_length` 的序列长度：
    - 波长很长（低频）的分量，按 `factor` 整体缩小频率（等价于拉伸周期），
      从而让原本会超出训练范围的位置也落在模型见过的旋转角范围内；
    - 波长很短（高频）的分量，保持原始频率不变（高频分量对绝对位置外推不敏感）；
    - 波长处于中间过渡区间的分量，使用 `smooth_factor` 做线性混合插值。

    Args:
        head_dim (int): 单个注意力头的维度，必须是偶数（因为要两两配对做旋转）。
        theta_base (float): RoPE 角频率公式中的底数 theta（值越大，频率衰减越慢）。
        context_length (int): 需要预计算的最大位置数（通常等于模型的 context_length）。
        freq_config (dict | None): 频率缩放配置，包含
            factor / low_freq_factor / high_freq_factor / original_context_length；
            若为 None 则不做频率缩放，使用标准 RoPE 公式。
        dtype: 中间计算使用的数据类型（默认 float32，保证三角函数计算精度）。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (cos, sin)，形状均为 (context_length, head_dim)，
            后续 apply_rope 会按需要的位置区间对其做切片使用。
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"  # RoPE 要求维度可以两两配对旋转，故必须为偶数

    # Compute the inverse frequencies
    # 计算逆频率（角速度）：inv_freq[i] = theta_base^(-2i/head_dim)，i = 0, 1, ..., head_dim/2 - 1
    # 维度越靠前（i 越小），频率越高（旋转越快）；维度越靠后，频率越低（旋转越慢），这与经典 Transformer 位置编码的思想一致
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments
    # 频率调整（Llama 3.2 长上下文外推所需的分段缩放策略），仅当传入 freq_config 时执行
    if freq_config is not None:
        # 波长阈值：波长 = 2π / 频率。波长越长代表旋转越慢（低频分量），越容易在长序列外推时失真
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        # 把每个频率分量换算成对应的波长，方便与上面两个阈值比较
        wavelen = 2 * torch.pi / inv_freq

        # 对于波长大于 low_freq_wavelen 的"低频"分量，将其频率整体除以 factor（即拉长周期，
        # 使其能覆盖更长的上下文范围）；其余分量暂时保持不变
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        # 计算平滑混合系数：用于在"低频阈值"与"高频阈值"之间的过渡区间做线性插值，
        # 避免频率缩放在阈值处发生突变
        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        # 对过渡区间的频率做"未缩放频率"与"缩放后频率"之间的线性插值混合
        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        # 判断哪些分量落在"中间过渡区"（既不算纯低频也不算纯高频）
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        # 对处于中间过渡区的分量，使用上面算出的平滑插值结果替换掉之前简单的 if/else 缩放结果
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama  # 最终得到融合了分段缩放策略的逆频率

    # Generate position indices
    # 生成位置索引 0, 1, ..., context_length - 1
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    # 外积：位置 × 频率，得到每个位置在每个频率分量上的旋转角度
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    # 把角度矩阵在最后一维复制拼接一份，使其维度从 head_dim/2 扩展到 head_dim，
    # 对应后面 apply_rope 中把 x 切成前后两半分别乘以相同的 cos/sin 值
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    # 预先计算好 cos/sin 值，存成查找表，避免在每次前向传播时重复计算三角函数
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    """把预计算的 RoPE 旋转变换应用到 Query 或 Key 张量上。

    实现的是"旋转矩阵"作用于向量的等价形式：把 head_dim 维向量拆成前半 x1 和
    后半 x2 两部分，构造 "旋转后的向量" rotated = (-x2, x1)，然后：
        x_rotated = x * cos + rotated * sin
    这等价于把 (x1, x2) 中每一对配对维度看作二维复数（或二维平面上的点），
    按照对应位置的角度做二维旋转（Complex 乘法 e^{iθ} 的展开形式）。

    Args:
        x (torch.Tensor): 形状 (batch_size, num_heads, seq_len, head_dim) 的
            Query 或 Key 张量。
        cos, sin (torch.Tensor): compute_rope_params 预计算得到的查找表，
            形状 (context_length, head_dim)。
        offset (int): 当前这段序列在整个上下文中的起始绝对位置（用于 KV 缓存
            增量解码场景下，从查找表中取出正确的旋转角切片）。

    Returns:
        torch.Tensor: 与输入 x 形状相同、已应用旋转位置编码的张量，
            数据类型转换回 x 原本的 dtype。
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"  # 再次确认维度可两两配对

    # Split x into first half and second half
    # 把最后一维（head_dim）从中间切成两半，分别对应二维旋转中的"实部"和"虚部"角色
    x1 = x[..., : head_dim // 2]  # First half  # 前半部分
    x2 = x[..., head_dim // 2:]  # Second half  # 后半部分

    # Adjust sin and cos shapes
    # 根据 offset 从预计算表中取出与当前序列位置对应的 cos/sin 切片，
    # 并扩展出 batch 和 head 两个维度（大小为1，利用广播机制），方便与 x 相乘
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    # 构造"旋转后"的向量：(-x2, x1)，这是二维旋转公式展开后 sin 分量对应的部分
    rotated = torch.cat((-x2, x1), dim=-1)
    # 旋转变换核心公式：x_rotated = x * cos(θ) + rotated * sin(θ)
    # 对每一对配对维度 (x_i, x_{i+d/2}) 而言，等价于标准二维旋转矩阵
    # [[cos, -sin], [sin, cos]] 作用在 (x_i, x_{i+d/2}) 上
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    # 旋转计算本身用较高精度（cos/sin 是 float32），完成后转换回原始 dtype（如 bfloat16）以节省显存/带宽
    return x_rotated.to(dtype=x.dtype)


##########################################
# Tokenizer
##########################################


class Llama3Tokenizer:
    """Thin wrapper around tiktoken that keeps track of Llama-3 special IDs."""
    # 中文说明：对 tiktoken BPE 分词器的一层轻量封装，额外维护 Llama 3 的特殊 token
    # （如文本起止符、对话角色分隔符等）及其固定 token id，供 ChatFormat 等上层逻辑使用。
    def __init__(self, model_path):
        """加载 tiktoken BPE 词表文件并构建 Encoding 对象。

        Args:
            model_path (str): Llama 3 官方 tokenizer 的 BPE 词表文件路径（如 tokenizer.model）。

        Raises:
            FileNotFoundError: 当 model_path 不是一个存在的文件时抛出。
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        # 加载可合并的 BPE 词表（rank 映射），这是 tiktoken 自定义 Encoding 所需的核心数据
        mergeable = load_tiktoken_bpe(model_path)

        # hard-coded from Meta's tokenizer.json
        # 硬编码的 Llama 3 特殊 token 及其固定 id（与 Meta 官方 tokenizer.json 保持一致）
        self.special = {
            "<|begin_of_text|>": 128000,   # 文本开始标记
            "<|end_of_text|>": 128001,     # 文本结束标记
            "<|start_header_id|>": 128006, # 对话角色头部开始标记（如 system/user/assistant）
            "<|end_header_id|>": 128007,   # 对话角色头部结束标记
            "<|eot_id|>": 128009,          # 单轮对话结束标记（end of turn）
        }
        # 补充填充剩余保留 token id（reserved），排除已在上面显式定义过的那几个 id，
        # 保证整个 128002~128257 区间内的 id 都能在 special 字典中找到对应名字
        self.special.update({f"<|reserved_{i}|>": 128002 + i
                             for i in range(256)
                             if 128002 + i not in self.special.values()})

        # 构建 tiktoken 的 Encoding 对象：使用 Llama 3 官方的正则切分模式（pat_str）
        # 将原始文本先按此正则粗切分成候选 token 片段，再结合 mergeable BPE 规则合并成最终 token
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"          # 英文缩写后缀（如 's, 't, 're 等），大小写不敏感
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"               # 可选的非字母数字前缀 + 连续字母序列（单词）
                    r"|\p{N}{1,3}"                              # 1~3 位连续数字
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"               # 可选空格 + 连续的非空白非字母数字符号（标点等）+ 可选换行
                    r"|\s*[\r\n]+"                              # 空白后跟一个或多个换行
                    r"|\s+(?!\S)"                               # 末尾的空白（后面不再跟非空白字符）
                    r"|\s+",                                    # 其余空白序列
            mergeable_ranks=mergeable,   # BPE 合并规则（token -> rank 的映射）
            special_tokens=self.special, # 上面定义的特殊 token 集合
        )

    def encode(self, text, bos=False, eos=False, **kwargs):
        """把文本编码为 token id 列表。

        Args:
            text (str): 待编码的原始文本。
            bos (bool): 是否在开头添加 `<|begin_of_text|>` 标记。
            eos (bool): 是否在结尾添加 `<|end_of_text|>` 标记。
            **kwargs: 透传给底层 tiktoken `Encoding.encode` 的额外参数（如 allowed_special）。

        Returns:
            list[int]: 编码得到的 token id 序列。
        """
        # 若 bos=True，先在最前面拼接开始符对应的 id；否则为空列表
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.model.encode(text)
        if eos:
            # 若 eos=True，在末尾追加结束符 id
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids):
        """把 token id 列表解码回文本字符串。"""
        return self.model.decode(ids)


class ChatFormat:
    """把用户/系统消息按 Llama 3 官方对话模板拼装成模型可接受的 token 序列。

    Llama 3 的对话模板结构大致为：
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n   # 等待模型续写回复
    """

    def __init__(self, tokenizer: Llama3Tokenizer, *,
                 default_system="You are a helpful assistant."):
        """
        Args:
            tokenizer (Llama3Tokenizer): 底层分词器实例。
            default_system (str): 当调用 encode 时未显式提供 system_message 时使用的默认系统提示词。
        """
        self.tok = tokenizer
        self.default_system = default_system

    def _header(self, role):
        """Encode <|start_header_id|>role<|end_header_id|>\n\n"""
        # 中文说明：构造某个对话角色（system/user/assistant）对应的头部 token 序列，
        # 格式固定为 <|start_header_id|> + role文本 + <|end_header_id|> + 两个换行符
        return (
            [self.tok.special["<|start_header_id|>"]]
            + self.tok.encode(role)
            + [self.tok.special["<|end_header_id|>"]]
            + self.tok.encode("\n\n")
        )

    def encode(self, user_message, system_message=None, allowed_special=None):
        """把一轮"系统提示 + 用户消息"编码为完整的模型输入 token 序列（含 assistant 头部占位）。

        Args:
            user_message (str): 用户输入的消息文本。
            system_message (str | None): 系统提示词；为 None 时使用 self.default_system。
            allowed_special: 透传给 tokenizer.encode 的 allowed_special 参数（用于系统消息中
                可能包含的特殊 token 文本字面量，允许其被识别为特殊 token 而非普通文本）。

        Returns:
            list[int]: 完整的、可直接喂给模型做续写生成的 token id 序列
                （已包含 assistant 角色的头部，等待模型生成后续内容）。
        """
        sys_msg = system_message if system_message is not None else self.default_system

        # 整个对话序列以 <|begin_of_text|> 开头
        ids = [self.tok.special["<|begin_of_text|>"]]

        # system
        # 拼接 system 角色头部 + 系统消息内容 + 单轮结束符
        ids += self._header("system")
        ids += self.tok.encode(sys_msg, allowed_special=allowed_special)
        ids += [self.tok.special["<|eot_id|>"]]

        # user
        # 拼接 user 角色头部 + 用户消息内容 + 单轮结束符
        ids += self._header("user")
        ids += self.tok.encode(user_message)
        ids += [self.tok.special["<|eot_id|>"]]

        # assistant header (no content yet)
        # 拼接 assistant 角色头部，但不填充内容——模型将从此处开始自回归续写回复
        ids += self._header("assistant")

        return ids

    def decode(self, ids):
        """把 token id 列表解码回文本字符串。"""
        return self.tok.decode(ids)


def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    """从模型生成的完整解码文本中截取出 assistant 回复正文部分（去掉前面的模板/历史内容）。

    Args:
        text (str): 已解码为字符串的完整生成结果（可能包含 system/user 等历史模板文本）。
        header_end (str): 用于定位"assistant 回复正文起始位置"的标记字符串，
            默认是 "assistant<|end_header_id|>\\n\\n"（即 assistant 角色头部结束的位置）。

    Returns:
        str: 截取并去除首尾空白后的 assistant 回复正文；若未找到该标记，则原样返回输入文本。
    """
    # Find the index of the first occurrence of "<|end_header_id|>"
    # 查找标记字符串在文本中第一次出现的位置
    index = text.find(header_end)

    if index != -1:
        # Return the substring starting after "<|end_header_id|>"
        # 找到了标记，截取标记之后的所有内容，并去除首尾空白字符作为最终回复正文
        return text[index + len(header_end):].strip()  # Strip removes leading/trailing whitespace
    else:
        # If the token is not found, return the original text
        # 未找到标记，说明格式不符合预期，直接原样返回整个文本
        return text


######################################################################
# Llama 3 fast (alternative code geared towards efficiency)
######################################################################
# 中文说明：下面这一组类是"高效版"实现，使用 PyTorch 内置的
# scaled_dot_product_attention（在支持的 GPU + 低精度 dtype 下可自动启用
# FlashAttention 等高效注意力内核）替代手写的注意力计算过程，从而获得更快的速度
# 和更低的显存占用。但代价是：这一组实现不支持 KV 缓存增量解码，
# 只适合"一次性给定完整序列做前向"的场景（如批量训练或非增量式推理）。

class GroupedQueryAttentionFast(nn.Module):
    """
    Drop-in replacement for GroupedQueryAttention but using PyTorch's
    scaled_dot_product_attention, which uses FlashAttention if run
    on an Ampere GPU (like A100) or newer and uses float16/bfloat16 or lower.
    """
    # 中文说明：GroupedQueryAttention 的"高效版"替代实现，接口更简单（不支持 mask/cache 参数），
    # 内部直接调用 torch.nn.functional.scaled_dot_product_attention 并开启 is_causal=True，
    # 让 PyTorch/底层硬件自动选择最优的注意力计算内核（如 FlashAttention）。
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        """初始化，参数含义与 GroupedQueryAttention 完全相同。"""
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, cos, sin):
        """前向传播（无 KV 缓存、无显式 mask 参数，因果关系交由 is_causal=True 自动处理）。

        Args:
            x (torch.Tensor): 形状 (batch_size, num_tokens, d_in)。
            cos, sin (torch.Tensor): RoPE 的 cos/sin 查找表。

        Returns:
            torch.Tensor: 形状 (batch_size, num_tokens, d_out) 的注意力输出。
        """
        b, num_tokens, _ = x.shape

        # Project to queries, keys, values
        # 分别做 Q/K/V 投影并 reshape 成多头形式：(b, heads_or_groups, num_tokens, head_dim)
        q = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Apply Rotary Positional Embedding
        # 对 Q/K 应用旋转位置编码（此处未传 offset，默认为 0，因为该"fast"版本不支持增量解码，
        # 每次都是从头对完整序列做全量前向）
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand key/value groups to full head count
        # 同样使用 repeat_interleave 把少量 KV 组扩展到与 Query 头数一致（GQA 核心操作）
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Efficient scaled dot-product attention
        # 调用 PyTorch 原生高效注意力实现：内部自动完成 QK^T 缩放、因果掩码、softmax、
        # 与 V 加权求和的全过程，并在满足硬件/精度条件时自动启用 FlashAttention 等融合内核
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            is_causal=True  # Enables Flash/FlexAttention kernels  # 开启因果模式，等价于施加下三角掩码，无需手动构造 mask 张量
        )

        # Combine heads and project
        # 合并多头结果并做输出投影
        attn_output = attn_output.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(attn_output)


class TransformerBlockFast(nn.Module):
    """
    Same as original TransformerBlock but uses
    GroupedQueryAttentionFast instead of GroupedQueryAttention.
    """
    # 中文说明：与 TransformerBlock 结构完全一致（Pre-Norm + GQA + SwiGLU 前馈 + 残差连接），
    # 唯一区别是注意力子层换成了不支持 KV 缓存、但计算更高效的 GroupedQueryAttentionFast。
    def __init__(self, cfg):
        """初始化，参数含义与 TransformerBlock 相同。"""
        super().__init__()
        self.att = GroupedQueryAttentionFast(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, cos, sin):
        """前向传播（无 mask、无 cache 参数，因果关系由内部 GroupedQueryAttentionFast 的 is_causal 处理）。

        Args:
            x (torch.Tensor): 形状 (batch_size, num_tokens, emb_dim)。
            cos, sin (torch.Tensor): RoPE 的 cos/sin 查找表。

        Returns:
            torch.Tensor: 形状 (batch_size, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x


class Llama3ModelFast(nn.Module):
    """
    Same as original Llama3Model but uses TransformerBlockFast
    instead of TransformerBlock, which in turn uses
    GroupedQueryAttentionFast instead of GroupedQueryAttention.
    """
    # 中文说明：Llama3Model 的高效版本，堆叠的是 TransformerBlockFast（内部使用
    # scaled_dot_product_attention）。不维护 KV 缓存/current_pos，
    # forward 也没有 cache 参数，每次调用都是对传入的完整序列做一次全量前向计算。
    def __init__(self, cfg):
        """初始化，结构与 Llama3Model 基本一致，仅替换为 Fast 版本的子模块，且不含 KV 缓存相关状态。"""
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, cos, sin`
            [TransformerBlockFast(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # 预计算 RoPE 的 cos/sin 查找表，逻辑与 Llama3Model 完全相同
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        """前向传播：对完整输入序列一次性计算所有位置的 logits（无 KV 缓存支持）。

        Args:
            in_idx (torch.LongTensor): 形状 (batch_size, num_tokens) 的 token id 序列。

        Returns:
            torch.Tensor: 形状 (batch_size, num_tokens, vocab_size) 的 logits。
        """
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        # 依次经过每一层 TransformerBlockFast（因果关系交由各层内部的 is_causal=True 自动处理）
        for block in self.trf_blocks:
            x = block(x, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits
