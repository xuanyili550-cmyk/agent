# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明（docstring）
========================
本文件是 Llama 3（含 Llama 3.2 1B / 3B）模型的**完整独立实现**，不依赖 HuggingFace
transformers 库，纯用 PyTorch 手写。核心技术点包括：

1. RoPE（Rotary Position Embedding，旋转位置编码）：
   - 用 `compute_rope_params` 预先计算好每个位置、每个频率对应的 cos/sin 值；
   - 用 `apply_rope` 把这些 cos/sin 值“旋转”作用到 query/key 向量上，
     从而在不引入额外可学习参数的情况下，把相对位置信息编码进注意力分数。
   - Llama 3.2 额外引入了 NTK-aware 的频率缩放（`rope_freq` 配置），
     使得模型能够外推到比训练时更长的上下文长度。

2. RMSNorm（Root Mean Square LayerNorm）：
   - 直接复用 `torch.nn.RMSNorm`，相比传统 LayerNorm 去掉了均值中心化，
     只用均方根做缩放，计算更省、精度损失更小，是 Llama 系列的标准做法。

3. GQA（Grouped-Query Attention，分组查询注意力）：
   - Query 仍然使用完整的头数（`num_heads`），但 Key/Value 只使用较少的
     “分组头数”（`num_kv_groups`），从而大幅减少 KV Cache 显存占用。
   - 通过 `repeat_interleave` 把每组 KV 头“复制”给该组内的多个 Query 头共享，
     核心是 `repeat_interleave` 与普通 `repeat` 在结果排列上的区别（见类内注释）。

4. SwiGLU（Swish-Gated Linear Unit）前馈网络：
   - `FeedForward` 类用两路线性层（`fc1` 门控路径、`fc2` 数值路径）
     分别计算，再用 SiLU（即 Swish）激活门控路径后与数值路径逐元素相乘，
     最后通过 `fc3` 投影回原维度，是 Llama 系列替代传统 ReLU-MLP 的做法。

5. 权重加载（`load_weights_into_llama`）：
   - 把从 HuggingFace safetensors / 官方权重文件中读出的、以
     `model.layers.{l}.xxx` 命名的权重张量，逐一搬运（`assign`）到本文件中
     手写模型对应的子模块参数上，包含形状校验与可选的权重绑定
     （word embedding 与输出层共享权重）处理。

此外文件末尾还提供了一套面向推理效率优化的“Fast”版本
（`GroupedQueryAttentionFast` / `TransformerBlockFast` / `Llama3ModelFast`），
使用 PyTorch 原生的 `scaled_dot_product_attention`（可在支持的 GPU 上自动
启用 FlashAttention 等高效内核），逻辑与慢速版本完全等价，仅实现方式不同。

本次修改**只新增了中文注释与 docstring**，未改动任何可执行代码
（变量名、函数签名、逻辑分支、缩进、字符串字面量、import 语句等均保持原样），
原有英文注释也全部保留。
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

import tiktoken
from tiktoken.load import load_tiktoken_bpe


# Llama 3.2 1B 模型的超参数配置字典
LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,           # Vocabulary size  # 词表大小（token 总数）
    "context_length": 131_072,       # Context length that was used to train the model  # 训练时使用的最大上下文长度（token 数）
    "emb_dim": 2048,                 # Embedding dimension  # 词嵌入/隐藏层维度
    "n_heads": 32,                   # Number of attention heads  # 注意力头（Query 头）数量
    "n_layers": 16,                  # Number of layers  # Transformer 层数（block 数）
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward  # 前馈网络（SwiGLU）中间层维度
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention  # GQA 中 Key/Value 的分组数（远小于 n_heads，用于节省显存）
    "rope_base": 500_000.0,          # The base in RoPE's "theta"  # RoPE 频率公式中的底数 theta_base
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage  # 使用 bfloat16 低精度以降低显存占用
    "rope_freq": {                   # RoPE frequency scaling  # RoPE 频率缩放参数（NTK-aware 缩放，用于长上下文外推）
        "factor": 32.0,               # 高频区域整体缩放因子
        "low_freq_factor": 1.0,       # 低频阈值因子
        "high_freq_factor": 4.0,      # 高频阈值因子
        "original_context_length": 8192,  # 原始（预训练时）的上下文长度，用于计算波长阈值
    }
}

# Llama 3.2 3B 模型的超参数配置字典（结构与 1B 相同，仅规模参数不同）
LLAMA32_CONFIG_3B = {
    "vocab_size": 128_256,           # Vocabulary size  # 词表大小
    "context_length": 131_072,       # Context length that was used to train the model  # 训练时最大上下文长度
    "emb_dim": 3072,                 # Embedding dimension  # 隐藏层维度（比 1B 更大）
    "n_heads": 24,                   # Number of attention heads  # 注意力头数量
    "n_layers": 28,                  # Number of layers  # Transformer 层数（比 1B 更多）
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward  # 前馈网络中间层维度
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention  # GQA 分组数
    "rope_base": 500_000.0,          # The base in RoPE's "theta"  # RoPE 底数
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage  # 低精度 dtype
    "rope_freq": {                   # RoPE frequency scaling  # RoPE 频率缩放参数
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}


class Llama3Model(nn.Module):
    """Llama 3 完整模型（标准/慢速实现版本）。

    结构：词嵌入 -> N 层 TransformerBlock（内部含 GQA 注意力 + SwiGLU 前馈）
          -> 最终 RMSNorm -> 输出线性层（映射到词表大小的 logits）。

    RoPE 所需的 cos/sin 表在构造时一次性预计算好，并注册为不参与训练、
    不参与 state_dict 持久化保存的 buffer（`persistent=False`），
    forward 时逐层传递给每个 TransformerBlock 使用。

    Args:
        cfg (dict): 模型配置字典，如 `LLAMA32_CONFIG_1B` / `LLAMA32_CONFIG_3B`，
            需包含 vocab_size、emb_dim、n_heads、n_layers、context_length、
            rope_base、rope_freq、dtype 等键。
    """
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        # 词嵌入层：把输入的 token id 映射为 emb_dim 维稠密向量
        # 权重形状：(vocab_size, emb_dim)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            # 用 ModuleList 而非 Sequential，是因为每个 block 的 forward 需要
            # 额外传入 mask、cos、sin 这几个非默认参数，Sequential 做不到
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最终归一化层（RMSNorm），在送入输出头之前对隐藏状态做归一化
        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        # 输出投影层：把隐藏状态 (emb_dim) 映射为词表 logits (vocab_size)，无偏置
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        # 预先计算 RoPE 所需的 cos / sin 表，形状均为 (context_length, head_dim)
        # head_dim = emb_dim // n_heads，即每个注意力头的维度
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        # 注册为 buffer：会随 .to(device) 移动，但不是可学习参数，
        # persistent=False 表示不会被保存进 state_dict（推理时可重新生成）
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        """前向传播。

        Args:
            in_idx (torch.LongTensor): 输入 token id 序列，形状 (batch_size, num_tokens)。

        Returns:
            torch.Tensor: 词表上的 logits，形状 (batch_size, num_tokens, vocab_size)。
        """
        # 查表得到词嵌入，形状 (batch_size, num_tokens, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        # 构造因果注意力掩码（causal mask）：上三角（不含对角线）为 True，
        # 表示这些位置在注意力计算时需要被屏蔽（不能看到未来的 token）
        # 形状：(num_tokens, num_tokens)
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        # 依次通过每一层 TransformerBlock，逐层更新隐藏状态 x
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        # 最终归一化
        x = self.final_norm(x)
        # 输出投影前先转换回配置的 dtype（防止归一化层内部升精度导致的 dtype 不一致）
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


class TransformerBlock(nn.Module):
    """单个 Transformer 解码器块（Pre-Norm 结构）。

    结构：x -> RMSNorm -> GQA 自注意力 -> 残差相加
            -> RMSNorm -> SwiGLU 前馈网络 -> 残差相加

    Args:
        cfg (dict): 模型配置字典，需包含 emb_dim、n_heads、n_kv_groups、dtype 等。
    """
    def __init__(self, cfg):
        super().__init__()
        # 分组查询注意力子模块
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        # SwiGLU 前馈网络子模块
        self.ff = FeedForward(cfg)
        # 注意力子层前的归一化（Pre-Norm）
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        # 前馈子层前的归一化（Pre-Norm）
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        """前向传播。

        Args:
            x (torch.Tensor): 输入隐藏状态，形状 (batch_size, num_tokens, emb_dim)。
            mask (torch.BoolTensor): 因果注意力掩码，形状 (num_tokens, num_tokens)。
            cos (torch.Tensor): RoPE 预计算的 cos 表，形状 (context_length, head_dim)。
            sin (torch.Tensor): RoPE 预计算的 sin 表，形状 (context_length, head_dim)。

        Returns:
            torch.Tensor: 输出隐藏状态，形状与输入相同 (batch_size, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        # 注意力子层的残差捷径
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back  # 残差相加，缓解深层网络梯度消失

        # Shortcut connection for feed-forward block
        # 前馈子层的残差捷径
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back  # 残差相加

        return x


class FeedForward(nn.Module):
    """SwiGLU 前馈网络（Llama 系列标准 MLP 结构）。

    与传统的 Linear->ReLU->Linear 不同，SwiGLU 用两路并行的线性变换：
    一路（fc1）经过 SiLU 激活作为“门控”，另一路（fc2）不激活作为“数值”，
    两者逐元素相乘后再经 fc3 投影回原维度。这种门控机制通常比 ReLU-MLP
    效果更好（是 GLU 变体家族的一种）。

    Args:
        cfg (dict): 模型配置字典，需包含 emb_dim、hidden_dim、dtype。
    """
    def __init__(self, cfg):
        super().__init__()
        # 门控路径的线性层：emb_dim -> hidden_dim，无偏置
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # 数值路径的线性层：emb_dim -> hidden_dim，无偏置
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # 输出投影层：hidden_dim -> emb_dim，无偏置
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状 (batch_size, num_tokens, emb_dim)。

        Returns:
            torch.Tensor: 输出张量，形状 (batch_size, num_tokens, emb_dim)。
        """
        x_fc1 = self.fc1(x)  # 门控路径，形状 (batch_size, num_tokens, hidden_dim)
        x_fc2 = self.fc2(x)  # 数值路径，形状 (batch_size, num_tokens, hidden_dim)
        # SiLU(x_fc1) 即 Swish 激活；与 x_fc2 逐元素相乘构成 SwiGLU 的门控效果
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)  # 投影回 emb_dim 维度


class GroupedQueryAttention(nn.Module):
    """分组查询注意力（Grouped-Query Attention, GQA）。

    与标准多头自注意力（MHA）不同，GQA 的 Query 仍使用完整的 `num_heads` 个头，
    但 Key / Value 只使用较少的 `num_kv_groups` 组（`num_kv_groups` < `num_heads`），
    每一组 KV 会被同一组内的若干个 Query 头共享（通过 `repeat_interleave` 复制）。
    这样可以显著减少推理时 KV Cache 的显存占用，同时相比 MQA（只有 1 组 KV）
    保留了更多表达能力，是 MHA 和 MQA 之间的折中方案。

    Args:
        d_in (int): 输入特征维度。
        d_out (int): 输出特征维度（也是 Query 的总维度，d_out = num_heads * head_dim）。
        num_heads (int): 注意力头（Query 头）数量。
        num_kv_groups (int): Key/Value 分组数，需能整除 num_heads。
        dtype: 权重的数据类型（如 torch.bfloat16）。
    """
    def __init__(
            self, d_in, d_out, num_heads, num_kv_groups, dtype=None
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        # 每个注意力头的维度
        self.head_dim = d_out // num_heads

        # Key/Value 投影层：只投影到 num_kv_groups * head_dim 维度（远小于 d_out），
        # 这正是 GQA 节省参数量与显存的关键所在
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        # 每个 KV 组需要被多少个 Query 头共享
        self.group_size = num_heads // num_kv_groups

        # Query 投影层：投影到完整的 d_out 维度（num_heads * head_dim）
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        # 注意力输出后的线性投影层
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状 (b, num_tokens, d_in)。
            mask (torch.BoolTensor): 因果掩码，形状至少覆盖 (num_tokens, num_tokens)。
            cos (torch.Tensor): RoPE cos 表。
            sin (torch.Tensor): RoPE sin 表。

        Returns:
            torch.Tensor: 注意力输出，形状 (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)  # Query 投影
        keys = self.W_key(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)  # Key 投影（维度更小）
        values = self.W_value(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)  # Value 投影（维度更小）

        # Reshape queries, keys, and values
        # 将最后一维拆分为“头数 x 每头维度”，为多头/分组计算做准备
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries
        # 把“头”维度换到第 1 维，方便按头做批量矩阵乘法
        keys = keys.transpose(1, 2)  # Shape: (b, num_kv_groups, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_kv_groups, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)

        # Apply RoPE
        # 对 Key 和 Query 分别施加旋转位置编码（Value 不需要施加 RoPE）
        keys = apply_rope(keys, cos, sin)
        queries = apply_rope(queries, cos, sin)

        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)
        # 关键的 GQA 操作：用 repeat_interleave 把每个 KV 组“原地重复” group_size 次，
        # 使得 KV 的头数从 num_kv_groups 扩展到 num_heads，与 Query 头数对齐，
        # 从而可以直接做逐头的注意力矩阵乘法
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]
        # 中文补充说明：repeat_interleave 是“每个元素连续重复”，
        # 而普通 repeat 是“整个序列重复多遍”——两者顺序不同，
        # 必须用 repeat_interleave 才能保证第 i 个 Query 头对应到正确的 KV 组。

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (b, num_heads, num_tokens, num_tokens)
        # 计算注意力分数：Q @ K^T，对每个头独立做点积
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Use the mask to fill attention scores
        # 用因果掩码把“未来位置”的注意力分数填成 -inf，softmax 后趋近于 0
        attn_scores = attn_scores.masked_fill(mask[:num_tokens, :num_tokens], -torch.inf)

        # 缩放点积注意力：除以 sqrt(head_dim) 防止数值过大导致 softmax 梯度消失
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 注意力权重与 Value 加权求和，再把头维度换回第 2 维
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 把多头拼接（reshape）回单一的 d_out 维度
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection  # 最终线性投影

        return context_vec


# ==============================================================================
# RoPE implementation summary
#
#
# There are two common styles to implement RoPE, which are
# mathematically equivalent;
# they mainly differ in how the rotation matrix pairs dimensions.
#
# 1) Split-halves style (this repo, Hugging Face Transformers):
#
#   For hidden dim d = 8 (example):
#
#       [ x0   x1   x2   x3   x4   x5   x6   x7 ]
#         │    │    │    │    │    │    │    │
#         ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
#        cos  cos  cos  cos  sin  sin  sin  sin
#
#   Rotation matrix:
#
#       [ cosθ   -sinθ    0      0   ... ]
#       [ sinθ    cosθ    0      0   ... ]
#       [  0       0    cosθ   -sinθ ... ]
#       [  0       0    sinθ    cosθ ... ]
#        ...
#
#   Here, the embedding dims are split into two halves and then
#   each one is rotated in blocks.
#
#
# 2) Interleaved (even/odd) style (original paper, Llama repo):
#
#   For hidden dim d = 8 (example):
#
#       [ x0   x1   x2   x3   x4   x5   x6   x7 ]
#         │    │    │    │    │    │    │    │
#         ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
#        cos  sin  cos  sin  cos  sin  cos  sin
#
#   Rotation matrix:
#       [ cosθ  -sinθ    0      0   ... ]
#       [ sinθ   cosθ    0      0   ... ]
#       [  0      0    cosθ   -sinθ ... ]
#       [  0      0    sinθ    cosθ ... ]
#        ...
#
#   Here, embedding dims are interleaved as even/odd cosine/sine pairs.
#
# Both layouts encode the same relative positions; the only difference is how
# dimensions are paired.
# ==============================================================================
# 中文小结：本文件采用“前后对半分”（split-halves）风格实现 RoPE，
# 即把 head_dim 维向量切成前半 x1 和后半 x2 两部分，分别与 cos/sin 表相乘、
# 组合，等价于把每一对 (x_i, x_{i+d/2}) 看作一个二维向量做旋转。
# 这与原始论文/Llama 官方仓库使用的“奇偶交错”（interleaved）风格在数学上等价，
# 只是维度配对方式不同，两者不能直接混用（否则权重加载/推理结果会出错）。


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None, dtype=torch.float32):
    """预先计算 RoPE 所需的 cos / sin 表。

    RoPE 的核心思想：对于每一对维度 (2i, 2i+1)（或按 split-halves 风格是
    (i, i + head_dim/2)），定义一个旋转角速度 `inv_freq[i]`，位置 `pos` 处
    该维度对的旋转角度为 `pos * inv_freq[i]`。把 query/key 向量按该角度旋转，
    即可让注意力分数 q·k 只依赖于两个 token 的**相对位置**，而不是绝对位置。

    Llama 3.2 额外做了 NTK-aware 频率缩放（当 `freq_config` 不为 None 时）：
    根据波长把频率分为低频、高频、中频三段，分别做不同程度的缩放，
    使得模型能够在推理时外推到比训练时（`original_context_length`）
    更长的上下文，同时尽量不损失短距离位置的分辨能力。

    Args:
        head_dim (int): 每个注意力头的维度，必须为偶数。
        theta_base (float): RoPE 频率公式中的底数 theta（默认 10000，
            Llama 3 中通常用更大的 500000 以适配长上下文）。
        context_length (int): 需要预计算的最大位置数（通常等于模型的
            context_length，也可以按需只算训练/推理实际用到的长度）。
        freq_config (dict | None): 若不为 None，则启用 NTK-aware 频率缩放，
            需包含 factor、low_freq_factor、high_freq_factor、
            original_context_length 四个键。
        dtype: 计算时使用的浮点精度（默认 float32，保证三角函数精度）。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (cos, sin)，形状均为
            (context_length, head_dim)。
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    # 计算逆频率 inv_freq[i] = 1 / theta_base^(2i/head_dim)，i = 0, 1, ..., head_dim/2 - 1
    # 形状：(head_dim // 2,)；i 越大频率越低（对应更“慢”的旋转，捕捉更长距离的位置关系）
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments
    # Llama 3.2 的 NTK-aware 频率缩放，用于支持长上下文外推
    if freq_config is not None:
        # 低频阈值对应的波长（波长越长，频率越低）
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        # 高频阈值对应的波长
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        # 每个频率分量对应的波长 = 2π / 频率
        wavelen = 2 * torch.pi / inv_freq

        # 波长大于低频阈值（即频率过低）的分量，整体除以 factor 做缩放；
        # 其余分量暂时保持原值
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        # 计算平滑插值系数，用于中频区域在“缩放”和“不缩放”之间做线性过渡
        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        # 中频区域的平滑插值频率
        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        # 判断哪些分量属于“中频”区间（介于高频阈值与低频阈值之间）
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        # 中频区间用平滑插值后的频率替换，其余保持之前 where 的结果
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    # 位置索引 0, 1, ..., context_length - 1
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    # 外积：每个位置 x 每个频率分量 = 该位置在该频率下的旋转角度
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    # 按 split-halves 风格，把角度矩阵复制一份拼在后面，
    # 使得前半 head_dim//2 维和后半 head_dim//2 维使用相同的角度
    # （对应 apply_rope 中 x1, x2 分别与 cos/sin 相乘的用法）
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    # 预先算好 cos/sin，避免每次前向传播重复计算三角函数
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    """把预计算好的 RoPE cos/sin 旋转变换应用到输入张量（Query 或 Key）上。

    采用 split-halves 风格：把 `head_dim` 对半切成前半 `x1` 和后半 `x2`，
    构造“旋转后”的向量 `rotated = concat(-x2, x1)`，再与原始 `x` 按
    `x * cos + rotated * sin` 组合，等价于对每一对维度 (x1_i, x2_i)
    做二维平面旋转，旋转角度由 `cos`/`sin` 表中对应位置的值决定。

    Args:
        x (torch.Tensor): 待旋转的张量（Query 或 Key），
            形状 (batch_size, num_heads, seq_len, head_dim)。
        cos (torch.Tensor): RoPE 预计算的 cos 表，形状至少为 (seq_len, head_dim)。
        sin (torch.Tensor): RoPE 预计算的 sin 表，形状至少为 (seq_len, head_dim)。

    Returns:
        torch.Tensor: 旋转后的张量，形状与输入 x 相同，dtype 与输入 x 相同。
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    # 把最后一维（head_dim）对半切开
    x1 = x[..., : head_dim // 2]  # First half  # 前半部分
    x2 = x[..., head_dim // 2:]  # Second half  # 后半部分

    # Adjust sin and cos shapes
    # 只取实际序列长度对应的 cos/sin，并广播出 batch 和 head 维度
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    # 构造“旋转 90 度”后的向量：(-x2, x1)，用于配合 cos/sin 实现二维旋转公式
    # 旋转公式：[x1', x2'] = [x1*cosθ - x2*sinθ,  x2*cosθ + x1*sinθ]
    # 这里通过 x*cos + concat(-x2, x1)*sin 一次性对前后两半同时完成上述旋转
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    # 旋转计算本身用较高精度更稳妥，计算完成后转换回原 dtype（如 bfloat16）以节省显存
    return x_rotated.to(dtype=x.dtype)


##########################################
# Tokenizer
##########################################
# 以下为分词器与对话格式相关的辅助类，与 RoPE/GQA 等模型结构无直接关系，
# 主要负责文本 <-> token id 的转换，以及 Llama 3 官方对话模板的拼接。


class Llama3Tokenizer:
    """Thin wrapper around tiktoken that keeps track of Llama-3 special IDs."""
    # 中文说明：对 tiktoken BPE 分词器的一层薄封装，
    # 负责加载 Meta 官方的 tokenizer 模型文件（.model），
    # 并维护 Llama 3 特殊 token（如 <|begin_of_text|> 等）到 id 的映射。
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        # 从 tiktoken 格式的 BPE 词表文件中加载可合并的 token（mergeable ranks）
        mergeable = load_tiktoken_bpe(model_path)

        # hard-coded from Meta's tokenizer.json
        # 硬编码的特殊 token id，与 Meta 官方 tokenizer.json 中的定义保持一致
        self.special = {
            "<|begin_of_text|>": 128000,   # 文本开始标记
            "<|end_of_text|>": 128001,     # 文本结束标记
            "<|start_header_id|>": 128006, # 对话角色 header 开始标记
            "<|end_header_id|>": 128007,   # 对话角色 header 结束标记
            "<|eot_id|>": 128009,          # 单轮对话结束标记（end of turn）
        }
        # 补充剩余的“保留”特殊 token（占位，避免与上面已用到的 id 冲突）
        self.special.update({f"<|reserved_{i}|>": 128002 + i
                             for i in range(256)
                             if 128002 + i not in self.special.values()})

        # 构造 tiktoken 编码器：正则切分规则 pat_str 定义了如何把原始文本
        # 切分成候选 token 片段（英文缩写、单词、数字、标点、空白等），
        # 再结合 mergeable_ranks（BPE 合并规则）和 special_tokens 完成编码
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                    r"|\p{N}{1,3}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+"
                    r"|\s+(?!\S)"
                    r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text, bos=False, eos=False, **kwargs):
        """把文本编码为 token id 列表。

        Args:
            text (str): 待编码的原始文本。
            bos (bool): 是否在开头加上 <|begin_of_text|> 标记。
            eos (bool): 是否在结尾加上 <|end_of_text|> 标记。
            **kwargs: 透传给底层 `tiktoken.Encoding.encode`（如 allowed_special）。

        Returns:
            list[int]: token id 列表。
        """
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.model.encode(text)
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids):
        """把 token id 列表解码回文本字符串。"""
        return self.model.decode(ids)


class ChatFormat:
    # 中文说明：负责按 Llama 3 官方对话模板，把 system/user 消息拼接成
    # 模型期望的 token 序列，格式形如：
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
    # <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
    # <|start_header_id|>assistant<|end_header_id|>\n\n（等待模型续写回复）

    def __init__(self, tokenizer: Llama3Tokenizer, *,
                 default_system="You are a helpful assistant."):
        """
        Args:
            tokenizer (Llama3Tokenizer): 底层分词器实例。
            default_system (str): 当调用 encode 时未显式传入 system_message，
                使用的默认系统提示词。
        """
        self.tok = tokenizer
        self.default_system = default_system

    def _header(self, role):
        """Encode <|start_header_id|>role<|end_header_id|>\n\n"""
        # 中文说明：编码某个对话角色（system/user/assistant）的 header 部分，
        # 格式为 <|start_header_id|>{role}<|end_header_id|>\n\n
        return (
            [self.tok.special["<|start_header_id|>"]]
            + self.tok.encode(role)
            + [self.tok.special["<|end_header_id|>"]]
            + self.tok.encode("\n\n")
        )

    def encode(self, user_message, system_message=None, allowed_special=None):
        """把一轮 system + user 消息编码为完整的 prompt token 序列。

        Args:
            user_message (str): 用户消息文本。
            system_message (str | None): 系统提示词，None 时使用 default_system。
            allowed_special: 透传给 system 消息编码时的 allowed_special 参数
                （用于允许文本中出现特殊 token 字面量）。

        Returns:
            list[int]: 拼接好的 token id 列表，末尾停在 assistant header 处，
                等待模型继续生成回复内容。
        """
        sys_msg = system_message if system_message is not None else self.default_system

        # 整个 prompt 以 <|begin_of_text|> 开头
        ids = [self.tok.special["<|begin_of_text|>"]]

        # system
        # 拼接 system 角色的 header + 内容 + 单轮结束标记
        ids += self._header("system")
        ids += self.tok.encode(sys_msg, allowed_special=allowed_special)
        ids += [self.tok.special["<|eot_id|>"]]

        # user
        # 拼接 user 角色的 header + 内容 + 单轮结束标记
        ids += self._header("user")
        ids += self.tok.encode(user_message)
        ids += [self.tok.special["<|eot_id|>"]]

        # assistant header (no content yet)
        # 拼接 assistant 角色的 header，但不填内容，留给模型自回归生成
        ids += self._header("assistant")

        return ids

    def decode(self, ids):
        """把 token id 列表解码回文本字符串（透传给底层分词器）。"""
        return self.tok.decode(ids)


def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    """从模型生成的完整文本中，截取出 assistant 回复的正文部分。

    模型生成的完整解码文本通常包含前面的 system/user 部分以及
    assistant 的 header，本函数通过查找 `header_end` 标记字符串，
    只保留其之后的内容（即真正的回复正文），并去除首尾空白。

    Args:
        text (str): 模型解码后的完整文本。
        header_end (str): assistant header 结束的标志字符串，
            默认是 "assistant<|end_header_id|>\\n\\n"。

    Returns:
        str: 截取并 strip 后的回复正文；若未找到标志字符串，原样返回 text。
    """
    # Find the index of the first occurrence of "<|end_header_id|>"
    # 查找 assistant header 结束标记第一次出现的位置
    index = text.find(header_end)

    if index != -1:
        # Return the substring starting after "<|end_header_id|>"
        # 返回该标记之后的子串，并去除首尾空白（Strip removes leading/trailing whitespace）
        return text[index + len(header_end):].strip()  # Strip removes leading/trailing whitespace
    else:
        # If the token is not found, return the original text
        # 未找到标记时，原样返回输入文本
        return text


######################################################################
# Llama 3 fast (alternative code geared towards efficiency)
######################################################################
# 中文说明：以下是面向推理效率优化的“快速版”实现，
# 逻辑与前面的标准版（GroupedQueryAttention / TransformerBlock / Llama3Model）
# 完全等价，唯一区别是用 PyTorch 内置的 `scaled_dot_product_attention`
# 替代手写的 QK^T -> mask -> softmax -> @V 流程，从而在支持的硬件上
# 自动启用 FlashAttention / FlexAttention 等高效融合内核，速度更快、显存更省。

class GroupedQueryAttentionFast(nn.Module):
    """
    Drop-in replacement for GroupedQueryAttention but using PyTorch's
    scaled_dot_product_attention, which uses FlashAttention if run
    on an Ampere GPU (like A100) or newer and uses float16/bfloat16 or lower.
    """
    # 中文说明：GroupedQueryAttention 的高效替代实现。
    # 参数构造与标准版完全一致，区别仅在 forward 中改用
    # torch.nn.functional.scaled_dot_product_attention 并设置 is_causal=True，
    # 由 PyTorch 内部自动选择最优的注意力计算内核（如 FlashAttention）。
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        """
        Args:
            d_in (int): 输入特征维度。
            d_out (int): 输出/Query 总维度。
            num_heads (int): Query 头数。
            num_kv_groups (int): Key/Value 分组数。
            dtype: 权重数据类型。
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        # 与标准版 GroupedQueryAttention 相同的投影层定义
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, cos, sin):
        """前向传播（注意：不再需要显式传入 mask，因果性由 is_causal=True 保证）。

        Args:
            x (torch.Tensor): 输入张量，形状 (b, num_tokens, d_in)。
            cos (torch.Tensor): RoPE cos 表。
            sin (torch.Tensor): RoPE sin 表。

        Returns:
            torch.Tensor: 注意力输出，形状 (b, num_tokens, d_out)。
        """
        b, num_tokens, _ = x.shape

        # Project to queries, keys, values
        # 投影并 reshape + transpose 为 (b, num_heads/num_kv_groups, num_tokens, head_dim)
        q = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Apply Rotary Positional Embedding
        # 对 Query、Key 施加 RoPE 旋转位置编码
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand key/value groups to full head count
        # 与标准版相同：用 repeat_interleave 把 KV 头数扩展到与 Query 头数一致
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Efficient scaled dot-product attention
        # 调用 PyTorch 原生融合算子完成 缩放点积 + 因果掩码 + softmax + 加权求和，
        # 在支持的 GPU 上会自动派发到 FlashAttention 等高效内核
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            is_causal=True  # Enables Flash/FlexAttention kernels  # 内部自动应用因果掩码
        )

        # Combine heads and project
        # 合并多头并做输出投影
        attn_output = attn_output.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(attn_output)


class TransformerBlockFast(nn.Module):
    """
    Same as original TransformerBlock but uses
    GroupedQueryAttentionFast instead of GroupedQueryAttention.
    """
    # 中文说明：结构与标准版 TransformerBlock 完全相同（Pre-Norm + 残差），
    # 唯一区别是内部注意力子模块换成了效率更高的 GroupedQueryAttentionFast，
    # 因此 forward 也不再需要传入 mask 参数。
    def __init__(self, cfg):
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
        """前向传播，逻辑与 TransformerBlock.forward 相同，只是不需要 mask 参数。

        Args:
            x (torch.Tensor): 输入隐藏状态，形状 (batch_size, num_tokens, emb_dim)。
            cos (torch.Tensor): RoPE cos 表。
            sin (torch.Tensor): RoPE sin 表。

        Returns:
            torch.Tensor: 输出隐藏状态，形状同输入。
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
    # 中文说明：整体结构与 Llama3Model 完全一致，
    # 区别仅在于内部使用 TransformerBlockFast（进而使用
    # GroupedQueryAttentionFast + scaled_dot_product_attention），
    # 因此 forward 中也不需要再手动构造因果 mask。
    def __init__(self, cfg):
        """
        Args:
            cfg (dict): 模型配置字典，同 Llama3Model。
        """
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, cos, sin`
            # 同样使用 ModuleList，因为每个 block 需要额外的 cos、sin 参数
            [TransformerBlockFast(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # 预先计算 RoPE cos/sin 表，逻辑与 Llama3Model 中完全相同
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
        """前向传播。

        Args:
            in_idx (torch.LongTensor): 输入 token id 序列，形状 (batch_size, num_tokens)。

        Returns:
            torch.Tensor: 词表 logits，形状 (batch_size, num_tokens, vocab_size)。
        """
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        # 注意：这里不再需要像 Llama3Model 那样显式构造 causal mask，
        # 因果性交由 GroupedQueryAttentionFast 内部的 is_causal=True 处理
        for block in self.trf_blocks:
            x = block(x, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


def assign(left, right, tensor_name="unknown"):
    """把 `right`（通常是从磁盘加载的预训练权重）安全地拷贝赋值给 `left`
    （模型内部对应的 `nn.Parameter`）。

    会先校验两者形状是否一致（不一致直接抛错，避免静默的权重错位），
    再在 `torch.no_grad()` 上下文中原地拷贝数值，最后返回拷贝后的 `left`
    （通常直接重新赋值给 `model.xxx.weight`，与原写法保持一致）。

    Args:
        left (torch.nn.Parameter | torch.Tensor): 模型中待赋值的参数张量。
        right (torch.Tensor | array-like): 预训练权重（可以是 Tensor，
            也可以是能被 `torch.as_tensor` 转换的其他数组类型）。
        tensor_name (str): 用于报错信息中标识具体是哪个权重张量出了问题。

    Returns:
        torch.nn.Parameter | torch.Tensor: 拷贝完成后的 `left`。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            # right 已经是 Tensor，直接原地拷贝数值（保留 left 原有的 dtype/device）
            left.copy_(right)
        else:
            # right 不是 Tensor（例如 numpy 数组），先转换为与 left 相同 dtype/device 的 Tensor 再拷贝
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    return left


def load_weights_into_llama(model, param_config, params):
    """把 HuggingFace 风格命名的预训练权重字典，逐层加载进手写的 Llama3Model
    （或 Llama3ModelFast）实例中。

    `params` 中的 key 遵循 HuggingFace 常见命名规则，例如：
    `model.layers.{l}.self_attn.q_proj.weight`、
    `model.layers.{l}.mlp.gate_proj.weight` 等；
    本函数负责把它们一一映射（`assign`）到本文件中对应模块的参数上，
    例如 `model.trf_blocks[l].att.W_query.weight`。

    Args:
        model (Llama3Model | Llama3ModelFast): 待加载权重的模型实例
            （结构需与 param_config 描述的层数等超参数匹配）。
        param_config (dict): 模型配置字典，至少需要包含 "n_layers"。
        params (dict[str, torch.Tensor]): 从 checkpoint（如 safetensors）
            中读出的、以 HuggingFace 命名规则组织的权重字典。

    Returns:
        None：函数直接原地修改 `model` 的参数，无返回值。
    """

    # 词嵌入层权重：HuggingFace 命名为 model.embed_tokens.weight
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    # 逐层加载每一层 TransformerBlock 的权重
    for l in range(param_config["n_layers"]):

        # Load attention weights
        # 加载 Query 投影权重
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        # 加载 Key 投影权重
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        # 加载 Value 投影权重
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        # 加载注意力输出投影权重
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        # 加载注意力子层前的 RMSNorm（input_layernorm）权重
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Load FeedForward weights
        # SwiGLU 中的门控路径权重（HuggingFace 命名为 gate_proj），对应本文件的 fc1
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        # SwiGLU 中的数值路径权重（HuggingFace 命名为 up_proj），对应本文件的 fc2
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        # SwiGLU 输出投影权重（HuggingFace 命名为 down_proj），对应本文件的 fc3
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        # 加载前馈子层前的 RMSNorm（post_attention_layernorm）权重
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Load output layer weights
    # 加载最终归一化层权重
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params.keys():
        # 存在独立的输出头权重时直接加载
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        # 否则说明该 checkpoint 使用了权重绑定（weight tying）：
        # 输出层与词嵌入层共享同一份权重矩阵
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")
