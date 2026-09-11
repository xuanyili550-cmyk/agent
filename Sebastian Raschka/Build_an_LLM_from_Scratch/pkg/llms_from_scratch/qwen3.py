# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明（新增，原文件无对应文档，保留上方英文版权声明不变）：

本文件是 Qwen3 系列大语言模型的“从零实现”，是一个完整且独立（self-contained）的
PyTorch 实现，不依赖 HuggingFace Transformers 的建模代码。核心内容包括：

1. 多套官方配置字典（0.6B / 1.7B / 4B / 8B / 14B / 32B 稠密模型，以及 30B-A3B
   混合专家 MoE 模型），用于快速实例化不同规模的 Qwen3。
2. Qwen3Model：整体模型骨架，负责词嵌入、堆叠 Transformer 层、最终归一化与输出头。
3. TransformerBlock：单个 Transformer 层，包含“注意力子层 + 前馈子层”，均使用
   Pre-Norm（先归一化再计算，再残差相加）结构。
4. GroupedQueryAttention（GQA，分组查询注意力）：Q 头数多于 K/V 头数，K/V 头在
   计算前通过 repeat_interleave 扩展以匹配 Q 头数，从而节省 KV Cache 显存；
   并支持 QK-Norm（对每个头的 Q、K 向量做 RMSNorm，提升训练稳定性）。
5. RoPE（旋转位置编码）：compute_rope_params 预计算不同位置、不同频率的
   cos/sin 表；apply_rope 将其应用到 Q、K 张量上，为注意力引入相对位置信息。
6. RMSNorm：均方根归一化，Qwen3 中不减均值、只除以均方根，是 LayerNorm 的
   轻量替代。
7. FeedForward / MoEFeedForward：稠密模型使用 SwiGLU 前馈网络；MoE 模型则使用
   “门控路由（gate）+ 多专家（experts）+ Top-K 选择 + 加权求和”的混合专家结构。
8. load_weights_into_qwen：将 HuggingFace 格式（safetensors）权重字典按名称
   逐一拷贝进本文件定义的模型结构中，处理 QK-Norm、MoE 专家、权重绑定
   （weight tying）等细节。
9. Qwen3Tokenizer：基于 tokenizers 库的分词器封装，支持 Qwen3 的聊天模板包装
   （<|im_start|>/<|im_end|> 等特殊符号）与思考模式（<think> 标签）。
10. download_from_huggingface / download_from_huggingface_from_snapshots：
    从 HuggingFace Hub 下载分词器文件或模型权重分片（safetensors）的工具函数。

注意：本次修改仅新增中文注释与文档字符串，未改动任何可执行代码、变量名、
函数签名、逻辑顺序、缩进或字符串内容，原有英文注释全部保留。
"""

import os
import json
import re
from pathlib import Path

import requests
import torch
import torch.nn as nn


# 0.6 billion parameters
# Qwen3 0.6B 稠密模型配置：词表大小、上下文长度、嵌入维度、注意力头数/层数、
# 前馈层中间维度、每个注意力头的维度、是否开启 QK-Norm、GQA 分组数、
# RoPE 的 theta 基数、模型权重的存储精度（bfloat16 以节省显存）
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,           # Vocabulary size
    "context_length": 40_960,        # Context length that was used to train the model
    "emb_dim": 1024,                 # Embedding dimension
    "n_heads": 16,                   # Number of attention heads
    "n_layers": 28,                  # Number of layers
    "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
    "head_dim": 128,                 # Size of the heads in GQA
    "qk_norm": True,                 # Whether to normalize queries and keys in GQA
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
}

# 1.7 billion parameters
# Qwen3 1.7B 配置：相比 0.6B，嵌入维度与前馈中间维度均翻倍，层数与头数不变
QWEN3_CONFIG_1_7B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2048,                 # 2x larger than above
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 6144,              # 2x larger than above
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 4 billion parameters
# Qwen3 4B 配置：进一步放大嵌入维度、头数、层数与前馈中间维度
QWEN3_CONFIG_4B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2560,                 # 25% larger than above
    "n_heads": 32,                   # 2x larger than above
    "n_layers": 36,                  # 29% larger than above
    "hidden_dim": 9728,              # ~3x larger than above
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 8 billion parameters
# Qwen3 8B 配置：嵌入维度与前馈中间维度继续增大，头数/层数与 4B 相同
QWEN3_CONFIG_8B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 4096,                 # 60% larger than above
    "n_heads": 32,
    "n_layers": 36,
    "hidden_dim": 12288,             # 26% larger than above
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 14 billion parameters
# Qwen3 14B 配置：各维度进一步放大
QWEN3_CONFIG_14B = {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,                 # 25% larger than above
        "n_heads": 40,                   # 25% larger than above
        "n_layers": 40,                  # 11% larger than above
        "hidden_dim": 17408,             # 42% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
}

# Qwen3 32B 配置：嵌入维度与 14B 相同，但头数、层数、前馈中间维度显著更大
QWEN3_CONFIG_32B = {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,
        "n_heads": 64,                   # 60% larger than above
        "n_layers": 64,                  # 60% larger than above
        "hidden_dim": 25600,             # 47% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
}

# Mixture of Experts Model
# Qwen3 30B-A3B（MoE 混合专家）配置：注意没有单一的 "hidden_dim"（稠密前馈
# 中间维度），而是使用 "num_experts"（专家总数）、"num_experts_per_tok"
# （每个 token 实际激活/路由到的专家数，即 Top-K 的 K）、
# "moe_intermediate_size"（单个专家内部前馈网络的中间维度）。
# 上下文长度和 rope_base 也远大于稠密模型，因为该模型面向长上下文场景。
QWEN3_CONFIG_30B_A3B = {
    "vocab_size": 151_936,
    "context_length": 262_144,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 48,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_base": 10_000_000.0,
    "dtype": torch.bfloat16,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
}


class Qwen3Model(nn.Module):
    """
    Qwen3 整体模型（不含语言模型头之外的其他任务头），对应 HuggingFace 中的
    `Qwen3Model`/`Qwen3ForCausalLM` 主干部分。

    结构：Token 嵌入 -> N 层 TransformerBlock（内部含 GQA 注意力 + 前馈/MoE）
    -> 最终 RMSNorm -> 线性输出头（映射到词表维度得到 logits）。

    RoPE 所需的 cos/sin 表在此处预计算一次，通过 register_buffer 注册为
    非持久（persistent=False）缓冲区（不会被保存到 state_dict 中，因为
    它们可以随时根据 head_dim/rope_base/context_length 重新计算）。

    参数:
        cfg (dict): 模型配置字典，例如 QWEN_CONFIG_06_B，需包含 vocab_size、
            emb_dim、n_layers、head_dim（可为 None）、n_heads、rope_base、
            context_length、dtype 等键。
    """
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        # 词嵌入层：将输入的 token id（形状 [batch, seq_len]）映射为
        # 稠密向量，输出形状 [batch, seq_len, emb_dim]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            # 堆叠 n_layers 个 Transformer 块；因为每层 forward 需要额外传入
            # mask、cos、sin 这几个参数，nn.Sequential 无法满足多输入需求，
            # 所以这里用 nn.ModuleList 手动在 forward 里逐层调用
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # 最终的 RMSNorm，在进入输出头之前对隐藏状态做归一化
        self.final_norm = RMSNorm(cfg["emb_dim"])
        # 输出头：将隐藏状态 [batch, seq_len, emb_dim] 线性映射为
        # [batch, seq_len, vocab_size] 的 logits，不使用偏置项
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        # 若配置未显式指定 head_dim，则按 emb_dim // n_heads 均分计算；
        # 否则（如 Qwen3 常见配置）使用配置中显式给出的 head_dim（可能与
        # emb_dim / n_heads 不相等，这是 GQA + 独立 head_dim 设计带来的灵活性）
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        # 预计算 RoPE 所需的 cos/sin 表，形状均为 (context_length, head_dim)，
        # 之后所有层共享同一份 cos/sin（因为它们只依赖位置和 head_dim，与
        # 具体层无关）
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        # 注册为非持久缓冲区：会随模型 .to(device) 移动，但不会出现在
        # state_dict 里，也不会被保存/加载（因为可重新计算，无需持久化）
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        """
        前向传播。

        参数:
            in_idx (torch.LongTensor): 输入 token id，形状 [batch_size, num_tokens]。

        返回:
            torch.Tensor: 预测的 logits，形状 [batch_size, num_tokens, vocab_size]。
        """
        # Forward pass
        # 词嵌入查表：[batch, num_tokens] -> [batch, num_tokens, emb_dim]
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        # 构造因果掩码（causal mask）：上三角（不含对角线）为 True，表示
        # 这些位置在注意力计算中需要被屏蔽（当前 token 不能看到未来 token）。
        # 形状：[num_tokens, num_tokens]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        # 依次通过每一个 Transformer 层，逐层更新隐藏状态 x，
        # 形状始终保持 [batch, num_tokens, emb_dim]
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        # 最终归一化
        x = self.final_norm(x)
        # 输出头映射到词表空间，得到未归一化的 logits；
        # 显式转换回 cfg["dtype"]（例如 bfloat16）以保证精度/显存一致
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


class TransformerBlock(nn.Module):
    """
    单个 Transformer 层（Pre-Norm 结构）。

    结构：
        x -> RMSNorm(norm1) -> GQA 注意力 -> 残差相加
          -> RMSNorm(norm2) -> 前馈网络(FeedForward 或 MoEFeedForward) -> 残差相加

    根据配置中是否存在有效的 "num_experts"，前馈子层会分别实例化为
    稠密的 FeedForward 或 稀疏的 MoEFeedForward（混合专家）。

    参数:
        cfg (dict): 模型配置字典。
    """
    def __init__(self, cfg):
        super().__init__()
        # 分组查询注意力（GQA）子层
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        # 若配置中显式声明了专家数量（num_experts > 0），说明这是 MoE 模型，
        # 使用混合专家前馈网络；否则使用普通的稠密 SwiGLU 前馈网络
        if "num_experts" in cfg and cfg["num_experts"] > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForward(cfg)
        # 注意力子层前的归一化（对应 HuggingFace 的 input_layernorm）
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        # 前馈子层前的归一化（对应 HuggingFace 的 post_attention_layernorm）
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        """
        参数:
            x (torch.Tensor): 输入隐藏状态，形状 [batch_size, num_tokens, emb_dim]。
            mask (torch.BoolTensor): 因果注意力掩码，形状 [num_tokens, num_tokens]。
            cos, sin (torch.Tensor): RoPE 预计算的余弦/正弦表，形状 (context_length, head_dim)。

        返回:
            torch.Tensor: 输出隐藏状态，形状与输入 x 相同 [batch_size, num_tokens, emb_dim]。
        """
        # Shortcut connection for attention block
        # 保存残差分支的输入（Pre-Norm：先归一化，归一化结果用于计算，
        # 但残差相加时使用归一化之前的原始输入）
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin,)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        # 前馈子层同样使用 Pre-Norm + 残差连接的结构
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x


class FeedForward(nn.Module):
    """
    稠密（非 MoE）前馈网络，使用 SwiGLU 激活结构：
        FFN(x) = fc3( SiLU(fc1(x)) * fc2(x) )

    其中 fc1、fc2 将 emb_dim 投影到 hidden_dim（分别称为“门控”和“上投影”），
    二者逐元素相乘后再由 fc3 投影回 emb_dim。三个线性层均无偏置。

    参数:
        cfg (dict): 需包含 "emb_dim"、"hidden_dim"、"dtype"。
    """
    def __init__(self, cfg):
        super().__init__()
        # fc1: emb_dim -> hidden_dim，经 SiLU 激活后作为门控信号
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # fc2: emb_dim -> hidden_dim，作为与门控相乘的“值”分支
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # fc3: hidden_dim -> emb_dim，将中间表示投影回原始维度
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        """
        参数:
            x (torch.Tensor): 形状 [batch_size, num_tokens, emb_dim]。
        返回:
            torch.Tensor: 形状 [batch_size, num_tokens, emb_dim]。
        """
        x_fc1 = self.fc1(x)  # [batch, num_tokens, hidden_dim]
        x_fc2 = self.fc2(x)  # [batch, num_tokens, hidden_dim]
        # SwiGLU：SiLU(fc1(x)) 作为门控，逐元素乘以 fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)  # 投影回 emb_dim


class MoEFeedForward(nn.Module):
    """
    混合专家（Mixture-of-Experts, MoE）前馈网络。

    与稠密 FeedForward 不同，MoE 拥有多个（num_experts 个）独立的 SwiGLU
    专家子网络，每个 token 并不会经过所有专家，而是由一个可学习的路由
    （gate，线性层）为其打分，选出得分最高的 Top-K（num_experts_per_tok）个
    专家，对这些专家的输出做加权（softmax 概率加权）求和。这样可以在增大
    模型总参数量的同时，保持每个 token 实际计算量（激活参数量）不变，
    是稀疏激活的核心思想。

    参数:
        cfg (dict): 需包含 "emb_dim"、"num_experts"、"num_experts_per_tok"、
            "moe_intermediate_size"、"dtype"。
    """
    def __init__(self, cfg):
        super().__init__()
        # 每个 token 激活的专家个数（Top-K 中的 K）
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        # 专家总数
        self.num_experts = cfg["num_experts"]
        self.emb_dim = cfg["emb_dim"]
        # 路由/门控线性层：emb_dim -> num_experts，为每个 token 对每个专家打分
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"])

        # 为每个专家分别创建一套 SwiGLU 前馈网络的三个线性层（fc1 门控、
        # fc2 上投影、fc3 下投影），存放在 ModuleList 中，按专家下标索引
        self.fc1 = nn.ModuleList([nn.Linear(cfg["emb_dim"], cfg["moe_intermediate_size"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        self.fc2 = nn.ModuleList([nn.Linear(cfg["emb_dim"], cfg["moe_intermediate_size"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        self.fc3 = nn.ModuleList([nn.Linear(cfg["moe_intermediate_size"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])

    def forward(self, x):
        """
        参数:
            x (torch.Tensor): 形状 [batch_size, seq_len, emb_dim]。

        返回:
            torch.Tensor: 形状 [batch_size, seq_len, emb_dim]，为各激活专家
                输出的加权（按路由 softmax 概率）求和结果。

        路由与稀疏计算流程：
            1. 用 gate 线性层给每个 token 对所有专家打分；
            2. 取 Top-K（num_experts_per_tok）个专家及其分数；
            3. 对这 K 个分数做 softmax，得到分配给每个被选中专家的权重；
            4. 按“专家”而不是按“token”循环：对每个在本 batch 中被至少
               一个 token 选中的专家，收集所有选中它的 token，一次性批量
               通过该专家的 SwiGLU 网络，再按对应权重加权累加回输出。
               这种按专家分组计算的方式避免了对未被选中的专家做无效计算。
        """
        scores = self.gate(x)  # (b, seq_len, num_experts) 每个 token 对每个专家的路由分数
        # 选出每个 token 分数最高的 K 个专家：
        # topk_scores/topk_indices 形状均为 (b, seq_len, num_experts_per_tok)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        # 只在被选中的 K 个专家范围内做 softmax，得到归一化的路由权重
        topk_probs = torch.softmax(topk_scores, dim=-1)

        batch, seq_len, _ = x.shape
        # 将 batch 和 seq_len 展平成一维“token 列表”，方便按专家分组处理
        x_flat = x.reshape(batch * seq_len, -1)  # (batch*seq_len, emb_dim)
        # 输出累加缓冲区，初始为全零，之后用 index_add_ 按 token 位置累加各专家贡献
        out_flat = torch.zeros(batch * seq_len, self.emb_dim, device=x.device, dtype=x.dtype)

        # 展平后的 Top-K 专家下标与对应权重，形状均为 (batch*seq_len, num_experts_per_tok)
        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)

        # 本次前向中，实际被至少一个 token 选中过的专家下标集合（去重）
        unique_experts = torch.unique(topk_indices_flat)

        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())
            # mask: (batch*seq_len, num_experts_per_tok)，标记每个 token 的
            # K 个候选专家槽位中，哪些槽位恰好选中了当前 expert_id
            mask = topk_indices_flat == expert_id
            if not mask.any():
                continue

            # token_mask: (batch*seq_len,)，标记哪些 token 选中了当前专家
            # （只要 K 个候选中有一个命中即为 True）
            token_mask = mask.any(dim=-1)
            # 选中当前专家的 token 在展平序列中的索引
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            # 收集这些 token 的输入向量：(num_selected, emb_dim)
            expert_input = x_flat.index_select(0, selected_idx)
            # 该专家的 SwiGLU 前馈计算：SiLU(fc1(x)) * fc2(x)，
            # 中间维度为 moe_intermediate_size
            hidden = torch.nn.functional.silu(self.fc1[expert_id](expert_input)) * self.fc2[expert_id](expert_input)
            # 投影回 emb_dim：(num_selected, emb_dim)
            expert_out = self.fc3[expert_id](hidden)

            # 从 mask 中取出被选中 token 对应的行，用于定位该专家在其
            # Top-K 候选列表中所处的槽位（每个 token 的 K 个候选中，
            # 当前专家占据哪一个位置），从而取出对应的路由权重
            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            # 按槽位取出每个选中 token 分配给当前专家的 softmax 权重
            selected_probs = torch.gather(topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices).squeeze(-1)

            # 将该专家的加权输出累加回对应 token 在 out_flat 中的位置
            # （同一个 token 可能被多个专家选中，因此用 index_add_ 累加
            # 而不是直接赋值）
            out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

        # 恢复为 (batch, seq_len, emb_dim) 形状
        return out_flat.reshape(batch, seq_len, self.emb_dim)


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（Grouped-Query Attention, GQA），并可选开启 QK-Norm。

    GQA 是多头注意力（MHA）与多查询注意力（MQA）之间的折中：Query 仍然使用
    完整的 num_heads 个头，但 Key/Value 只使用较少的 num_kv_groups 个“组”，
    每组内的多个 Query 头共享同一份 Key/Value（group_size = num_heads //
    num_kv_groups）。这样可以显著减少推理时 KV Cache 的显存占用，同时相比
    MQA 保留更多的表达能力。

    QK-Norm：在计算注意力分数之前，分别对每个头的 Query、Key 向量做一次
    RMSNorm（沿 head_dim 维度），这是 Qwen3 用于提升训练稳定性的技巧。

    参数:
        d_in (int): 输入特征维度（等于模型的 emb_dim）。
        num_heads (int): Query 的注意力头数。
        num_kv_groups (int): Key/Value 的分组数（必须能整除 num_heads）。
        head_dim (int, optional): 每个头的维度；若为 None 则用 d_in //
            num_heads 推导。
        qk_norm (bool): 是否对 Q、K 做 RMSNorm。
        dtype: 线性层参数的存储精度。
    """
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        # num_heads 必须是 num_kv_groups 的整数倍，这样每组能均分到相同
        # 数量的 Query 头（即下面的 group_size）
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        # 每个 KV 组对应多少个 Query 头，后续用于 repeat_interleave 扩展 K/V
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        # 所有 Query 头拼接后的总维度（可能与 d_in 不同，因为 head_dim
        # 可以独立于 d_in / num_heads 配置）
        self.d_out = num_heads * head_dim

        # Query 投影：d_in -> num_heads * head_dim
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        # Key/Value 投影：注意输出维度只有 num_kv_groups * head_dim，
        # 远小于 Query 的 num_heads * head_dim，这正是 GQA 节省显存的关键
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        # 输出投影：把拼接后的多头注意力结果 d_out 映射回模型维度 d_in
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            # 对每个头内部的 head_dim 维度做 RMSNorm（注意 eps 与 emb_dim
            # 的 RMSNorm 不同，这里用更小的 1e-6）
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        """
        参数:
            x (torch.Tensor): 输入隐藏状态，形状 [b, num_tokens, d_in]。
            mask (torch.BoolTensor): 因果掩码，形状 [num_tokens, num_tokens]，
                True 的位置会被填充为 -inf（即禁止关注）。
            cos, sin (torch.Tensor): RoPE 预计算表，形状 (context_length, head_dim)。

        返回:
            torch.Tensor: 注意力输出，形状 [b, num_tokens, d_in]。
        """
        b, num_tokens, _ = x.shape

        # Apply projections
        # 线性投影得到 Q/K/V（此时头维度尚未拆分）
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        # 拆分出头维度，并将头维度换到第二轴，形状变为
        # (b, num_heads, num_tokens, head_dim)；K/V 同理但头数是 num_kv_groups
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        # QK-Norm：沿最后一维 head_dim 对 Q、K 做 RMSNorm，形状不变
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        # 对 Q、K 施加旋转位置编码，注入相对位置信息；形状保持不变
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        # GQA 的核心步骤：把 K/V 沿“头”维度重复 group_size 次，使其头数
        # 从 num_kv_groups 扩展为 num_heads，从而能与 Query 的每个头一一
        # 对应做点积注意力。repeat_interleave 保证同一组内相邻的 Query 头
        # 共享同一份 K/V（而不是简单地 tile 整体重复）
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        # 缩放点积注意力：Q @ K^T，形状 (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        # 用因果掩码把未来位置的分数置为 -inf，softmax 后对应权重趋近于 0
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        # 按 head_dim 的平方根缩放后做 softmax，得到注意力权重
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        # 加权求和得到上下文向量，再把多头维度换回并拼接：
        # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        # -> reshape 为 (b, num_tokens, d_out)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        # 输出投影回模型维度 d_in
        return self.out_proj(context)


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
#
# 中文补充说明（新增）：本文件采用上面第 1 种“前后两半（split-halves）”风格
# 实现 RoPE，与 Hugging Face Transformers 保持一致。即把 head_dim 均分为
# 前半 x1 和后半 x2 两部分，分别与 cos/sin 表结合做旋转，而不是把奇偶维度
# 交错配对（第 2 种风格，Llama 原始实现所用）。两种写法在数学上等价，
# 只是维度配对方式不同，实现时不能混用，否则加载的权重与实现方式不匹配
# 会导致结果错误。


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """
    预计算 RoPE（旋转位置编码）所需的余弦、正弦表。

    RoPE 通过给每一对维度赋予一个随位置线性增长、随维度指数衰减的旋转角度，
    使得两个位置的 Query/Key 做点积后，结果只依赖于它们的相对位置差，
    从而免去显式的位置嵌入相加操作。

    参数:
        head_dim (int): 单个注意力头的维度，必须为偶数（因为要成对旋转）。
        theta_base (float): RoPE 频率公式中的底数 theta（Qwen3 中较大，
            例如 1e6 或 1e7，用于支持更长的上下文）。
        context_length (int): 需要预计算的最大位置数（序列长度上限）。
        dtype: 计算所用/返回的张量精度。

    返回:
        (cos, sin): 两个形状均为 (context_length, head_dim) 的张量，
            表示每个位置在每个维度上对应的旋转角度的余弦、正弦值。
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    # 频率随维度指数衰减：inv_freq[i] = theta_base^(-2i/head_dim)，
    # 共 head_dim // 2 个频率值，维度越靠后频率越低（旋转越慢）
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    # 位置索引 0, 1, ..., context_length-1
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    # 外积：每个位置 × 每个频率，得到该位置在该频率下的旋转角度
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    # 将角度矩阵在维度方向上复制一份并拼接（前半、后半使用相同角度），
    # 对应“split-halves”风格：head_dim 前一半和后一半共享同一组角度
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    # 预先算好 cos/sin，避免在每次前向传播时重复计算三角函数
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    将预计算好的 RoPE 余弦/正弦表应用到 Query 或 Key 张量上。

    采用“前后两半（split-halves）”风格：把 head_dim 均分为前半 x1、
    后半 x2，构造“旋转后”的向量 rotated = concat(-x2, x1)，再与原始
    x 分别乘以 cos、sin 并相加，等价于对每一对 (x1[i], x2[i]) 做
    二维旋转变换。

    参数:
        x (torch.Tensor): 待旋转的 Query/Key 张量，
            形状 (batch_size, num_heads, seq_len, head_dim)。
        cos, sin (torch.Tensor): compute_rope_params 预计算的表，
            形状 (context_length, head_dim)，这里会按当前 seq_len 截取。

    返回:
        torch.Tensor: 应用 RoPE 后的张量，形状与输入 x 相同。
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    # 把 head_dim 一分为二：前半 x1、后半 x2，各占 head_dim // 2
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2:]  # Second half

    # Adjust sin and cos shapes
    # 按实际序列长度截取 cos/sin，并广播出 batch 和 num_heads 维度：
    # (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    # 构造旋转向量：[-x2, x1]，与 x 本身按位置对应的 cos/sin 组合，
    # 等价于对每一对维度 (x1, x2) 做二维旋转：
    #   new_x1 = x1*cos - x2*sin,  new_x2 = x2*cos + x1*sin
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    # 旋转计算本身用较高精度更稳妥，但结果转换回原始（可能是低精度）dtype
    # 不会显著影响效果，可以节省显存
    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    """
    均方根归一化（Root Mean Square Layer Normalization）。

    与标准 LayerNorm 不同，RMSNorm 不减去均值，只用均方根（RMS）对输入
    做缩放归一化，然后乘以可学习的缩放参数 scale（以及可选的偏置 shift）。
    计算量更小，且在 Qwen3 等模型中被证明同样有效。

    参数:
        emb_dim (int): 归一化的特征维度（沿最后一维计算方差）。
        eps (float): 防止除零的小常数。
        bias (bool): 是否使用可学习的偏置项（Qwen3 默认不使用）。
        qwen3_compatible (bool): 若为 True，则在计算前将输入强制转换为
            float32 以保证数值稳定性（与 Qwen3 官方实现的行为对齐），
            计算完成后再转换回原始 dtype。
    """
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        # 可学习缩放参数，初始化为全 1，形状 (emb_dim,)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 可选的可学习偏置参数，初始化为全 0；若 bias=False 则为 None
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        """
        参数:
            x (torch.Tensor): 输入张量，最后一维大小为 emb_dim。

        返回:
            torch.Tensor: 归一化后的张量，形状与输入相同，dtype 恢复为
                输入原始的 dtype。
        """
        input_dtype = x.dtype

        if self.qwen3_compatible:
            # 提升到 float32 计算，避免低精度（如 bfloat16）下方差计算
            # 精度不足导致的数值不稳定
            x = x.to(torch.float32)

        # 沿最后一维计算均方值（不减均值），再取平方根的倒数（rsqrt）
        # 作为缩放系数，实现“除以均方根”的归一化
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        # 乘以可学习缩放参数
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        # 转换回输入原始的 dtype（例如 bfloat16），以便与模型其他部分
        # 精度保持一致
        return norm_x.to(input_dtype)


def load_weights_into_qwen(model, param_config, params):
    """
    将 HuggingFace 格式（例如从 safetensors 加载得到的扁平化权重字典）
    的 Qwen3 权重，逐一拷贝到本文件定义的 Qwen3Model 结构对应的参数中。

    处理的内容包括：
        - 词嵌入权重（model.embed_tokens.weight）
        - 每一层的 Q/K/V/O 注意力投影权重
        - 可选的 QK-Norm 缩放参数（q_norm/k_norm）
        - 每一层的输入/输出前的两个 RMSNorm 缩放参数
        - 前馈网络权重：根据 param_config 中是否配置了 num_experts，
          分别按“稠密 FeedForward”或“MoE（路由 + 多专家）”两种方式加载
        - 最终归一化权重与输出头权重（若无独立 lm_head，则使用权重绑定，
          即让输出头与词嵌入共享同一份权重矩阵）

    参数:
        model (Qwen3Model): 待加载权重的模型实例（结构需与 param_config 匹配）。
        param_config (dict): 模型配置字典（如 QWEN_CONFIG_06_B），用于确定
            层数、是否为 MoE 等结构信息。
        params (dict): 形如 {权重名称字符串: 权重张量/数组} 的扁平字典，
            键名遵循 HuggingFace Qwen3 的命名规则
            （如 "model.layers.0.self_attn.q_proj.weight"）。

    返回:
        None（原地修改 model 的参数）。
    """
    def assign(left, right, tensor_name="unknown"):
        """
        内部工具函数：将 right（待加载的权重）安全地拷贝进 left
        （模型中已存在的 nn.Parameter），并做形状校验。

        参数:
            left (torch.Tensor): 模型中的目标参数张量。
            right: 待加载的权重，可以是 torch.Tensor，也可以是其他
                可转换为张量的对象（如 numpy 数组）。
            tensor_name (str): 用于报错信息中标识具体是哪个张量。

        返回:
            torch.Tensor: 拷贝完成后的 left（其 .data 已被原地更新）。
        """
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                # 直接原地拷贝张量内容（不改变 left 本身的 dtype/device）
                left.copy_(right)
            else:
                # 非张量类型（例如 numpy 数组）先转换为与 left 相同
                # dtype/device 的张量，再拷贝
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    # 加载词嵌入权重：形状应为 (vocab_size, emb_dim)
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att

        # Q, K, V projections
        # Query 投影权重：形状 (num_heads*head_dim, emb_dim)
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        # Key 投影权重：形状 (num_kv_groups*head_dim, emb_dim)，
        # 明显小于 Query 投影，体现 GQA 的“压缩 KV”特点
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Output projection
        # 注意力输出投影权重：形状 (emb_dim, num_heads*head_dim)
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK norms
        # 若该层启用了 QK-Norm，则加载对应的 RMSNorm 缩放参数
        # （形状均为 (head_dim,)）
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Attention layernorm
        # 注意力子层前的 RMSNorm（对应 HuggingFace 的 input_layernorm）
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward weights
        # 根据配置判断是否为 MoE 模型，走不同的前馈权重加载分支
        if param_config.get("num_experts", 0) > 0:
            # Load router (gating) weights
            # 加载路由（门控）权重：形状 (num_experts, emb_dim)
            block.ff.gate.weight = assign(
                block.ff.gate.weight,
                params[f"model.layers.{l}.mlp.gate.weight"],
                f"model.layers.{l}.mlp.gate.weight"
            )
            # Load expert weights
            # 依次加载每一个专家的三个投影权重：
            # gate_proj -> 本实现中的 fc1，up_proj -> fc2，down_proj -> fc3
            for e in range(param_config["num_experts"]):
                prefix = f"model.layers.{l}.mlp.experts.{e}"
                block.ff.fc1[e].weight = assign(
                    block.ff.fc1[e].weight,
                    params[f"{prefix}.gate_proj.weight"],
                    f"{prefix}.gate_proj.weight"
                )
                block.ff.fc2[e].weight = assign(
                    block.ff.fc2[e].weight,
                    params[f"{prefix}.up_proj.weight"],
                    f"{prefix}.up_proj.weight"
                )
                block.ff.fc3[e].weight = assign(
                    block.ff.fc3[e].weight,
                    params[f"{prefix}.down_proj.weight"],
                    f"{prefix}.down_proj.weight"
                )

        else:
            # 稠密（非 MoE）前馈网络权重加载：
            # gate_proj -> fc1，up_proj -> fc2，down_proj -> fc3
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )

        # 前馈子层前的 RMSNorm（对应 HuggingFace 的 post_attention_layernorm）
        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    # 加载最终归一化层的缩放参数
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        # 若权重字典中存在独立的语言模型头权重，则直接加载
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        # 否则说明该模型使用“权重绑定”（weight tying）：输出头与词嵌入
        # 共享同一份权重矩阵（都是 (vocab_size, emb_dim) 形状），
        # 常见于较小规模的模型以节省参数量
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")


class Qwen3Tokenizer:
    """
    Qwen3 分词器的轻量封装，基于 HuggingFace `tokenizers` 库的
    `Tokenizer.from_file` 加载 tokenizer.json，并在此基础上叠加：
        - Qwen3 专用特殊符号（聊天标记、视觉/多模态占位符、思考标签等）
          与其 token id 的映射；
        - 可选的聊天模板包装（把用户输入包装成
          "<|im_start|>user\\n...<|im_end|>\\n..." 的格式）；
        - 思考模式（add_thinking）开关，控制是否在生成提示中插入
          空的 <think></think> 块。

    属性:
        pad_token_id (int): 填充符 id，固定使用 "<|endoftext|>" 对应的 id。
        eos_token_id (int): 结束符 id；对于非 Base（即对话/指令）模型，
            优先使用 "<|im_end|>"，否则退回 "<|endoftext|>"。
    """
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<think>", "</think>"
    ]
    # 用于在编码前，按特殊符号（如 <|im_start|> 或 <think>）切分原始文本的
    # 正则表达式；括号保留分隔符本身，使 re.split 结果中包含这些特殊符号
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None,
                 apply_chat_template=True, add_generation_prompt=False, add_thinking=False):
        """
        参数:
            tokenizer_file_path (str): 本地 tokenizer.json 文件路径。
            repo_id (str, optional): HuggingFace 仓库 id；若本地文件不存在
                且提供了 repo_id，会自动从 HuggingFace Hub 下载。
            apply_chat_template (bool): encode() 默认是否包装聊天模板。
            add_generation_prompt (bool): 是否在末尾追加
                "<|im_start|>assistant\\n" 提示，引导模型开始生成回复。
            add_thinking (bool): 为 True 时生成提示后不预填 <think></think>
                空块（让模型自己决定是否思考）；为 False 时会预先插入
                一个空的 <think>\\n\\n</think> 块（跳过显式思考）。
        """
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_file = Path(tokenizer_file_path)
        if not tok_file.is_file() and repo_id:
            # 本地不存在分词器文件但提供了 repo_id，则尝试从 HuggingFace 下载
            download_from_huggingface(
                repo_id=repo_id,
                filename=tok_file.name,
                local_dir=str(tok_file.parent),
            )
        self._tok = Tokenizer.from_file(str(tok_file))
        self._special_to_id = {}
        for t in self._SPECIALS:
            tid = self._tok.token_to_id(t)
            if tid is not None:
                # 只记录该分词器词表中实际存在的特殊符号，避免不同模型
                # 变体（可能缺少某些多模态占位符）导致 KeyError
                self._special_to_id[t] = tid

        # 填充符固定为 <|endoftext|> 对应的 id
        self.pad_token_id = self._special_to_id["<|endoftext|>"]
        self.eos_token_id = self.pad_token_id

        if repo_id and "Base" not in repo_id:
            # 非 Base（即经过指令/对话微调）的模型，通常用 <|im_end|>
            # 作为对话轮次结束符
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        """
        将文本编码为 token id 列表。

        参数:
            text (str): 待编码的原始文本。
            chat_wrapped (bool, optional): 是否用聊天模板包装该文本；
                若为 None，则使用初始化时的 self.apply_chat_template。

        返回:
            list[int]: 编码得到的 token id 序列。
        """
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            # 若整段文本（去除首尾空白后）恰好就是一个特殊符号本身
            # （且不含换行），直接返回该符号的 id，不做聊天模板包装
            # 或常规分词
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []
        # 先按特殊符号把文本切成若干片段（特殊符号本身也会作为独立片段
        # 保留在切分结果中），filter(None, ...) 去掉切分产生的空字符串
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                # 片段本身就是特殊符号：直接查表得到对应 id，不经过
                # 普通 BPE/subword 编码
                ids.append(self._special_to_id[part])
            else:
                # 普通文本片段：交给底层 tokenizer 做常规编码
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        """
        将 token id 列表解码回文本，保留特殊符号（不跳过）。

        参数:
            ids (list[int]): token id 序列。
        返回:
            str: 解码得到的文本。
        """
        return self._tok.decode(ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        """
        将用户消息包装成 Qwen3 聊天模板格式。

        参数:
            user_msg (str): 用户原始输入文本。
        返回:
            str: 包装后的完整提示文本，形如：
                "<|im_start|>user\\n{user_msg}<|im_end|>\\n"
                （若 add_generation_prompt 为 True，还会追加助手轮次的
                起始标记，以及可选的空 <think></think> 块）。
        """
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                # 开启思考模式：只给出换行，留空由模型自行生成
                # <think>...</think> 思考内容
                s += "\n"
            else:
                # 关闭思考模式：直接预填一个空的思考块，提示模型跳过
                # 显式思考，直接给出最终回答
                s += "\n<think>\n\n</think>\n\n"
        return s


def download_from_huggingface(repo_id, filename, local_dir, revision="main"):
    """
    从 HuggingFace Hub 下载单个文件（例如 tokenizer.json）到本地目录。

    使用 HuggingFace 的 "resolve" 直链下载方式（而非 huggingface_hub
    的高层 API），若目标文件已存在则跳过下载。

    参数:
        repo_id (str): HuggingFace 仓库 id，例如 "Qwen/Qwen3-0.6B"。
        filename (str): 需要下载的文件名。
        local_dir (str): 本地保存目录，会自动创建（含父目录）。
        revision (str): 分支/版本名，默认 "main"。

    返回:
        str: 下载（或已存在）文件的本地路径。
    """
    base_url = "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/{revision}/{filename}"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    dest_path = os.path.join(local_dir, filename)

    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
    else:
        print(f"Downloading {url} to {dest_path}...")
        # stream=True 配合分块写入，避免一次性把整个文件读入内存
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return dest_path


def download_from_huggingface_from_snapshots(repo_id, local_dir):
    """
    使用 huggingface_hub 的 snapshot_download 下载整个模型仓库快照，
    并加载其中的 safetensors 权重（支持单文件与多分片两种情况），
    返回扁平化的权重字典，可直接传给 load_weights_into_qwen 使用。

    参数:
        repo_id (str): HuggingFace 仓库 id。
        local_dir (str): 本地缓存/保存目录。

    返回:
        dict[str, torch.Tensor]: 形如 {权重名称: 张量} 的完整权重字典。

    异常:
        FileNotFoundError: 既找不到分片索引文件
            (model.safetensors.index.json) 也找不到单文件
            (model.safetensors) 时抛出。
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file  # or your preferred loader

    # 下载（或复用缓存的）整个仓库快照，返回本地目录路径
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)

    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    single_file_path = os.path.join(repo_dir, "model.safetensors")

    if os.path.exists(index_path):
        # Multi-shard model
        # 多分片模型：索引文件记录了每个权重名称对应存放在哪个分片文件中
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        # set(...) 去重，避免同一个分片文件被重复加载
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)
    elif os.path.exists(single_file_path):
        # Single-shard model
        # 单文件模型：直接下载/定位唯一的 model.safetensors 并加载
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
    else:
        raise FileNotFoundError("No model.safetensors or model.safetensors.index.json found.")

    return weights_dict
