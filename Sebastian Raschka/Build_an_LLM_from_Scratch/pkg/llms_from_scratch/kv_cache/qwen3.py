# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明（新增，原文件无模块级 docstring，此处补充，不影响任何可执行代码）：

本文件实现了 **带 KV 缓存（KV Cache）的 Qwen3 模型**，是《Build a Large Language Model
From Scratch》一书配套代码在推理场景下的优化版本。相较于训练版实现，本文件的核心变化是：
在自回归生成（autoregressive generation）时，通过缓存历史 token 的 Key/Value 张量，
避免每一步都重新计算全部历史位置的 K/V 投影，从而将逐 token 生成的计算复杂度从
O(n^2) 降低为摊销的 O(n)。

涉及的关键技术点（后续在对应类/函数处会有更细致的中文注释）：
1. RoPE（Rotary Position Embedding，旋转位置编码）：
   通过复数旋转的方式将位置信息编码进 Query/Key 向量，具有良好的外推性和相对位置特性。
2. RMSNorm（Root Mean Square LayerNorm）：
   相比标准 LayerNorm 去掉了均值中心化，只用均方根做归一化，计算更轻量，是 Qwen3/LLaMA
   系列模型的标准归一化方式。
3. QK-Norm（Query/Key Normalization）：
   在计算注意力之前，对 Query 和 Key 分别做一次 RMSNorm，用于稳定训练/推理时的数值范围，
   是 Qwen3 相较于 LLaMA 的一个改进点。
4. GQA（Grouped Query Attention，分组查询注意力）：
   Query 头数多于 Key/Value 头数，多个 Query 头共享同一组 Key/Value，
   在保持模型表达能力的同时大幅减少 KV 缓存的显存占用。
5. KV Cache（键值缓存）：
   在生成阶段，把每一层已经计算过的 Key/Value 张量沿序列维度拼接保存，
   新的 token 只需要计算自己的 Q/K/V，然后与缓存中的历史 K/V 做注意力计算即可。
6. MoE（Mixture-of-Experts，混合专家，若配置中开启）：
   对于配置了 `num_experts` 的模型，前馈网络（FeedForward）替换为由多个专家网络组成的
   MoE 层，每个 token 只激活 top-k 个专家，从而在增大模型参数量的同时控制实际计算量。

以下代码除新增的中文注释/docstring 外，其余可执行代码（变量名、函数签名、逻辑顺序、
缩进、字符串、import 语句等）均与原文件保持完全一致，不做任何修改。
"""

# 从同目录下的 utils 模块导入 KVCache 类（用于在外部管理/组织每一层的 KV 缓存）
# noqa: F401 表示这里虽然没有直接使用该名字，也不要触发"未使用导入"的静态检查警告
from .utils import KVCache   # noqa: F401
# 从上一级目录的 qwen3 模块导入：各规格模型的配置字典、分词器、权重加载函数、
# 以及从 HuggingFace 下载模型权重相关的工具函数。这些内容在"非 KV Cache 版"的
# qwen3.py 中定义，这里直接复用，避免重复实现。
from ..qwen3 import (   # noqa: F401
    QWEN_CONFIG_06_B, QWEN3_CONFIG_1_7B, QWEN3_CONFIG_4B,
    QWEN3_CONFIG_8B, QWEN3_CONFIG_14B, QWEN3_CONFIG_32B,
    Qwen3Tokenizer, load_weights_into_qwen,
    download_from_huggingface,
    download_from_huggingface_from_snapshots
)

import torch
import torch.nn as nn


class Qwen3Model(nn.Module):
    """Qwen3 语言模型的顶层容器（支持 KV Cache 的推理版本）。

    整体结构为：Token Embedding -> N 层 TransformerBlock（含 GQA 注意力 + FeedForward/MoE）
    -> 最终 RMSNorm -> 输出线性层（映射到词表大小的 logits）。

    与训练版模型的主要区别：
    - `forward` 方法接受可选的 `cache` 参数（一个类似 KVCache 的对象），用于在自回归
      生成时复用之前计算好的 Key/Value，避免重复计算历史 token 的注意力键值；
    - 内部维护 `current_pos` 用于记录当前已经生成到的序列位置，从而在增量推理时
      正确地为 RoPE 计算位置偏移，以及正确构造因果注意力掩码（causal mask）。

    Args:
        cfg (dict): 模型配置字典，需包含 vocab_size、emb_dim、n_layers、n_heads、
            head_dim、n_kv_groups、qk_norm、rope_base、context_length、dtype 等键。
    """
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        # 词嵌入层：将输入的 token id 映射为 emb_dim 维的向量
        # 输入形状 (batch, num_tokens) -> 输出形状 (batch, num_tokens, emb_dim)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        # 堆叠 n_layers 个 Transformer 块。
        # 使用 ModuleList 而不是 nn.Sequential，是因为 Sequential 的 forward 只接受单一输入，
        # 而这里每个 TransformerBlock 的 forward 需要同时接收 x、mask、cos、sin 等多个参数。
        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # 最终输出前的 RMSNorm，对最后一层 Transformer 块的输出做归一化
        self.final_norm = RMSNorm(cfg["emb_dim"])
        # 输出投影层（也称为 LM Head）：将隐藏状态映射为词表大小的 logits，不使用偏置项
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        # 计算每个注意力头的维度 head_dim：若配置未显式指定，则用 emb_dim // n_heads 推导
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        # 预先计算 RoPE（旋转位置编码）所需的 cos/sin 表，覆盖整个 context_length 范围，
        # 这样在前向传播时可以直接按位置切片查表，无需重复计算三角函数
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        # 注册为 buffer（非可训练参数），且 persistent=False 表示不会被保存进 state_dict，
        # 因为 cos/sin 完全可以根据配置重新计算出来，没必要占用 checkpoint 空间
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # Track current position in KV cache  # 记录 KV 缓存中当前已经写入到的位置（即已生成 token 的总数）

    def forward(self, in_idx, cache=None):
        """执行一次前向传播，支持"全量前向"（无缓存，训练/首次预填充）
        和"增量前向"（有缓存，自回归生成时逐 token 或分块推理）两种模式。

        Args:
            in_idx (torch.Tensor): 输入 token id，形状 (batch_size, num_tokens)。
                在使用 KV Cache 生成时，num_tokens 通常为 1（每步只喂入新生成的 1 个 token），
                或者在"预填充（prefill）"阶段为 prompt 的长度。
            cache: 可选，形如 KVCache 的缓存管理对象，提供 get(i)/update(i, ...) 接口，
                用于存取第 i 层 Transformer 块的 (keys, values) 缓存。若为 None，
                则表示不使用缓存，走标准的全量因果注意力计算。

        Returns:
            torch.Tensor: 输出 logits，形状 (batch_size, num_tokens, vocab_size)。
        """
        # Forward pass
        # 1) 词嵌入：token id -> 向量，形状 (batch, num_tokens, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        if cache is not None:
            # 增量推理模式：本次输入的 token 在整个序列中的起始/结束位置
            # 例如已经生成了 pos_start 个 token，本次新增 num_tokens 个，
            # 则新 token 占据 [pos_start, pos_end) 区间
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end  # 更新缓存位置指针，供下一次调用使用
            # 构造因果掩码：注意这里的掩码形状是 (pos_end, pos_end)，
            # 但通过切片 [pos_start:pos_end, :pos_end] 只取出"新 token 对所有历史+自身位置"的那部分，
            # 即 query 只有 num_tokens 行，但 key 维度覆盖了全部历史 pos_end 列
            # torch.triu(..., diagonal=1) 生成上三角（不含对角线）为 True 的矩阵，
            # True 位置会在注意力分数中被填充为 -inf，从而实现"看不到未来 token"的因果约束
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            # 无缓存模式（相当于一次性对整个序列做标准的因果自注意力）
            pos_start = 0  # Not strictly necessary but helps torch.compile  # 非严格必需，但有助于 torch.compile 做静态图优化
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        # Shape (1, 1, num_tokens, num_tokens) to broadcast across batch and heads
        # 增加 batch 维和 head 维（都是长度为 1 的维度），方便后续与
        # (batch, num_heads, num_tokens, kv_len) 形状的注意力分数张量做广播
        mask = mask[None, None, :, :]

        # 依次通过每一层 TransformerBlock
        for i, block in enumerate(self.trf_blocks):
            # 取出第 i 层对应的历史 KV 缓存（若 cache 为 None，则 blk_cache 也为 None）
            blk_cache = cache.get(i) if cache else None
            # 将当前隐藏状态、因果掩码、RoPE 的 cos/sin 表、起始位置、该层缓存传入
            # 返回新的隐藏状态 x，以及更新后的该层 KV 缓存 new_blk_cache
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                # 将本层新的 (keys, values) 写回缓存管理器，供下一次调用时复用
                cache.update(i, new_blk_cache)

        # 最终归一化
        x = self.final_norm(x)
        # 输出投影到词表维度，注意这里显式转换回配置指定的 dtype
        # （因为 RMSNorm 内部可能会临时转换为 float32 计算，输出时再转回原精度）
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        """重置 KV 缓存的位置指针，通常在开始一次新的生成序列之前调用，
        以便重新从位置 0 开始计数（配合外部同时清空 cache 中各层的 K/V 张量使用）。
        """
        self.current_pos = 0


class TransformerBlock(nn.Module):
    """单个 Transformer 解码器块（Pre-Norm 结构）。

    结构为：
        x -> RMSNorm(norm1) -> GroupedQueryAttention -> 残差相加
          -> RMSNorm(norm2) -> FeedForward/MoEFeedForward -> 残差相加

    Qwen3 采用 Pre-Norm（先归一化再进入子层），并且两个子层各自独立使用一个 RMSNorm，
    这与 GPT-2 使用 LayerNorm 的经典 Pre-Norm Transformer 结构类似，只是把 LayerNorm
    换成了计算更轻量的 RMSNorm。

    Args:
        cfg (dict): 模型配置字典。
    """
    def __init__(self, cfg):
        super().__init__()
        # 注意力子层：分组查询注意力（GQA），支持可选的 QK-Norm
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        # 根据配置决定前馈网络是普通 FeedForward 还是 MoE（混合专家）版本
        if "num_experts" in cfg and cfg["num_experts"] > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForward(cfg)
        # 两个 RMSNorm 分别用于注意力子层前、前馈子层前（Pre-Norm），eps 用于数值稳定
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        """
        Args:
            x (torch.Tensor): 输入隐藏状态，形状 (batch_size, num_tokens, emb_dim)。
            mask (torch.Tensor): 因果注意力掩码，形状可广播为
                (1, 1, num_tokens, kv_len)。
            cos, sin (torch.Tensor): RoPE 预计算的余弦/正弦表，形状 (context_length, head_dim)。
            start_pos (int): 当前输入序列在完整序列中的起始位置索引，用于 RoPE 位置偏移。
            cache: 该层对应的 KV 缓存，形如 (prev_keys, prev_values) 元组或 None。

        Returns:
            (torch.Tensor, tuple): 更新后的隐藏状态（形状同输入 x），
                以及该层新的 KV 缓存（用于写回外部缓存管理器）。
        """
        # Shortcut connection for attention block
        shortcut = x  # 保存残差连接的输入
        x = self.norm1(x)  # Pre-Norm：先归一化再进入注意力子层
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back  # 残差相加，缓解深层网络的梯度消失问题

        # Shortcut connection for feed-forward block
        shortcut = x  # 保存前馈子层的残差输入
        x = self.norm2(x)  # Pre-Norm：先归一化再进入前馈子层
        x = self.ff(x)  # 前馈网络（或 MoE）变换
        x = x + shortcut  # Add the original input back  # 残差相加

        return x, next_cache


class FeedForward(nn.Module):
    """标准的 SwiGLU 风格前馈网络（Qwen3/LLaMA 系列常用结构）。

    与经典 Transformer 的 FFN（Linear -> ReLU/GELU -> Linear）不同，这里使用门控结构：
    对输入分别做两次线性投影 fc1、fc2，其中 fc1 的输出经过 SiLU 激活后与 fc2 的输出逐元素相乘
    （即 "门控"），再通过 fc3 投影回原始维度。这种结构（SwiGLU）在实践中通常比单纯的
    ReLU/GELU FFN 有更好的效果。

    维度变化：emb_dim -> hidden_dim（fc1、fc2 并行） -> emb_dim（fc3）
    """
    def __init__(self, cfg):
        super().__init__()
        # fc1、fc2 都将 emb_dim 映射到 hidden_dim，二者权重独立，不共享
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # fc3 将 hidden_dim 映射回 emb_dim，完成升维后再降维的前馈变换
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状 (batch_size, num_tokens, emb_dim)。
        Returns:
            torch.Tensor: 输出张量，形状与输入相同 (batch_size, num_tokens, emb_dim)。
        """
        x_fc1 = self.fc1(x)  # (b, num_tokens, hidden_dim)
        x_fc2 = self.fc2(x)  # (b, num_tokens, hidden_dim)
        # SwiGLU 门控：SiLU(fc1(x)) * fc2(x)，逐元素相乘，形状不变 (b, num_tokens, hidden_dim)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)  # 投影回 emb_dim: (b, num_tokens, emb_dim)


class MoEFeedForward(nn.Module):
    """混合专家（Mixture-of-Experts）前馈网络。

    与普通 FeedForward 的区别：这里维护 num_experts 个独立的专家子网络（每个都是一个
    SwiGLU 风格的小型 FFN），对每个 token，先由一个门控网络（gate）计算出对各专家的打分，
    取 top-k（num_experts_per_tok）个专家，只对这几个专家做计算，再按 softmax 后的权重
    加权求和得到最终输出。这样可以在保持较低实际计算量（激活参数量）的同时，
    大幅增加模型的总参数量（容量）。

    实现细节：为了效率，代码按"专家"而非按"token"循环——找出所有被选中的
    unique 专家 id，对每个专家一次性收集所有分配给它的 token，批量计算，
    再用 index_add_ 把结果加回到对应 token 的输出位置上。
    """
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]  # 每个 token 激活的专家数量（top-k 的 k）
        self.num_experts = cfg["num_experts"]  # 专家总数
        self.emb_dim = cfg["emb_dim"]
        # 门控（路由）网络：将 emb_dim 映射为 num_experts 个打分，用于选择激活哪些专家
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"])

        # 为每个专家分别创建一套 SwiGLU 风格前馈网络的三个线性层（fc1/fc2/fc3），
        # 使用 ModuleList 存放，下标即为专家 id
        self.fc1 = nn.ModuleList([nn.Linear(cfg["emb_dim"], cfg["moe_intermediate_size"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        self.fc2 = nn.ModuleList([nn.Linear(cfg["emb_dim"], cfg["moe_intermediate_size"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])
        self.fc3 = nn.ModuleList([nn.Linear(cfg["moe_intermediate_size"], cfg["emb_dim"], bias=False, dtype=cfg["dtype"])
                                  for _ in range(cfg["num_experts"])])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状 (batch_size, seq_len, emb_dim)。
        Returns:
            torch.Tensor: 输出张量，形状与输入相同 (batch_size, seq_len, emb_dim)。
        """
        scores = self.gate(x)  # (b, seq_len, num_experts)  # 门控网络输出每个 token 对每个专家的打分
        # 对每个 token，选出打分最高的 num_experts_per_tok 个专家及其打分值
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        # 对选中的 top-k 打分做 softmax，得到这几个专家的加权系数（归一化到和为 1）
        topk_probs = torch.softmax(topk_scores, dim=-1)

        batch, seq_len, _ = x.shape
        # 将 batch 和 seq_len 维度展平，方便按"token"为单位做索引选择和聚合
        x_flat = x.reshape(batch * seq_len, -1)  # (batch*seq_len, emb_dim)
        # 初始化输出容器，后续通过 index_add_ 累加各专家对各 token 的贡献
        out_flat = torch.zeros(batch * seq_len, self.emb_dim, device=x.device, dtype=x.dtype)

        # 展平 top-k 的专家索引和对应权重，形状变为 (batch*seq_len, num_experts_per_tok)
        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)

        # 找出本次前向传播中实际被选中过的所有专家 id（去重），避免遍历所有专家浪费计算
        unique_experts = torch.unique(topk_indices_flat)

        # 按专家分组批量计算：对每个被激活的专家，收集所有选中它的 token，一次性前向计算
        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())
            # mask: 形状 (batch*seq_len, num_experts_per_tok)，标记该 token 的 top-k 槽位中
            # 是否恰好是当前这个专家
            mask = topk_indices_flat == expert_id
            if not mask.any():
                continue

            # 对每个 token，只要它的 top-k 中包含当前专家，就在 token 维度上标记为 True
            token_mask = mask.any(dim=-1)
            # 取出被选中的 token 在展平序列中的下标
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            # 收集这些 token 的输入向量，形状 (num_selected, emb_dim)
            expert_input = x_flat.index_select(0, selected_idx)
            # 该专家的 SwiGLU 前馈计算：SiLU(fc1(x)) * fc2(x) -> fc3
            hidden = torch.nn.functional.silu(self.fc1[expert_id](expert_input)) * self.fc2[expert_id](expert_input)
            expert_out = self.fc3[expert_id](hidden)  # (num_selected, emb_dim)

            # 找到每个被选中 token 在其 top-k 槽位中，当前专家具体位于第几个槽位（用于取对应权重）
            mask_selected = mask[selected_idx]  # (num_selected, num_experts_per_tok)
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)  # (num_selected, 1)
            # 按槽位下标取出对应的门控权重（softmax 后的概率）
            selected_probs = torch.gather(topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices).squeeze(-1)

            # 将该专家的输出按权重加权后，累加回对应 token 在 out_flat 中的位置
            out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

        # 恢复原始的 (batch, seq_len, emb_dim) 形状
        return out_flat.reshape(batch, seq_len, self.emb_dim)


class GroupedQueryAttention(nn.Module):
    """分组查询注意力（Grouped Query Attention, GQA），并支持 KV Cache 增量推理。

    GQA 要点：
    - Query 使用 num_heads 个头；
    - Key/Value 只使用 num_kv_groups 个"组"（组数 <= num_heads），
      每组被 group_size = num_heads // num_kv_groups 个 Query 头共享；
    - 这样可以显著减少 Key/Value 的投影参数量以及 KV Cache 的显存占用
      （相较于每个 Query 头都有独立 K/V 的标准多头注意力 MHA），
      同时相较于所有 Query 头共享同一组 K/V 的 MQA（Multi-Query Attention），
      GQA 在效果和效率之间做了折中。

    QK-Norm 要点：
    - 若 qk_norm=True，会在对 Q、K 做 reshape（拆分成多头）之后、应用 RoPE 之前，
      分别对每个头的 Q、K 向量（最后一维 head_dim）做一次 RMSNorm，
      用于稳定注意力分数的数值范围，是 Qwen3 相较于 LLaMA 系列的一个改进点。

    KV Cache 要点：
    - forward 接收上一次调用留下的 cache=(prev_keys, prev_values)，
      将本次新计算出的 keys_new/values_new 沿"序列长度"维度（dim=2）与历史值拼接，
      得到完整的 keys/values 后再用于注意力计算，并把拼接后的结果作为 next_cache 返回，
      供下一次调用继续复用。
    """
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        # 要求 Query 头数必须能被 KV 组数整除，这样每组能被"整数个" Query 头均分共享
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        # 每个 KV 组被多少个 Query 头共享，例如 num_heads=8, num_kv_groups=2 时 group_size=4，
        # 表示每 4 个 Query 头共用同一组 Key/Value
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            # 若未显式指定每头维度，则要求 d_in 能被 num_heads 整除，并据此均分
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        # 所有 Query 头拼接后的总维度（多头注意力输出投影前的维度）
        self.d_out = num_heads * head_dim

        # Query 投影：输出维度为 num_heads * head_dim（每个头独立的 Q）
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        # Key/Value 投影：输出维度只有 num_kv_groups * head_dim（组数远小于头数，显著省参数/显存）
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        # 输出投影：把多头拼接后的注意力输出映射回 d_in 维度
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            # QK-Norm：分别对 Query、Key 的最后一维（head_dim）做 RMSNorm
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        """
        Args:
            x (torch.Tensor): 输入隐藏状态，形状 (b, num_tokens, d_in)。
            mask (torch.Tensor): 因果注意力掩码，可广播为 (1, 1, num_tokens, kv_len)。
            cos, sin (torch.Tensor): RoPE 的余弦/正弦查找表，形状 (context_length, head_dim)。
            start_pos (int): 当前输入序列相对于完整序列的起始位置，用于 RoPE 偏移；
                当 cache 为 None 时会被重置为 0（见下方代码）。
            cache: 可选的 (prev_keys, prev_values) 元组，历史 Key/Value 缓存，
                形状均为 (b, num_kv_groups, prev_len, head_dim)；若为 None 表示无缓存。

        Returns:
            (torch.Tensor, tuple): 注意力输出（形状 (b, num_tokens, d_in)），
                以及供下次调用复用的新缓存 next_cache = (keys, values)。
        """
        b, num_tokens, _ = x.shape

        # Apply projections
        # 分别做 Q/K/V 的线性投影
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        # 将最后一维拆分为 (头数, 每头维度)，再把"头"维度换到序列长度前面，
        # 变为惯用的 (batch, heads, seq_len, head_dim) 形状，便于后续做批量矩阵乘法
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # 注意 keys/values 用的是 num_kv_groups（组数）而非 num_heads（头数），
        # 这正是 GQA 与标准多头注意力（MHA）的关键区别所在
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # 此时形状：queries (b, num_heads, num_tokens, head_dim)
        #           keys_new/values_new (b, num_kv_groups, num_tokens, head_dim)

        # Optional normalization
        # QK-Norm：在应用 RoPE 之前，对 Q、K 的最后一维（head_dim）做 RMSNorm，
        # 有助于稳定数值范围、提升训练/推理稳定性
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # Apply RoPE
        # 对 Q、K 应用旋转位置编码。offset=start_pos 保证在增量推理时，
        # 新 token 使用的是其在完整序列中的"真实位置"对应的旋转角度，而不是从 0 开始
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            # KV Cache 命中：把历史 Key/Value 与本次新计算出的 Key/Value
            # 沿序列长度维度（dim=2）拼接，得到覆盖"历史 + 当前"全部位置的完整 K/V
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
            # 拼接后的结果即为下一次调用要使用的缓存
            next_cache = (keys, values)
        else:
            # 无缓存模式（例如训练，或不使用缓存的一次性全量前向）：
            # 直接用本次计算出的 K/V，不做任何拼接
            start_pos = 0  # reset RoPE  # 重置 RoPE 偏移（此分支不使用缓存，位置从 0 计起）
            keys, values = keys_new, values_new
            next_cache = (keys, values)

        # Expand K and V to match number of heads
        # 将 K/V 沿"头"维度重复 group_size 次，使其头数从 num_kv_groups 扩展到 num_heads，
        # 从而可以与 num_heads 个 Query 头一一对应做逐头的注意力计算。
        # repeat_interleave 保证同一组内相邻的 Query 头对应同一份被复制的 K/V
        # （而非简单 tile 整体重复），语义上等价于"多个 Query 头共享同一组 K/V"
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        # 扩展后形状：keys/values (b, num_heads, kv_len, head_dim)，kv_len 为历史+当前的总长度

        # Attention
        # 标准缩放点积注意力：Q @ K^T，得到形状 (b, num_heads, num_tokens, kv_len) 的注意力分数
        attn_scores = queries @ keys.transpose(2, 3)
        # 应用因果掩码：被掩盖的位置（未来 token）填充为 -inf，softmax 后趋近于 0
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        # 按 head_dim 的平方根缩放后做 softmax，得到注意力权重
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        # 注意力权重与 Value 加权求和，再把多头维度换回去并拼接成 d_out 维，
        # 形状: (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        # -> reshape 为 (b, num_tokens, d_out)
        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        # 最终通过输出投影层映射回 d_in 维度，同时返回本层更新后的 KV 缓存
        return self.out_proj(context), next_cache


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """预计算 RoPE（旋转位置编码）所需的余弦/正弦查找表。

    RoPE 的核心思想：把每个位置 p 对每一对特征维度 (2i, 2i+1) 施加一个角度为
    p * theta_i 的二维旋转变换，其中 theta_i = theta_base^(-2i/head_dim) 随维度指数衰减。
    这样，任意两个位置的 Query、Key 做点积时，结果只依赖于它们的"相对位置差"，
    从而天然具备相对位置编码的性质，且不需要额外的可学习参数。

    Args:
        head_dim (int): 每个注意力头的维度，必须为偶数（因为要按二维一组做旋转）。
        theta_base (float): RoPE 的基数（频率的底数），默认 10000，值越大低频分量衰减越慢。
        context_length (int): 需要预计算的最大位置数（即支持的最大序列长度）。
        dtype: 计算所用的数据类型。

    Returns:
        (torch.Tensor, torch.Tensor): cos、sin 两个查找表，形状均为 (context_length, head_dim)。
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    # 计算每一对维度对应的"逆频率" theta_i = theta_base^(-2i/head_dim)，
    # 形状为 (head_dim // 2,)，i 从 0 到 head_dim//2 - 1
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    # 生成位置索引 0, 1, ..., context_length - 1
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    # 位置索引与逆频率做外积，得到每个位置、每个频率分量对应的旋转角度
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    # 将角度矩阵复制一份并在最后一维拼接，使其维度从 head_dim//2 扩展到 head_dim，
    # 这样可以直接与拆分为"前半/后半"的 Q、K 向量做逐元素运算（对应 apply_rope 中的用法）
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    # 预先计算好 cos、sin 值，避免在每次前向传播时重复计算三角函数，提升推理效率
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    """将预计算好的 RoPE 旋转变换应用到输入张量（Query 或 Key）上。

    实现的是"旋转矩阵"的等价形式：对于把 head_dim 拆分成前半 x1、后半 x2 两部分，
    RoPE 变换等价于：
        x_rotated = x * cos + rotate_half(x) * sin
    其中 rotate_half(x) = concat(-x2, x1)。
    这与直接对每一对 (x_{2i}, x_{2i+1}) 做二维旋转矩阵乘法在数学上是等价的，
    只是通过"前后半交换取反再拼接"的技巧，用更少的张量操作实现了同样的效果。

    Args:
        x (torch.Tensor): 待旋转的张量（Query 或 Key），
            形状 (batch_size, num_heads, seq_len, head_dim)。
        cos, sin (torch.Tensor): compute_rope_params 预计算得到的查找表，
            形状 (context_length, head_dim)，此处会按 offset 切片出所需的 seq_len 段。
        offset (int): 当前 x 的第一个 token 在完整序列中的绝对位置，
            用于在增量推理（KV Cache）时正确对齐旋转角度。

    Returns:
        torch.Tensor: 应用 RoPE 后的张量，形状与输入 x 相同。
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    # 将最后一维（head_dim）拆分为前半、后半两部分，各占 head_dim // 2
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2:]  # Second half

    # Adjust sin and cos shapes
    # 根据 offset 从预计算表中切出当前这段序列对应的 cos/sin，
    # 并增加 batch、head 两个长度为 1 的维度以便广播到 (batch, num_heads, seq_len, head_dim)
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    # 构造"旋转后的一半互换"张量：(-x2, x1)，与原始 x 逐元素运算即可得到旋转结果
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    # 应用完旋转之后可以安全地转换回原始（可能是较低精度的）dtype，
    # 因为旋转变换本身不会引入需要高精度累积的数值问题
    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    """RMSNorm（Root Mean Square Layer Normalization，均方根层归一化）。

    与标准 LayerNorm 的区别：LayerNorm 会先减去均值再除以标准差（做"中心化 + 缩放"），
    而 RMSNorm 省略了减均值这一步，只用输入的均方根（root mean square）做缩放归一化：
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale [+ shift]
    这样计算更简单、速度更快，同时在实践中（LLaMA、Qwen 等模型）被证明效果不逊于 LayerNorm。

    Args:
        emb_dim (int): 归一化的特征维度大小（对最后一维做归一化）。
        eps (float): 防止除零的小常数。
        bias (bool): 是否使用可学习的偏移量（shift）。Qwen3 通常不使用（bias=False）。
        qwen3_compatible (bool): 若为 True，会在计算前将输入强制转换为 float32，
            以保证与官方 Qwen3 实现在数值上一致（RMSNorm 对精度较敏感）。
    """
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        # 可学习的缩放参数，初始化为全 1（即初始时退化为纯粹的"归一化，不缩放"）
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 可选的可学习偏移参数，初始化为全 0；若 bias=False 则不创建（为 None）
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，最后一维大小为 emb_dim，
                典型形状如 (batch, ..., emb_dim)（在本文件中既用于隐藏状态，
                也用于 QK-Norm 场景下形状为 (b, num_heads, seq_len, head_dim) 的张量）。
        Returns:
            torch.Tensor: 归一化后的张量，形状与输入相同，dtype 恢复为输入原始 dtype。
        """
        input_dtype = x.dtype

        if self.qwen3_compatible:
            # 为了与官方 Qwen3 实现数值对齐，归一化计算过程使用 float32 精度
            x = x.to(torch.float32)

        # 计算最后一维上的均方值（mean of squares），保持维度以便广播
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # 用均方根的倒数（rsqrt）对 x 做缩放，实现"均方根归一化"
        norm_x = x * torch.rsqrt(variance + self.eps)
        # 乘以可学习的缩放参数
        norm_x = norm_x * self.scale

        if self.shift is not None:
            # 若启用偏置，则加上可学习的偏移量
            norm_x = norm_x + self.shift

        # 转换回输入原本的 dtype（例如 float16/bfloat16），保持与模型其余部分精度一致
        return norm_x.to(input_dtype)
