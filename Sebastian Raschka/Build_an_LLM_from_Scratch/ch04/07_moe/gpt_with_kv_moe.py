# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

"""
【中文模块说明】
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 3~4 章内容的汇总实现，并在此基础上扩展了两个进阶主题：

1. KV 缓存（KV Cache）：在自回归生成（一次生成一个 token）时，如果每次都把
   完整的历史序列重新过一遍 Attention，会有大量重复计算。KV 缓存的思路是把
   已经算过的 Key/Value 张量缓存下来，新来一个 token 时只需要计算它自己的
   Q/K/V，然后把新的 K/V 拼接（concat）到缓存里，从而把每步生成的计算量从
   O(seq_len) 降到 O(1)（相对于历史长度而言），显著加速推理。
2. MoE（Mixture-of-Experts，混合专家）前馈网络：把标准 Transformer 中单一的
   前馈网络（FeedForward）替换成多个「专家」子网络，每个 token 只激活其中
   Top-k 个专家参与计算（而不是全部专家都参与），这样可以在增大模型总参数量
   的同时，保持单个 token 的实际计算量（FLOPs）不变，是训练超大模型常用的
   稀疏化手段（如 Mixtral、DeepSeek-MoE 等模型的核心思想）。

文件结构：
- Chapter 3 部分：多头自注意力 MultiHeadAttention（含 KV 缓存实现）。
- Chapter 4 部分：LayerNorm、GELU、FeedForward（稠密前馈网络）、
  MoEFeedForward（稀疏专家前馈网络）、TransformerBlock（把注意力和前馈网络
  用残差连接组装起来）、GPTModel（完整的 GPT 结构：词嵌入 + 位置嵌入 +
  多层 TransformerBlock + 输出头）。
- 一个支持 KV 缓存的贪心解码函数 generate_text_simple_cached。
- 一个 main() 函数，用于命令行运行、构造一个小型 GPT 模型并生成一段文本，
  同时打印 MoE 前馈层的耗时/显存统计信息，方便对比稠密 FFN 与 MoE FFN
  的性能差异。
"""

import argparse
import time
import tiktoken
import torch
import torch.nn as nn

# 全局列表，用于统计每次调用 MoE 前馈层（或普通前馈层）所耗费的时间（毫秒）
# 和峰值显存增量（字节）。这两个列表在 generate_text_simple_cached 开始时会被清空，
# 生成结束后用来打印平均耗时/显存，帮助直观比较 MoE 稀疏计算带来的开销变化。
MOE_FF_TIME_MS = []
MOE_FF_MEM_BYTES = []


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """多头自注意力（Multi-Head Self-Attention）模块，支持可选的 KV 缓存。

    这是 Transformer 的核心组件：把输入序列中每个 token 的表示，通过
    Query/Key/Value 三个线性投影，计算 token 与 token 之间的注意力权重，
    再据此对 Value 做加权求和，得到融合了上下文信息的新表示。

    参数说明：
        d_in (int): 输入特征维度（即每个 token 输入向量的维度）。
        d_out (int): 输出特征维度，同时也是所有注意力头拼接后的总维度。
        dropout (float): 注意力权重上的 dropout 概率，用于正则化、防止过拟合。
        num_heads (int): 注意力头的数量，d_out 会被平均切分给每个头。
        qkv_bias (bool): Q/K/V 三个线性层是否使用偏置项（bias），默认关闭
            （这与 GPT-2 等模型的常见实现一致）。

    关键张量形状约定：
        输入 x: (batch, num_tokens, d_in)
        输出 context_vec: (batch, num_tokens, d_out)
    """

    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头分到的维度大小 = 总输出维度 / 头数，
        # 例如 d_out=768, num_heads=12 时 head_dim=64。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # KV cache-related code
        # 中文：KV 缓存相关代码。
        # cache_k / cache_v 用于保存历史 token 的 Key / Value 张量。
        # 注册为非持久化（persistent=False）的 buffer，意味着它们不会被保存进
        # state_dict（因为这是运行期的临时状态，不属于模型「参数」）。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0  # 中文：记录当前已经处理到的“绝对位置”，用于生成因果掩码
        ####################################################

    def forward(self, x, use_cache=False):
        """前向传播：计算多头自注意力输出。

        参数：
            x (Tensor): 形状 (batch, num_tokens, d_in) 的输入张量。
            use_cache (bool): 是否启用 KV 缓存。生成阶段传入 True 时，
                会把本次算出的 K/V 追加到缓存中，并且只针对新 token 计算
                因果掩码需要的“绝对位置”；训练阶段/一次性前向传播通常传 False。

        返回：
            Tensor，形状 (batch, num_tokens, d_out)，融合了上下文信息的表示。
        """
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)
        # 中文：分别对输入做线性投影得到 Q/K/V，此时形状均为 (b, num_tokens, d_out)。
        # 注意变量名里的 "_new" 后缀表示这是“本次输入”新算出的 K/V（尚未和缓存拼接）。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把 d_out 这一维拆分成 (num_heads, head_dim)，
        # 这样每个头可以独立地计算自己的注意力，是多头注意力“并行多个子空间”思想的体现。
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache-related
        # 中文：KV 缓存的核心逻辑——
        # 如果启用缓存：
        #   - 若缓存为空（第一次调用，通常对应处理完整 prompt），直接把新算出的 K/V 存为缓存；
        #   - 否则（后续每步只输入 1 个新 token），把新 K/V 沿着序列长度维（dim=1）拼接到旧缓存后面。
        # 这样可以避免每次都重新计算历史 token 的 K/V，是 KV 缓存加速自回归生成的关键。
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            # 中文：不使用缓存时，直接用本次算出的 K/V（即普通的、非增量式的注意力计算）。
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维提前，方便后面对每个头分别做批量矩阵乘法（bmm）。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 中文：计算缩放点积注意力的“打分”部分：Q @ K^T。
        # queries: (b, num_heads, num_tokens_Q, head_dim)
        # keys.transpose(2,3): (b, num_heads, head_dim, num_tokens_K)
        # 结果 attn_scores: (b, num_heads, num_tokens_Q, num_tokens_K)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        ####################################################
        # causal mask
        # 中文：因果掩码（causal mask）——保证每个 query 位置只能看到“它自己以及之前”的
        # key 位置，不能看到未来的 token，这是自回归语言模型的基本约束。
        # 这里用绝对位置（而不是简单的上三角掩码）来实现，是为了兼容 KV 缓存场景：
        # 当使用缓存时，本次输入的 query 只有 1 个新 token，但它的“绝对位置”其实是
        # 序列中很靠后的位置，需要能看到缓存里所有更早的 key。
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            # 中文：使用缓存时，query 的绝对位置从 ptr_current_pos 开始，
            # 长度为本次新输入的 token 数（增量生成时通常是 1）。
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q  # 中文：更新指针，为下一次调用做准备
        else:
            # 中文：不使用缓存时（比如一次性喂入整段 prompt/训练时），
            # query 的位置就是 0..num_tokens_Q-1，并重置指针。
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        # 中文：mask_bool[i, j] = True 表示 query 位置 i 不能看到 key 位置 j
        # （即 j 是“未来”位置），后面会把这些位置的注意力分数置为 -inf。
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        # Use the mask to fill attention scores
        # 中文：把被掩盖（未来）的位置的注意力分数设为负无穷，
        # 这样经过 softmax 后这些位置的权重会趋近于 0，达到“看不到未来”的效果。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：对分数做缩放（除以 sqrt(head_dim)，即 keys.shape[-1] 的平方根）
        # 再做 softmax 得到注意力权重；缩放是为了防止点积数值过大导致 softmax 梯度消失。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：用注意力权重对 Value 做加权求和，得到每个 query 位置融合上下文后的表示；
        # 再把 num_heads 维转回到 num_tokens 之前，方便下一步合并所有头。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多个头的输出重新拼接（concat）回单一维度 d_out，
        # 然后经过输出投影 out_proj 做一次线性变换（融合各头信息）。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    def reset_cache(self):
        """清空 KV 缓存并重置位置指针。

        中文说明：每次开始一段新的、独立的生成任务前，都应该调用此方法，
        否则新的生成会错误地拼接上一次遗留的历史 K/V 缓存。
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）。

    对每个 token 的特征向量（最后一维）做归一化，使其均值为 0、方差为 1，
    再通过可学习的缩放（scale）和平移（shift）参数恢复模型需要的表达能力。
    这是稳定深层网络训练、缓解梯度爆炸/消失的常用手段。

    参数：
        emb_dim (int): 特征维度（例如词嵌入维度），归一化在这一维上进行。
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除以 0 的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 中文：可学习的缩放参数 gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 中文：可学习的平移参数 beta

    def forward(self, x):
        """输入/输出形状均为 (..., emb_dim)，通常是 (batch, seq_len, emb_dim)。"""
        mean = x.mean(dim=-1, keepdim=True)
        # 中文：使用有偏方差估计（unbiased=False，即除以 N 而不是 N-1），
        # 与 GPT-2 官方实现保持一致。
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数（这里使用的是其 tanh 近似形式，与 GPT-2 论文/实现一致）。

    GELU 相比 ReLU 更平滑，在 Transformer 类模型中被广泛使用。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 中文：GELU 的 tanh 近似公式：
        # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        # 输入输出形状相同。
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """标准（稠密）前馈网络（Feed-Forward Network，FFN）。

    结构为：Linear(emb_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> emb_dim)。
    这是 Transformer Block 中除注意力外的另一个核心子层，
    对每个 token 的表示独立地做非线性变换（逐位置的 MLP）。

    与下面的 MoEFeedForward 相对，这里所有 token 都经过同一套参数，
    是「稠密」计算；而 MoE 版本则是每个 token 只选择部分专家参与计算，
    属于「稀疏」计算。

    参数：
        cfg (dict): 需要包含 "emb_dim"（输入/输出维度）和 "hidden_dim"（中间隐藏层维度）。
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["hidden_dim"]),
            GELU(),
            nn.Linear(cfg["hidden_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """输入/输出形状: (batch, seq_len, emb_dim)。"""
        return self.layers(x)


class MoEFeedForward(nn.Module):
    """混合专家（Mixture-of-Experts, MoE）前馈网络。

    核心思想：不是只用一个前馈网络处理所有 token，而是准备多个「专家」
    （每个专家是一个独立的小型前馈网络），并用一个「门控（gate）」网络
    为每个 token 打分，选出得分最高的 Top-k 个专家来处理该 token，
    最后按照门控给出的权重（softmax 后的概率）对各专家输出做加权求和。

    这样做的好处：模型的总参数量可以随专家数量线性增长（表达能力更强），
    但由于每个 token 只激活少数几个专家，实际计算量（FLOPs）并不随专家数
    等比例增长，从而实现「大参数量、可控计算量」的稀疏模型，
    是 Mixtral、DeepSeek-MoE、Switch Transformer 等大模型的关键技术。

    这里每个专家采用类似 SwiGLU 的门控前馈结构：
        hidden = SiLU(fc1(x)) * fc2(x)
        out = fc3(hidden)
    即用 fc1 的输出经过 SiLU 激活后，与 fc2 的输出逐元素相乘做门控，
    再经过 fc3 投影回原始维度。

    参数：
        cfg (dict): 需要包含：
            - "emb_dim": token 表示的维度
            - "hidden_dim": 每个专家内部隐藏层的维度
            - "num_experts": 专家总数
            - "num_experts_per_tok": 每个 token 实际激活（Top-k）的专家数
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]  # 中文：每个 token 激活的专家数 k
        self.num_experts = cfg["num_experts"]                  # 中文：专家总数
        self.emb_dim = cfg["emb_dim"]

        # 中文：门控网络（gate/router），把每个 token 的表示映射为对每个专家的“打分”，
        # 形状 (emb_dim -> num_experts)，不带偏置。
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False)
        # 中文：为每个专家分别创建独立的 fc1（升维 + 门控分支之一）
        self.fc1 = nn.ModuleList(
            [
                nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )
        # 中文：fc2 是门控分支的另一半（与 fc1 输出逐元素相乘，构成类似 SwiGLU 的结构）
        # 注意：这里 fc2 的输出维度写的是 cfg["hidden_dim"]（与 fc1 相同），
        # 而不是常见 SwiGLU 里 fc2 输入输出维度对称的写法；保持原代码不变。
        self.fc2 = nn.ModuleList(
            [
                nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )
        # 中文：fc3 把隐藏维度投影回 emb_dim，作为该专家的最终输出
        self.fc3 = nn.ModuleList(
            [
                nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x):
        # x: (batch, seq_len, emb_dim)
        # 中文：门控网络给出每个 token 对每个专家的打分（logits）
        scores = self.gate(x)  # (b, seq_len, num_experts)
        # 中文：选出打分最高的 top-k 个专家及其索引；
        # topk_scores/topk_indices 形状均为 (b, seq_len, num_experts_per_tok)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        # 中文：只在被选中的 top-k 专家之间做 softmax 归一化，得到每个专家的加权系数，
        # 保证同一个 token 分配给多个专家的权重之和为 1。
        topk_probs = torch.softmax(topk_scores, dim=-1)

        batch, seq_len, _ = x.shape
        # 中文：把 batch 和 seq_len 两维展平成一维“token 维”，方便后续按 token 做
        # 索引选择（index_select）和分组处理，形状变为 (batch*seq_len, emb_dim)。
        x_flat = x.reshape(batch * seq_len, -1)
        # 中文：用于累加各专家输出的容器，初始为全 0，形状 (batch*seq_len, emb_dim)。
        out_flat = torch.zeros(batch * seq_len, self.emb_dim, device=x.device, dtype=x.dtype)

        topk_indices_flat = topk_indices.reshape(-1, self.num_experts_per_tok)
        topk_probs_flat = topk_probs.reshape(-1, self.num_experts_per_tok)

        # 中文：找出本次前向传播中，实际被至少一个 token 选中的专家集合（去重），
        # 这样可以避免遍历所有专家，只对“用得上”的专家做计算，提升效率。
        unique_experts = torch.unique(topk_indices_flat)

        # 中文：按专家分组处理——对每个被激活的专家，找出选择了它的所有 token，
        # 批量计算这些 token 在该专家上的输出，再按门控权重加权累加回结果张量。
        # 这种“专家为中心”的循环方式，是稀疏 MoE 常见的高效实现思路
        # （避免对每个专家都在全部 token 上做计算）。
        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())

            # 中文：mask[i, j] = True 表示 token i 的第 j 个 top-k 选择正好是当前专家
            mask = topk_indices_flat == expert_id
            if not mask.any():
                continue

            # 中文：只要 token 的 top-k 选择中包含当前专家（不管排第几位），就算被选中
            token_mask = mask.any(dim=-1)
            # 中文：取出被选中 token 的下标（在展平后的 token 维度上）
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            # 中文：取出这些被选中 token 的输入表示，形状 (num_selected, emb_dim)
            expert_input = x_flat.index_select(0, selected_idx)
            # 中文：SwiGLU 风格的门控前馈计算：
            # SiLU(fc1(x)) 作为门控信号，逐元素乘以 fc2(x)，得到隐藏表示
            hidden = torch.nn.functional.silu(self.fc1[expert_id](expert_input)) * self.fc2[
                expert_id
            ](expert_input)
            # 中文：再经过 fc3 投影回 emb_dim，得到该专家对这批 token 的输出
            expert_out = self.fc3[expert_id](hidden)

            # 中文：找到每个被选中 token 在其 top-k 列表中，当前专家所处的“槽位”下标，
            # 用于从 topk_probs_flat 里取出对应的门控权重。
            mask_selected = mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(
                topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices
            ).squeeze(-1)

            # 中文：把该专家的输出按门控权重缩放后，累加到对应 token 在 out_flat 中的位置。
            # 由于一个 token 可能被多个专家选中，这里用 index_add_ 做原地累加，
            # 最终每个 token 的输出 = 它所有被选中专家输出的加权和。
            out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))

        # 中文：把展平的 token 维度还原回 (batch, seq_len, emb_dim)
        return out_flat.reshape(batch, seq_len, self.emb_dim)


class TransformerBlock(nn.Module):
    """一个完整的 Transformer 块：多头自注意力子层 + 前馈网络子层，
    每个子层都配有前置 LayerNorm、残差连接（shortcut）和 dropout。

    结构（Pre-LN 风格）：
        x = x + Dropout(Attention(LayerNorm(x)))
        x = x + Dropout(FFN_or_MoE(LayerNorm(x)))

    参数：
        cfg (dict): 模型配置字典，需包含 emb_dim, n_heads, drop_rate, qkv_bias,
            以及决定前馈网络类型的 num_experts（>0 时使用 MoEFeedForward，
            否则使用普通 FeedForward）等字段。
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
        # 中文：根据配置里的 num_experts 决定使用稀疏 MoE 前馈网络还是稠密前馈网络，
        # 这使得同一份 TransformerBlock 代码可以灵活切换两种模式，便于对比实验。
        self.ff = MoEFeedForward(cfg) if cfg["num_experts"] > 0 else FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        """输入/输出形状: (batch, num_tokens, emb_dim)。

        use_cache: 透传给内部的注意力模块，控制是否启用 KV 缓存。
        """
        # Shortcut connection for attention block
        # 中文：注意力子层的残差连接——先保存输入作为 shortcut，
        # 归一化后过注意力，dropout 后再加回原始输入。
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        # 中文：把 use_cache 参数传给注意力层，让其决定是否读写 KV 缓存。
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        # 中文：前馈网络子层的残差连接，结构与上面注意力子层类似。
        shortcut = x
        x = self.norm2(x)
        # 中文：以下这段是为了统计 MoE（或普通）前馈层的耗时和显存开销而加入的
        # 性能分析（profiling）代码，与模型的前向计算逻辑本身无关。
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize()          # 中文：等待 GPU 上之前的操作全部完成，保证计时准确
            torch.cuda.reset_peak_memory_stats()  # 中文：重置峰值显存统计，方便单独测量本次前馈的显存增量
            base_mem = torch.cuda.memory_allocated()  # 中文：记录调用前的已分配显存基线
        start = time.perf_counter()
        x = self.ff(x)
        if use_cuda:
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated()
            MOE_FF_MEM_BYTES.append(peak_mem - base_mem)  # 中文：记录本次前馈调用的显存增量
        MOE_FF_TIME_MS.append((time.perf_counter() - start) * 1000.0)  # 中文：记录本次前馈调用耗时（毫秒）
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的类 GPT-2 自回归语言模型，支持 KV 缓存加速生成。

    结构：词嵌入 + 可学习位置嵌入 -> N 层 TransformerBlock -> 最终 LayerNorm
    -> 线性输出头（映射到词表大小的 logits）。

    参数：
        cfg (dict): 模型配置，至少包含：
            - vocab_size: 词表大小
            - context_length: 支持的最大上下文长度（位置嵌入表的行数）
            - emb_dim: 嵌入/隐藏维度
            - drop_rate: dropout 概率
            - n_layers: TransformerBlock 层数
            以及 TransformerBlock/MultiHeadAttention/MoEFeedForward 所需的其他字段。
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])       # 中文：词嵌入表
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])   # 中文：可学习的绝对位置嵌入表
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache-related
        # 中文：这里改用 nn.ModuleList 而不是 nn.Sequential，
        # 是因为 KV 缓存场景下每个 TransformerBlock.forward 需要额外传入
        # use_cache 参数，而 nn.Sequential 只支持单输入的链式调用，
        # 所以需要手动写 for 循环来传递这个额外参数（见下面 forward 方法）。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0  # 中文：记录整个模型当前已经处理到的绝对 token 位置（用于位置嵌入）
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        """前向传播，计算下一个 token 的 logits。

        参数：
            in_idx (LongTensor): 形状 (batch_size, seq_len)，输入的 token id 序列。
                当 use_cache=True 时，通常在“预填充（prefill）”阶段传入整个
                prompt，之后每一步只传入 1 个新生成的 token。
            use_cache (bool): 是否启用 KV 缓存进行增量式生成。

        返回：
            logits (Tensor): 形状 (batch_size, seq_len, vocab_size)，
                每个位置对词表中每个 token 的预测得分（未归一化）。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 中文：形状 (batch_size, seq_len, emb_dim)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        # 中文：位置嵌入也需要考虑 KV 缓存——
        # 使用缓存时，本次输入只是序列的一小段（通常是新生成的 1 个 token），
        # 它对应的“绝对位置”应该从 self.current_pos 开始，而不是从 0 开始；
        # 不使用缓存时（比如训练或一次性推理整段序列），位置就是 0..seq_len-1。
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len  # 中文：更新全局位置指针，为下一次调用做准备
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)  # 中文：unsqueeze(0) 增加 batch 维以便广播相加
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：词嵌入与位置嵌入相加，得到每个 token 融合了内容信息和位置信息的初始表示。
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        # 中文：依次经过每一层 TransformerBlock，并把 use_cache 透传下去，
        # 使每一层内部的注意力模块都能正确地读写各自的 KV 缓存。
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)  # 中文：形状 (batch_size, seq_len, vocab_size)
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        """重置所有层的 KV 缓存以及全局位置指针。

        中文说明：在开始一次新的、独立的生成任务之前必须调用，
        否则会残留上一次生成过程中的历史 K/V，导致注意力计算错误
        （例如把不属于当前 prompt 的历史信息也纳入计算）。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用贪心解码（greedy decoding）自回归生成文本，支持可选的 KV 缓存加速。

    参数：
        model (GPTModel): 已实例化的 GPT 模型。
        idx (LongTensor): 形状 (batch_size, base_len)，作为生成起点的输入 token id 序列
            （即 prompt）。
        max_new_tokens (int): 需要新生成的 token 数量。
        context_size (int, optional): 模型支持的最大上下文长度，
            默认使用 model.pos_emb.num_embeddings（即位置嵌入表的行数）。
        use_cache (bool): 是否启用 KV 缓存。启用时只需在“预填充”阶段
            计算一次完整 prompt 的 K/V，之后每步只需计算新 token 的 K/V
            并复用缓存，大幅减少重复计算；关闭时每步都要把当前上下文窗口
            完整地重新计算一遍注意力。

    返回：
        LongTensor，形状 (batch_size, base_len + max_new_tokens)，
        包含原始 prompt 以及新生成的全部 token。

    附带效果：函数结束前会打印 MoE（或普通）前馈层的平均耗时、
        平均/峰值显存增量，便于比较不同配置（是否用 MoE、是否用 KV 缓存）
        下的推理性能。
    """
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings
    batch_size, base_len = idx.shape
    total_len = base_len + max_new_tokens
    # 中文：预先分配好足够长度的输出缓冲区，避免生成过程中反复拼接张量（更高效）。
    generated = torch.empty(
        batch_size, total_len, dtype=idx.dtype, device=idx.device
    )
    generated[:, :base_len] = idx  # 中文：把原始 prompt 写入缓冲区前部
    cur_len = base_len
    use_cuda = torch.cuda.is_available()
    MOE_FF_TIME_MS.clear()   # 中文：清空上一次调用遗留的统计数据
    MOE_FF_MEM_BYTES.clear()

    with torch.no_grad():  # 中文：生成阶段不需要反向传播，关闭梯度计算以节省显存和加速
        if use_cache:
            # Init cache with full prompt
            # 中文：先重置缓存（清空可能残留的历史状态），
            # 然后用完整 prompt 做一次“预填充（prefill）”前向传播，
            # 这一步会把 prompt 中所有 token 的 K/V 一次性写入缓存。
            model.reset_kv_cache()
            prompt_start = max(0, cur_len - ctx_len)
            logits = model(generated[:, prompt_start:cur_len], use_cache=True)

            if use_cuda:
                torch.cuda.synchronize()

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                # 中文：贪心采样——直接取概率（logits）最大的 token 作为下一个 token，
                # 只取最后一个位置（-1）的预测，因为那对应的是“下一个待生成 token”的分布。
                next_idx = logits[:, -1].argmax(dim=-1)
                # b) append it to the running sequence (in-place)
                generated[:, cur_len] = next_idx
                cur_len += 1
                # c) feed model only the new token
                # 中文：得益于 KV 缓存，这里只需要把刚生成的这 1 个新 token 喂给模型，
                # 而不需要把整个历史序列都重新输入，这正是 KV 缓存加速的关键所在。
                logits = model(generated[:, cur_len - 1 : cur_len], use_cache=True)

                if use_cuda:
                    torch.cuda.synchronize()
        else:
            # 中文：不使用缓存的朴素实现——每一步都把当前上下文窗口（受 ctx_len 限制）
            # 完整地重新输入模型，重复计算所有历史 token 的注意力，速度较慢，
            # 但作为对照组便于验证 KV 缓存版本结果的正确性、比较性能差异。
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

    # 中文：打印本次生成过程中，前馈层（MoE 或普通 FFN）的平均耗时统计
    if MOE_FF_TIME_MS:
        avg_ffn_time = sum(MOE_FF_TIME_MS) / len(MOE_FF_TIME_MS)
        print(f"Avg MoE FF time/call: {avg_ffn_time:.3f} ms")
    # 中文：打印本次生成过程中，前馈层的平均/峰值显存增量统计（仅在 GPU 上有意义）
    if MOE_FF_MEM_BYTES:
        avg_ffn_mem = sum(MOE_FF_MEM_BYTES) / len(MOE_FF_MEM_BYTES)
        max_ffn_mem = max(MOE_FF_MEM_BYTES)

        def to_mb(bytes_val):
            return bytes_val / (1024 ** 2)
        print(f"Avg MoE FF mem delta/call: {to_mb(avg_ffn_mem):.2f} MB (max {to_mb(max_ffn_mem):.2f} MB)")

    return generated[:, :cur_len]


def main():
    """命令行入口：解析参数、构建一个小型 GPT（可选 MoE）模型，
    并用它对一段起始文本做自回归生成，最后打印生成结果与性能统计信息。

    中文说明：可以通过命令行参数灵活控制模型规模（emb_dim/n_heads/n_layers 等）、
    是否启用 KV 缓存（--no_kv_cache）、是否启用 MoE 前馈网络以及专家数量/
    Top-k 数（--num_experts / --num_experts_per_tok），
    从而方便地做 A/B 对比实验（稠密 vs 稀疏、有无缓存）。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=768*4, help="Intermediate FFN or MoE size.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV caching during generation.",
    )

    parser.add_argument(
        "--num_experts",
        type=int,
        default=0,
        help="Number of experts. If 0, use dense FFN. If >0, use MoE.",
    )
    parser.add_argument(
        "--num_experts_per_tok",
        type=int,
        default=2,
        help="Top-k experts per token when using MoE (ignored if num_experts=0).",
    )

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")  # 中文：使用 GPT-2 的 BPE 分词器
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,            # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        # 中文：context_length 设置为“需要生成的 token 数 + prompt 长度”，
        # 保证位置嵌入表足够覆盖本次生成会用到的所有绝对位置。
        "emb_dim": args.emb_dim,        # Embedding dimension
        "hidden_dim": args.hidden_dim,  # Intermediate size
        "n_heads": args.n_heads,        # Number of attention heads
        "n_layers": args.n_layers,      # Number of layers
        "drop_rate": 0.0,               # Dropout rate
        "qkv_bias": False,              # Query-Key-Value bias
        "num_experts": args.num_experts,
        # 中文：只有当 num_experts > 0（即启用 MoE）时，num_experts_per_tok 才有意义，
        # 否则统一设为 0（此时 TransformerBlock 会退回使用普通 FeedForward）。
        "num_experts_per_tok": args.num_experts_per_tok if args.num_experts > 0 else 0,
    }
    torch.manual_seed(123)  # 中文：固定随机种子，保证模型初始化权重可复现
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)  # 中文：使用 bfloat16 精度以节省显存、加速计算
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)  # 中文：增加 batch 维，形状 (1, prompt_len)
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
