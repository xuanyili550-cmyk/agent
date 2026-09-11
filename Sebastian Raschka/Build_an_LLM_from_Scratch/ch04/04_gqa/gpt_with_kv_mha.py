# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

# ==================================================================================
# 中文说明（模块级 docstring）
# ------------------------------------------------------------------------------
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书中
# 第 3 章（多头自注意力 Multi-Head Attention）与第 4 章（GPT 整体架构：LayerNorm、
# GELU、前馈网络、Transformer Block、GPTModel）内容的汇总实现。
#
# 与教材主线版本相比，本文件（位于 ch04/04_gqa 目录，属于 GQA 相关的对比示例）额外
# 加入了 **KV 缓存（Key-Value Cache）** 的支持，用来演示"标准多头注意力 + KV 缓存"
# 在自回归文本生成（autoregressive generation）时如何避免重复计算历史 token 的
# Key/Value，从而加速逐 token 生成。文件名中的 "mha" 即 Multi-Head Attention
# （多头注意力），是后续 GQA（Grouped-Query Attention，分组查询注意力）等变体的
# 对照基线（baseline）版本。
#
# 该文件可以直接作为脚本运行（见文件末尾的 `if __name__ == "__main__":`），会构建
# 一个 GPT-2 124M 规模的模型（默认配置），对提示词 "Hello, I am" 进行续写，并打印
# 生成速度（tokens/sec）等信息，方便直观感受 KV 缓存带来的推理加速效果。
# ==================================================================================

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """标准多头自注意力（Multi-Head Attention）模块，并内置了 KV 缓存支持。

    这是第 3 章讲解的多头自注意力机制的实现：将输入通过三个线性层分别投影为
    Query（查询）、Key（键）、Value（值），再切分成多个"头"（heads）并行计算
    带因果掩码（causal mask）的缩放点积注意力（scaled dot-product attention），
    最后把各头的输出拼接起来并做一次线性变换。

    与教材基础版本的区别：本实现额外维护了 `cache_k` / `cache_v` 两个缓冲区
    （buffer），用于在自回归生成时缓存历史 token 的 Key/Value，这样每次生成
    新 token 时只需要计算新 token 自己的 Key/Value，而不必对整个序列重新计算，
    从而将逐 token 生成的计算复杂度从 O(n^2) 降低到接近 O(n)。

    参数说明：
        d_in (int): 输入特征维度（即 embedding 维度）。
        d_out (int): 输出特征维度，也是 Q/K/V 投影后的总维度
            （等于 num_heads * head_dim）。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool): Q/K/V 线性层是否使用偏置项（bias）。
    """
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头的维度 = 总输出维度 / 头数。
        # 例如 d_out=768, num_heads=12，则 head_dim=64。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # KV cache-related code
        # 中文：KV 缓存相关代码。
        # 用 register_buffer 注册，而不是普通属性，是为了让它们能随模型一起
        # 被 .to(device)/.to(dtype) 移动到正确的设备和精度；persistent=False
        # 表示这两个缓存不会被保存进 state_dict（它们只是运行时的中间状态，
        # 不属于需要持久化的模型参数）。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0  # 记录当前已经处理到序列中的第几个位置，用于因果掩码的位置计算
        ####################################################

    def forward(self, x, use_cache=False):
        """前向传播：计算带因果掩码的多头自注意力，可选启用 KV 缓存。

        参数：
            x (Tensor): 输入张量，形状 (b, num_tokens, d_in)。
                - 训练/无缓存模式下，num_tokens 通常是整段序列长度；
                - 启用 KV 缓存做增量推理时，num_tokens 通常等于 1（每次只喂入新生成的 1 个 token）。
            use_cache (bool): 是否使用/更新 KV 缓存。

        返回：
            context_vec (Tensor): 注意力输出，形状 (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)
        # 中文：三个线性投影分别得到本次前向新计算出的 Key / Value / Query，
        # 形状都是 (b, num_tokens, d_out)。注意这里的 keys_new/values_new
        # 只是"新 token"对应的 K/V，还没有和历史缓存拼接。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆成 (num_heads, head_dim)，
        # 相当于把一个大矩阵隐式地切分成多个头，每个头独立做注意力计算。
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # KV cache-related
        # 中文：KV 缓存的核心逻辑。
        if use_cache:
            if self.cache_k is None:
                # 中文：第一次调用（通常是喂入完整 prompt），缓存为空，直接把本次的 K/V 存入缓存。
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                # 中文：后续调用（增量生成新 token 时），把新 token 的 K/V
                # 沿着序列维度（dim=1）拼接到历史缓存后面，
                # 这样缓存里始终保存"从头到当前"的全部 Key/Value，
                # 而不需要对历史 token 重新计算投影。
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            # 中文：不使用缓存时，keys/values 就是本次输入对应的全部 K/V（标准做法）。
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度提前，方便后续对每个头做批量矩阵乘法（bmm）。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 中文：计算缩放点积注意力的原始得分（attention scores）。
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文：queries 形状 (b, num_heads, num_tokens_Q, head_dim)，
        # keys.transpose(2,3) 形状 (b, num_heads, head_dim, num_tokens_K)，
        # 矩阵乘法后 attn_scores 形状为 (b, num_heads, num_tokens_Q, num_tokens_K)，
        # 表示每个 query 位置对每个 key 位置的原始注意力打分。

        ####################################################
        # causal mask
        # 中文：因果掩码（causal mask）——保证每个位置只能"看到"自己及之前的 token，
        # 不能看到未来的 token，这是自回归语言模型的关键约束。
        # 这里用绝对位置（position id）而不是简单的下三角矩阵来构造掩码，
        # 是因为启用 KV 缓存后，每次 forward 传入的 query 数量可能只有 1 个新 token，
        # 但它仍需要"看到"缓存中所有历史 key 的位置，所以要用真实的绝对位置来比较。
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        device = queries.device
        if use_cache:
            # 中文：使用缓存时，本次 query 对应的绝对位置是从
            # ptr_current_pos 开始，连续 num_tokens_Q 个位置
            # （例如上一次已经处理到位置 5，这次新增 1 个 token，则其绝对位置是 6）。
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q  # 更新指针，为下一次调用做准备
        else:
            # 中文：不使用缓存时，每次都是从位置 0 开始的完整序列。
            q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
        # 中文：mask_bool[i, j] = True 表示 "query 位置 i 的绝对位置 < key 位置 j 的绝对位置"，
        # 也就是 key 在 query 的"未来"，这种位置需要被屏蔽（不能被看到）。
        mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        # Use the mask to fill attention scores
        # 中文：把被掩盖的位置的注意力得分置为 -inf，这样经过 softmax 后其权重趋近于 0，
        # 相当于完全屏蔽了对"未来" token 的关注。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：对得分除以 sqrt(head_dim) 做缩放（避免点积数值过大导致 softmax 梯度消失），
        # 然后在最后一维（key 维度）上做 softmax，得到归一化的注意力权重。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 中文：对注意力权重做 dropout 正则化

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：注意力权重与 Value 加权求和，得到每个头的上下文向量（context vector），
        # 再转置回 (b, num_tokens, num_heads, head_dim) 便于后续合并多头。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多头的输出重新拼接（合并）回 (b, num_tokens, d_out)。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：最后再经过一个线性层（输出投影），融合各头信息。

        return context_vec

    def reset_cache(self):
        """清空 KV 缓存并重置位置指针，通常在每次开始新的一轮生成前调用。"""
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）模块。

    对最后一维（特征/embedding 维度）做归一化：减去均值、除以标准差，
    再用可学习的缩放参数 scale 和偏移参数 shift 做仿射变换。
    这是 Transformer 中稳定训练、缓解梯度问题的关键组件。

    参数：
        emb_dim (int): 特征维度大小（即 embedding 维度）。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：极小值，防止除以 0
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习的缩放参数 gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数 beta

    def forward(self, x):
        """
        参数：
            x (Tensor): 形状 (..., emb_dim)，通常为 (batch, seq_len, emb_dim)。
        返回：
            Tensor: 与输入同形状，已在最后一维上做归一化 + 仿射变换。
        """
        mean = x.mean(dim=-1, keepdim=True)  # 中文：沿最后一维求均值，keepdim 保持维度便于广播
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 中文：沿最后一维求方差（有偏估计，与原始论文一致）
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 中文：标准化：减均值除以标准差
        return self.scale * norm_x + self.shift  # 中文：仿射变换，恢复模型的表达能力


class GELU(nn.Module):
    """GELU（Gaussian Error Linear Unit）激活函数的近似（tanh）实现。

    GPT-2 等模型中常用的激活函数，比 ReLU 更平滑，在 0 附近有更好的梯度性质。
    这里使用的是其 tanh 近似公式，而非精确的高斯误差函数形式。
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
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        # 中文：GELU 的 tanh 近似公式：
        # GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))


class FeedForward(nn.Module):
    """Transformer 中的前馈网络（Feed-Forward Network, FFN / MLP）子层。

    结构为：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)。
    先升维再降维（"倒瓶颈"结构），增加模型的非线性表达能力。

    参数：
        cfg (dict): 配置字典，需包含 "emb_dim" 键（embedding 维度）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 中文：升维到 4 倍，扩大中间表示空间
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 中文：再降维回原始维度
        )

    def forward(self, x):
        """
        参数：
            x (Tensor): 形状 (batch, seq_len, emb_dim)。
        返回：
            Tensor: 形状 (batch, seq_len, emb_dim)，与输入相同。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """一个完整的 Transformer 块（Block），包含：
    多头自注意力子层 + 前馈网络子层，每个子层都配有残差连接（shortcut）
    和前置层归一化（Pre-LayerNorm）。

    结构（Pre-Norm 结构）：
        x -> LN -> Attention -> Dropout -> (+ 残差) -> LN -> FFN -> Dropout -> (+ 残差)

    参数：
        cfg (dict): 配置字典，需包含 "emb_dim", "n_heads", "drop_rate", "qkv_bias" 等键。
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 中文：注意力子层前的层归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 中文：前馈网络子层前的层归一化
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        """
        参数：
            x (Tensor): 形状 (batch, seq_len, emb_dim)。
            use_cache (bool): 是否启用 KV 缓存（透传给内部的注意力模块）。
        返回：
            Tensor: 形状 (batch, seq_len, emb_dim)，与输入相同。
        """
        # Shortcut connection for attention block
        # 中文：保存残差连接的"捷径"输入，供后面与注意力输出相加
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        # 中文：将 use_cache 参数透传给多头注意力模块，以启用/禁用 KV 缓存。
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差连接——把注意力子层的输出加回原始输入，缓解深层网络的梯度消失问题。

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：前馈网络子层同样使用残差连接。

        return x


class GPTModel(nn.Module):
    """完整的 GPT 模型（解码器架构，decoder-only Transformer）。

    结构：Token Embedding + 位置 Embedding -> Dropout ->
          N 层 TransformerBlock 堆叠 -> 最终 LayerNorm -> 输出线性层（预测下一个 token 的 logits）。

    支持通过 use_cache 参数启用 KV 缓存以加速自回归生成。

    参数：
        cfg (dict): 配置字典，需包含以下键：
            - "vocab_size": 词表大小
            - "context_length": 支持的最大上下文长度（决定位置 embedding 表大小）
            - "emb_dim": embedding 维度
            - "n_layers": Transformer 块的层数
            - "drop_rate": dropout 概率
            - 以及传给 TransformerBlock 的其他键（n_heads, qkv_bias 等）
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 中文：词元（token）嵌入表
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 中文：可学习的绝对位置嵌入表
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache-related
        # 中文：这里改用 nn.ModuleList 而不是 nn.Sequential，
        # 是因为需要在 forward 里手动给每一层传入 use_cache 参数，
        # 而 nn.Sequential 只能顺序传递单一输入，无法传递额外的关键字参数。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0  # 中文：记录当前已生成/处理到的绝对位置，用于取正确的位置 embedding
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 中文：输出投影层，把 emb_dim 维的隐藏状态映射为词表大小的 logits，用于预测下一个 token。

    def forward(self, in_idx, use_cache=False):
        """
        参数：
            in_idx (Tensor): 输入 token id 序列，形状 (batch_size, seq_len)，dtype 为 long。
            use_cache (bool): 是否启用 KV 缓存进行增量推理。
        返回：
            logits (Tensor): 形状 (batch_size, seq_len, vocab_size)，
                每个位置对词表中每个 token 的预测得分（未归一化）。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 中文：形状 (batch_size, seq_len, emb_dim)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        # 中文：根据是否使用缓存，决定本次输入 token 对应的"绝对位置"编号。
        if use_cache:
            # 中文：增量生成模式下，本次输入的 seq_len 个 token 的绝对位置
            # 紧接着上一次结束的位置（self.current_pos）继续编号。
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            # 中文：非缓存模式下，每次都是从位置 0 开始的完整序列。
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        # 中文：unsqueeze(0) 在最前面增加 batch 维，形状变为 (1, seq_len, emb_dim)，
        # 后面通过广播（broadcasting）与 tok_embeds 相加。
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：token embedding 与位置 embedding 相加，融合"是什么词"和"在哪个位置"两种信息。
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        # 中文：逐层遍历 TransformerBlock，并把 use_cache 透传下去，
        # 每一层内部的注意力模块会各自维护自己的 KV 缓存。
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)  # 中文：最终的层归一化
        logits = self.out_head(x)  # 中文：投影到词表维度，得到预测 logits
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        """重置整个模型的 KV 缓存（清空每一层注意力模块的缓存）及位置计数器。

        通常在开始生成一个新序列之前调用，避免与上一次生成残留的缓存混淆。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用贪心解码（greedy decoding）自回归生成文本，可选启用 KV 缓存加速。

    参数：
        model (GPTModel): 已构建好的 GPT 模型实例。
        idx (Tensor): 初始 token id 序列（prompt），形状 (batch_size, seq_len)。
        max_new_tokens (int): 需要新生成的 token 数量。
        context_size (int, 可选): 模型支持的最大上下文长度；若为 None，
            则使用 model.pos_emb.num_embeddings（即位置嵌入表的大小）。
        use_cache (bool): 是否启用 KV 缓存来加速生成。

    返回：
        Tensor: 拼接了原始 prompt 和新生成 token 的完整序列，
            形状 (batch_size, seq_len + max_new_tokens)。
    """
    model.eval()  # 中文：切换到推理模式，关闭 dropout 等训练专用行为
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():  # 中文：推理阶段不需要计算梯度，节省显存和算力
        if use_cache:
            # Init cache with full prompt
            # 中文：先清空缓存，然后把完整的 prompt 一次性喂入模型，
            # 这一步会把 prompt 中每个 token 的 K/V 都存入缓存（"预填充" / prefill 阶段）。
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                # 中文：取最后一个位置的 logits，选取概率最高的 token（贪心采样，非随机采样）
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                # 中文：把新生成的 token 拼接到序列末尾
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                # 中文：关键优化点——由于 K/V 已经缓存，这里只需要把"新生成的这一个 token"
                # 喂给模型，模型内部会自动把它的 K/V 与历史缓存拼接后计算注意力，
                # 而不必重新计算之前所有 token 的 K/V，大幅减少重复计算。
                logits = model(next_idx, use_cache=True)
        else:
            # 中文：不使用缓存的朴素做法——每生成一个新 token，
            # 都要把"当前为止的整个序列"重新喂给模型做一次完整前向计算，
            # 计算量随序列长度增长而显著增加（存在大量重复计算）。
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    """脚本入口：构建一个 GPT-2 124M 规模的模型，对示例文本做续写，并统计生成速度。

    主要流程：
        1. 解析命令行参数（模型规模、生成长度等）。
        2. 用 tiktoken 的 GPT-2 分词器对起始文本进行编码。
        3. 构建模型配置并实例化 GPTModel，转换为 bfloat16 精度并放到可用设备（GPU/CPU）上。
        4. 调用 generate_text_simple_cached 进行自回归生成（默认启用 KV 缓存）。
        5. 打印生成结果、耗时、生成速度（tokens/sec）以及（若有 GPU）显存占用峰值。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Run GPT with standard multi-head attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        # 中文：上下文长度设为"待生成 token 数 + 提示词长度"，
        # 保证位置嵌入表足够覆盖整个生成过程中出现的所有绝对位置。
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
    }
    torch.manual_seed(123)  # 中文：固定随机种子，保证模型初始化权重可复现
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)  # 中文：使用 bfloat16 精度以节省显存、加快计算（此处模型未训练，仅演示推理流程）
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 中文：CUDA 是异步执行的，计时前需要同步，确保之前的 GPU 操作已完成
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 中文：计时结束前再次同步，确保生成过程中的所有 GPU 计算都已完成
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
        # 中文：打印生成过程中 GPU 显存分配的峰值，用于对比不同注意力实现（如 MHA vs GQA）的显存开销。


if __name__ == "__main__":
    main()
