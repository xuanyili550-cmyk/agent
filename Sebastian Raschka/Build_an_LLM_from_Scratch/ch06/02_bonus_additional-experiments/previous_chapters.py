# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-5.
# This file can be run as a standalone script.

"""
中文模块说明（模块级 docstring）：
本文件是第 6 章 "02_bonus_additional-experiments" 小节复用的前几章代码集合，
汇总了第 2~5 章中已经实现过的组件，供本 bonus 小节的实验脚本直接 import 使用：
    - 第 2 章：GPTDatasetV1（滑动窗口构造输入/目标序列）、create_dataloader_v1（构造 DataLoader）
    - 第 3 章：MultiHeadAttention（多头自注意力，支持通过 disable_causal_mask 关闭因果掩码，
      便于本 bonus 小节做“双向注意力 vs 因果注意力”的对比实验）
    - 第 4 章：LayerNorm、GELU、FeedForward、TransformerBlock、GPTModel、generate_text_simple
      （完整的 GPT 结构与最基础的贪心解码函数）
    - 第 5 章：assign、load_weights_into_gpt（加载 OpenAI 官方 GPT-2 预训练权重）、
      generate（支持温度采样 + top-k 采样 + 提前停止的文本生成函数）

本文件可以作为独立脚本运行，也可以被同目录下的其它实验脚本 import。

注释规范说明：
    - 本次仅为原文件添加中文注释/docstring，未改变任何原有逻辑（除非明确标注为“确定性 bug 修复”）。
    - 对于有潜在风险但并非本文件独有、且只在特定条件下才会触发的问题（例如 batch_size > 1 时的
      边界情况），一律只做标注、不做修改，避免偏离原书/原仓库的参考实现。
"""

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    """
    中文说明：
        第 2 章实现的数据集类，用于把一段长文本切分成若干组
        (输入序列, 目标序列) 的训练样本对。

        采用“滑动窗口”的方式在 token 序列上滑动截取长度为 max_length 的片段：
        目标序列（target）就是输入序列（input）整体右移一位后的结果，
        即经典的“预测下一个 token”自回归训练目标。

    参数：
        txt (str): 原始文本（尚未分词）。
        tokenizer: 具备 .encode() 方法的分词器实例（本文件中使用 tiktoken 的 gpt2 编码）。
        max_length (int): 每个训练样本的 token 序列长度（即模型的上下文窗口长度）。
        stride (int): 滑动窗口每次移动的步长；stride < max_length 时窗口之间会有重叠。

    属性（张量形状）：
        self.input_ids: List[Tensor]，每个元素形状为 (max_length,)。
        self.target_ids: List[Tensor]，每个元素形状为 (max_length,)，
            是对应 input_ids 整体右移一位的结果。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 中文：把整段文本一次性编码为 token id 序列；<|endoftext|> 是 GPT-2 的特殊分隔符，允许出现在文本中
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 中文：用滑动窗口把整本书切分成长度为 max_length、可能相互重叠（取决于 stride）的多段序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]            # 输入片段：token_ids[i : i+max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标片段：相对输入整体右移一位（预测下一个 token）
            self.input_ids.append(torch.tensor(input_chunk))     # 形状 (max_length,)
            self.target_ids.append(torch.tensor(target_chunk))   # 形状 (max_length,)

    def __len__(self):
        """中文：返回数据集中样本（输入/目标序列对）的总数量。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        中文：
            按索引取出一条训练样本。
        返回：
            (input_ids[idx], target_ids[idx])，两者形状均为 (max_length,)。
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    中文说明：
        基于 GPTDatasetV1 构造一个 PyTorch DataLoader，用于按 batch 迭代训练数据。

    参数：
        txt (str): 原始文本。
        batch_size (int): 每个 batch 包含的样本数。
        max_length (int): 每条样本的 token 序列长度（上下文窗口长度）。
        stride (int): 滑动窗口步长。
        shuffle (bool): 是否打乱样本顺序。
        drop_last (bool): 是否丢弃最后一个样本数不足 batch_size 的 batch（训练时常设为 True，避免 batch 尺寸不一致导致的问题）。
        num_workers (int): DataLoader 使用的子进程数，用于并行加载数据。

    返回：
        torch.utils.data.DataLoader，每次迭代产出一个 (input_batch, target_batch) 二元组，
        两者形状均为 (batch_size, max_length)。
    """
    # Initialize the tokenizer
    # 中文：初始化 GPT-2 使用的 BPE 分词器（tiktoken 实现）
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    # 中文：构造滑动窗口数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # 中文：包装成 DataLoader，负责批处理、打乱、丢弃末尾不满 batch 的数据等
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """
    中文说明：
        第 3 章实现的多头自注意力模块（Multi-Head Self-Attention）。
        相比 ch05 主线代码，这里额外增加了 disable_causal_mask 参数：
        当设为 True 时不再应用因果掩码（causal mask），
        即允许每个位置看到"未来"的 token（双向注意力）。
        这是本 bonus 小节做实验用的开关（例如对比分类任务中使用因果 vs 双向注意力的效果差异）。

    参数：
        d_in (int): 输入特征维度。
        d_out (int): 输出特征维度（同时也是 Q/K/V 投影后的总维度），必须能被 num_heads 整除。
        context_length (int): 支持的最大序列长度，用于预先构造好因果掩码矩阵。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头数。
        qkv_bias (bool): Q/K/V 线性层是否使用 bias。
        disable_causal_mask (bool): 是否关闭因果掩码（True 则退化为双向/全局注意力）。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, disable_causal_mask=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头的维度 = 总输出维度 / 头数

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 中文：Q 投影，(d_in) -> (d_out)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)    # 中文：K 投影，(d_in) -> (d_out)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 中文：V 投影，(d_in) -> (d_out)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # 中文：多头拼接后的输出投影层，(d_out) -> (d_out)
        self.dropout = nn.Dropout(dropout)

        if not disable_causal_mask:
            # 中文：仅在需要因果掩码时才注册；上三角（不含对角线）为 1，表示"未来位置"，后续会被置为 -inf
            self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.disable_causal_mask = disable_causal_mask

    def forward(self, x):
        """
        中文说明：
            前向传播，计算（可选是否因果掩码的）多头自注意力输出。

        参数：
            x (Tensor): 形状 (b, num_tokens, d_in)，b=batch size，num_tokens=序列长度。

        返回：
            Tensor，形状 (b, num_tokens, d_out)，与输入序列长度一致、通道数变为 d_out。
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆分成 (num_heads, head_dim)，为多头并行计算做准备
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度换到前面，方便对每个头独立做批量矩阵乘法
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文：queries (b, num_heads, num_tokens, head_dim) @ keys^T (b, num_heads, head_dim, num_tokens)
        #      -> attn_scores 形状 (b, num_heads, num_tokens, num_tokens)，即每个 query 位置对每个 key 位置的原始打分

        if not self.disable_causal_mask:
            # Original mask truncated to the number of tokens and converted to boolean
            # 中文：把预先注册的 (context_length, context_length) 掩码裁剪到当前实际序列长度 (num_tokens, num_tokens)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            # Use the mask to fill attention scores
            # 中文：把"未来位置"（mask 为 True 处）的注意力得分置为 -inf，softmax 后权重趋于 0，从而实现因果（只看过去）注意力
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 中文：按 head_dim 的平方根做缩放（scaled dot-product），再在最后一维（key 位置维）做 softmax 得到注意力权重
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：attn_weights (b, num_heads, num_tokens, num_tokens) @ values (b, num_heads, num_tokens, head_dim)
        #      -> (b, num_heads, num_tokens, head_dim)，再转置回 (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多头拼回单一维度：(b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：对拼接后的多头输出再做一次线性变换（输出投影）

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    中文说明：
        第 4 章手写实现的 Layer Normalization（层归一化）。
        在最后一个维度（特征维度 emb_dim）上做归一化，并配有可学习的缩放（scale）和平移（shift）参数，
        用于替代 nn.LayerNorm，方便讲解其内部计算细节。

    参数：
        emb_dim (int): 需要归一化的特征维度大小。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 中文：可学习缩放参数 gamma，形状 (emb_dim,)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 中文：可学习平移参数 beta，形状 (emb_dim,)

    def forward(self, x):
        """
        中文说明：
            对输入张量最后一维做归一化：减均值除以标准差，再做仿射变换 scale * norm_x + shift。

        参数：
            x (Tensor): 形状 (..., emb_dim)，通常是 (batch_size, num_tokens, emb_dim)。

        返回：
            Tensor，形状与输入相同 (..., emb_dim)。
        """
        mean = x.mean(dim=-1, keepdim=True)  # 中文：沿最后一维求均值，形状 (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 中文：沿最后一维求方差（有偏估计，与 GPT-2 官方实现一致），形状 (..., 1)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 中文：标准化，形状与 x 相同
        return self.scale * norm_x + self.shift  # 中文：逐元素仿射变换（广播到 emb_dim 维）


class GELU(nn.Module):
    """
    中文说明：
        GELU（Gaussian Error Linear Unit）激活函数的手写实现，
        使用的是 GPT-2 论文中采用的 tanh 近似公式，而非精确的误差函数形式。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        中文说明：
            GELU 的 tanh 近似实现：
            0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

        参数：
            x (Tensor): 任意形状。

        返回：
            Tensor，形状与输入相同。
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    中文说明：
        Transformer Block 中的前馈网络（Position-wise Feed-Forward），
        结构为：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)，
        先升维再降维，是 GPT 系列模型的标准做法。

    参数：
        cfg (dict): 配置字典，需包含键 "emb_dim"。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 中文：升维，(emb_dim) -> (4*emb_dim)
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 中文：降维回原始维度，(4*emb_dim) -> (emb_dim)
        )

    def forward(self, x):
        """
        中文说明：
            前向传播，逐 token 独立地做非线性变换（不同 token 之间不交互信息）。

        参数：
            x (Tensor): 形状 (batch_size, num_tokens, emb_dim)。

        返回：
            Tensor，形状与输入相同 (batch_size, num_tokens, emb_dim)。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    中文说明：
        标准的 GPT Transformer Block（Pre-LayerNorm 结构）：
            x -> LN -> 多头自注意力 -> Dropout -> 残差相加
              -> LN -> 前馈网络 -> Dropout -> 残差相加

    参数：
        cfg (dict): 模型配置字典，需包含 emb_dim / context_length / n_heads / drop_rate / qkv_bias 等键。
        disable_causal_mask (bool): 透传给内部 MultiHeadAttention，是否关闭因果掩码。
    """
    def __init__(self, cfg, disable_causal_mask=False):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            disable_causal_mask=disable_causal_mask
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        中文说明：
            前向传播，依次经过“注意力子层 + 残差”和“前馈子层 + 残差”。

        参数：
            x (Tensor): 形状 (batch_size, num_tokens, emb_dim)。

        返回：
            Tensor，形状与输入相同 (batch_size, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        # 中文：保存注意力子层的残差分支（shortcut）
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差相加，缓解深层网络梯度消失问题

        # Shortcut connection for feed-forward block
        # 中文：保存前馈子层的残差分支
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差相加

        return x


class GPTModel(nn.Module):
    """
    中文说明：
        完整的 GPT 模型：token 嵌入 + 位置嵌入 -> Dropout -> 多层 TransformerBlock
        -> 最终 LayerNorm -> 线性输出头（映射到词表维度的 logits）。

    参数：
        cfg (dict): 模型配置字典，需包含 vocab_size / emb_dim / context_length / drop_rate / n_layers 等键。
        disable_causal_mask (bool): 透传给每一层 TransformerBlock，是否关闭因果掩码
            （本 bonus 小节用于对比因果 vs 双向注意力对分类任务效果的影响）。
    """
    def __init__(self, cfg, disable_causal_mask=False):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])       # 中文：token 嵌入表，(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])   # 中文：可学习的绝对位置嵌入表，(context_length, emb_dim)
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg, disable_causal_mask) for _ in range(cfg["n_layers"])])
        # 中文：堆叠 n_layers 个 TransformerBlock

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 中文：输出投影层，把隐藏状态映射为词表大小的 logits（不带 bias，与 GPT-2 官方实现一致）

    def forward(self, in_idx):
        """
        中文说明：
            前向传播，输入 token id 序列，输出每个位置在整个词表上的 logits 分布。

        参数：
            in_idx (Tensor): 形状 (batch_size, seq_len)，dtype 为整型（token id）。

        返回：
            Tensor，形状 (batch_size, seq_len, vocab_size)，未经过 softmax 的原始 logits。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 中文：(batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 中文：位置索引 (seq_len,) -> 位置嵌入 (seq_len, emb_dim)，会广播加到每个样本上
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：token 嵌入 + 位置嵌入，广播相加，形状 (batch_size, seq_len, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)  # 中文：依次通过所有 TransformerBlock，形状保持 (batch_size, seq_len, emb_dim)
        x = self.final_norm(x)
        logits = self.out_head(x)  # 中文：(batch_size, seq_len, emb_dim) -> (batch_size, seq_len, vocab_size)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    中文说明：
        最基础的贪心（greedy）自回归文本生成函数：每一步都选取概率最高（logits 最大）的 token，
        不涉及温度采样、top-k 等技巧。

    参数：
        model (nn.Module): GPTModel 实例。
        idx (Tensor): 形状 (batch_size, T)，当前已有的 token id 序列（初始上下文）。
        max_new_tokens (int): 需要新生成的 token 数量。
        context_size (int): 模型支持的最大上下文长度，超出部分会被截断。

    返回：
        Tensor，形状 (batch_size, T + max_new_tokens)，在原序列末尾追加了新生成的 token。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 中文：只保留最近 context_size 个 token 作为模型输入，防止超出模型支持的最大长度
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)  # 中文：(batch_size, cur_len) -> (batch_size, cur_len, vocab_size)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 中文：只取序列最后一个位置的 logits，用于预测下一个 token
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 中文：贪心解码，取 logits 最大值对应的词表索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 中文：把新生成的 token 拼接到序列末尾，供下一轮迭代使用
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# Chapter 5
#####################################
def assign(left, right):
    """
    中文说明：
        辅助函数：把 right（通常是从 OpenAI 官方 GPT-2 checkpoint 里读出的 numpy 数组）
        包装成 nn.Parameter，用于赋值给模型的某个权重/偏置参数。
        赋值前会先校验两者形状是否一致，避免加载到形状不匹配的错误权重。

    参数：
        left (Tensor): 目标模型参数（仅用于读取/校验其 .shape）。
        right (np.ndarray): 待加载的权重数据。

    返回：
        torch.nn.Parameter，数据内容与 right 相同，形状与 left 相同。

    异常：
        ValueError: 当 left.shape 与 right.shape 不一致时抛出。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    中文说明：
        把从 OpenAI 官方 TensorFlow 版 GPT-2 checkpoint 中提取出的参数字典 params，
        逐一搬运并赋值给本项目手写的 GPTModel 实例 gpt 中对应的权重/偏置。

        需要特别注意的是官方 checkpoint 中 QKV 是合并存储在一个矩阵 c_attn 里的
        （最后一维大小为 3*emb_dim），这里用 np.split 按最后一维切成 q/k/v 三份；
        另外官方 TensorFlow 的 Linear 权重矩阵与 PyTorch nn.Linear 的权重矩阵是转置关系，
        所以搬运时统一做了 .T 转置。

    参数：
        gpt (GPTModel): 待加载权重的模型实例（结构需要与 params 中的层数、维度匹配）。
        params (dict): 从官方 checkpoint 解析出的嵌套字典，结构大致为
            {"wpe":..., "wte":..., "g":..., "b":..., "blocks": [ {"attn":..., "mlp":..., "ln_1":..., "ln_2":...}, ... ]}。

    返回：
        None（原地修改 gpt 的参数）。
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])  # 中文：位置嵌入表
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])  # 中文：token 嵌入表

    for b in range(len(params["blocks"])):
        # 中文：官方权重把 Q/K/V 的投影矩阵拼在一起存成 c_attn，形状最后一维是 3*emb_dim，需要按最后一维切成三份
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)  # 中文：TF 权重与 PyTorch nn.Linear 权重是转置关系，故 .T
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 中文：同理，c_attn 里的 bias 也是 Q/K/V 拼接存储，按最后一维切成三份
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 中文：注意力输出投影层（c_proj）权重同样需要转置
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 中文：前馈网络第一层（升维，c_fc）
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        # 中文：前馈网络第二层（降维，c_proj）
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 中文：两个 LayerNorm（注意力前 / 前馈前）的 scale(gamma) 和 shift(beta)
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 中文：最终的 LayerNorm 参数
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 中文：GPT-2 的输出投影层与 token 嵌入表共享权重（weight tying），因此这里直接复用 wte
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    中文说明：
        第 5 章实现的进阶文本生成函数，在 generate_text_simple 基础上增加了：
            1) top_k 采样：只保留 logits 最大的 k 个候选，其余置为 -inf；
            2) 温度采样（temperature > 0）：对 logits 做温度缩放后按概率分布采样，
               temperature 越大生成结果越随机，越小越接近贪心解码；
            3) 提前停止：当采样到 eos_id 指定的结束符时提前跳出循环。

    参数：
        model (nn.Module): GPTModel 实例。
        idx (Tensor): 形状 (batch_size, T)，初始上下文的 token id 序列。
        max_new_tokens (int): 最多新生成多少个 token。
        context_size (int): 模型支持的最大上下文长度，超出部分会被截断。
        temperature (float): 温度系数；<=0 时退化为贪心解码（argmax）。
        top_k (int, optional): 只在概率最高的 top_k 个候选中采样；为 None 时不做限制。
        eos_id (int, optional): 结束符 token id；采样到该 id 时提前停止生成。

    返回：
        Tensor，形状 (batch_size, T + n)，n <= max_new_tokens
        （如果触发提前停止，n 会小于 max_new_tokens）。

    风险/边界情况标注（仅标注，不修改，理由见下）：
        1) 第 331 行附近的 top_k 分支：
               top_logits, _ = torch.topk(logits, top_k)
               min_val = top_logits[:, -1]                       # 形状 (batch_size,)
               logits = torch.where(logits < min_val, ...)       # logits 形状 (batch_size, vocab_size)
           当 batch_size == 1 时，(1,) 可以顺利广播到 (1, vocab_size)，代码可以正常工作；
           但当 batch_size > 1 且 vocab_size != batch_size 时，
           (batch_size, vocab_size) 与 (batch_size,) 在最后一维上无法满足 PyTorch 的广播规则
           （只有当两个尺寸相等，或其中一个为 1 时才能广播），会抛出 RuntimeError。
           这是原书/原仓库在 ch05 起就存在的写法，在所有章节里都保持一致，
           且仅在 batch_size > 1 时才会触发，因此按任务要求归类为“风险项”，只标注不修改。
        2) 第 351 行附近：
               if idx_next == eos_id:
           当 batch_size > 1 时，idx_next 形状为 (batch_size, 1)，
           idx_next == eos_id 会得到一个包含多个布尔值的张量，
           用在 if 语句中会抛出 "Boolean value of Tensor with more than one element is ambiguous"。
           同样，这是原仓库沿用至今的写法，仅在 batch_size > 1 时触发，
           按任务要求归类为“风险项”，只标注不修改。
    """
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 中文：截断到模型支持的最大上下文长度
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # 中文：只取最后一个位置的 logits，形状 (batch_size, vocab_size)

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            # 中文：取 logits 最大的 top_k 个值，top_logits 形状 (batch_size, top_k)
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # 中文：top_k 个值里最小的那个，作为阈值；形状 (batch_size,)
            # 风险标注：见函数 docstring 中第 1 条——batch_size > 1 时这里的广播可能不符合预期/报错，只标注不修改
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
            # 中文：小于阈值的 logits 全部置为 -inf，softmax 后概率趋于 0，相当于只在 top_k 候选里采样

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature  # 中文：温度缩放，temperature 越小分布越陡峭（越接近贪心）

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            # 中文：减去每行最大值再做 softmax，是数值稳定的常见技巧（避免 exp 溢出），
            #      在数学上等价于不减，但在某些设备（如 mps）上数值结果更稳定一致
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            # 中文：转换为概率分布，形状 (batch_size, vocab_size)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            # 中文：按概率分布随机采样一个 token，而非固定取最大值，增加生成多样性

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
            # 中文：温度为 0（或未设置）时退化为贪心解码

        # 风险标注：见函数 docstring 中第 2 条——batch_size > 1 时这里的布尔判断可能抛出
        # "Boolean value of Tensor with more than one element is ambiguous" 错误，只标注不修改
        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        # 中文：把新生成的 token 拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
