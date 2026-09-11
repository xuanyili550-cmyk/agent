# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# ============================================================================
# 本文件是第 2 章「GPT 模型架构」的配套模块，实现了 GPT 的核心构件：
#   - 数据管道：GPTDatasetV1 / create_dataloader_v1（滑动窗口切分训练样本）
#   - 注意力：MultiHeadAttention（多头因果自注意力）
#   - 归一化：LayerNorm（层归一化）
#   - 激活：GELU（高斯误差线性单元的 tanh 近似）
#   - 前馈：FeedForward（逐位置的 MLP，先扩张 4 倍再收缩）
#   - Transformer 块：TransformerBlock（Pre-LN 结构 + 两处残差连接）
# 02.ipynb 中的 GPTModel 会把这些构件堆叠起来组成完整模型。
# ============================================================================

import tiktoken                                       # OpenAI 的 BPE 分词器（GPT-2 使用同一套 tokenizer）
import torch                                          # PyTorch 主库，提供张量与自动求导
import torch.nn as nn                                 # 神经网络模块（层、参数、容器等）
from torch.utils.data import Dataset, DataLoader      # 数据集抽象与批量加载器


# ----------------------------------------------------------------------------
# GPTDatasetV1：把一整段文本切成「输入序列 -> 目标序列」的监督学习样本。
# 语言模型的训练目标是「预测下一个 token」，因此 target 就是 input 整体右移一位。
# 通过滑动窗口（sliding window）在长文本上生成大量重叠的定长片段。
# ----------------------------------------------------------------------------
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # input_ids / target_ids 分别保存所有样本的输入 token 序列与目标 token 序列
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 先把整篇文本编码成一维 token id 列表；allowed_special 允许保留 <|endoftext|> 特殊标记
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 以 stride 为步长滑动窗口：每次取 max_length 个 token 作为输入，
        # 对应目标是同一窗口整体右移一位（即每个位置预测其下一个 token）。
        # stride < max_length 时窗口相互重叠，可从有限文本中构造更多训练样本。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]            # 输入：[i, i+max_length)
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标：右移一位 [i+1, i+max_length+1)
            self.input_ids.append(torch.tensor(input_chunk))     # 转成张量便于后续批处理
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # Dataset 协议要求：返回样本总数（供 DataLoader 计算 batch 数量）
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Dataset 协议要求：按索引返回一个（输入, 目标）样本对
        return self.input_ids[idx], self.target_ids[idx]


# ----------------------------------------------------------------------------
# create_dataloader_v1：封装 tokenizer + 数据集 + DataLoader 的便捷工厂函数。
# 返回可迭代的批量加载器，每次产出形状为 (batch_size, max_length) 的输入/目标张量。
# ----------------------------------------------------------------------------
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    # 使用 GPT-2 的 BPE 分词器（词表大小 50257），与预训练权重保持一致
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    # 用滑动窗口把文本切成样本集合
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # drop_last=True 丢弃最后不足一个 batch 的残余样本，保证每个 batch 尺寸一致（训练更稳定）
    # num_workers 控制并行加载的子进程数（0 表示在主进程加载）
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


# ----------------------------------------------------------------------------
# MultiHeadAttention：多头因果自注意力，是 Transformer 的核心。
# 直觉：每个 token 通过 Query 去"询问"其它所有 token 的 Key，得到匹配分数（注意力权重），
#       再按权重聚合各 token 的 Value，得到融合了上下文信息的表示（context vector）。
# "多头"= 把表示切成 num_heads 份并行做注意力，让不同头关注不同的语义子空间。
# "因果"(causal) = 每个位置只能看到自己及之前的 token（用上三角掩码屏蔽未来），
#                  这是自回归语言建模的前提，避免"偷看"答案。
# ----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # d_out 必须能被 num_heads 整除，才能把输出维度均匀分给每个头
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
                                            # 每个头的维度 = 总输出维度 / 头数；多头总参数量与单头相当

        # Q/K/V 三个线性投影：把输入 (d_in) 映射到 (d_out) 空间。
        # qkv_bias=False 时不加偏置（GPT-2 原版即如此）。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
                                                 # 合并所有头后再做一次线性投影，混合各头信息
        self.dropout = nn.Dropout(dropout)       # 作用在注意力权重上，训练时随机丢弃以正则化
        # 预先构造上三角因果掩码（对角线以上为 1）。register_buffer 使其随模型迁移设备(.to)
        # 但不作为可训练参数。triu(..., diagonal=1) 表示屏蔽严格未来位置（不含对角线自身）。
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # 输入形状：x = (batch, num_tokens, d_in)
        b, num_tokens, d_in = x.shape

        # 线性投影得到 Q/K/V，形状均为 (b, num_tokens, d_out)
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 把最后一维 d_out 拆成 (num_heads, head_dim)，实现"多头"切分（无需真正复制数据）
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 交换维度，把 num_heads 提到前面，使每个头成为独立的 (num_tokens, head_dim) 矩阵，
        # 便于后续对每个头做批量矩阵乘法。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 注意力分数 = Q · Kᵀ：对每个头，(num_tokens, head_dim) @ (head_dim, num_tokens)
        # 得到 (b, num_heads, num_tokens, num_tokens)，即每个 query 对每个 key 的相似度。
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        # 把预存掩码裁剪到当前序列长度并转 bool（True 处代表"未来"，需屏蔽）
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 将被屏蔽位置的分数设为 -inf，softmax 后其权重趋近 0（无法关注未来 token）
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放点积：除以 sqrt(head_dim) 防止维度增大时点积过大、softmax 梯度饱和；
        # dim=-1 对每个 query 的所有 key 做归一化，得到概率分布形式的注意力权重。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 训练时随机置零部分权重（正则化）

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 用注意力权重对 Value 加权求和得到上下文向量，再转置把 num_tokens 换回第二维
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 把多个头重新拼回 d_out 维；contiguous() 保证内存连续以便 view 重塑
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
                                                  # 输出投影：融合各头信息，得到最终 (b, num_tokens, d_out)

        return context_vec


# ----------------------------------------------------------------------------
# LayerNorm：层归一化。对每个 token 的特征向量（最后一维）做标准化，
# 使其均值为 0、方差为 1，再用可学习的 scale/shift 恢复表达能力。
# 作用：稳定各层激活分布、加速收敛、缓解梯度问题。与 BatchNorm 不同，
# LayerNorm 在特征维度上归一化，不依赖 batch，适合变长序列与小 batch。
# ----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的极小值（数值稳定）
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习缩放参数 γ，初始为 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习平移参数 β，初始为 0

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)               # 沿特征维求均值，keepdim 便于广播
        var = x.var(dim=-1, keepdim=True, unbiased=False) # 方差用有偏估计（除以 N，与 GPT-2 一致）
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 标准化到均值0方差1
        return self.scale * norm_x + self.shift           # 仿射变换恢复模型的表达自由度


# ----------------------------------------------------------------------------
# GELU：高斯误差线性单元激活函数（GPT 系列使用）。
# 相比 ReLU 的硬截断，GELU 是平滑的，负值区间不是直接归零而是按概率保留，
# 通常带来更好的收敛与表现。这里用 tanh 近似公式（GPT-2 原版实现）：
#   GELU(x) ≈ 0.5·x·(1 + tanh[ √(2/π)·(x + 0.044715·x³) ])
# ----------------------------------------------------------------------------
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 逐元素计算，形状不变。0.044715 与 √(2/π) 是近似公式中的经验常数。
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# ----------------------------------------------------------------------------
# FeedForward：Transformer 块内的逐位置前馈网络（position-wise MLP）。
# 结构：Linear(emb -> 4·emb) -> GELU -> Linear(4·emb -> emb)。
# "先扩张 4 倍再收缩"是 Transformer 惯例：在更高维空间做非线性变换，
# 提升表达能力，然后压回原维度以便残差相加。它对每个 token 独立作用。
# ----------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 升维：emb_dim -> 4·emb_dim
            GELU(),                                          # 非线性激活
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 降维：4·emb_dim -> emb_dim（还原维度）
        )

    def forward(self, x):
        # 形状不变：(b, num_tokens, emb_dim) -> (b, num_tokens, emb_dim)
        return self.layers(x)


# ----------------------------------------------------------------------------
# TransformerBlock：一个完整的 Transformer 层，GPTModel 会堆叠 n_layers 个。
# 采用 Pre-LN（前置层归一化）结构：先归一化再进子层，最后加残差。
# 相比原始 Post-LN，Pre-LN 使深层网络训练更稳定、更易收敛。
# 两处残差连接（shortcut）分别包住注意力子层与前馈子层，
# 让梯度可直接回流、缓解深层退化，是训练深层 Transformer 的关键。
# ----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 多头因果自注意力子层：d_in = d_out = emb_dim，头数与 dropout 来自配置
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)                    # 逐位置前馈子层
        self.norm1 = LayerNorm(cfg["emb_dim"])        # 注意力子层前的 LayerNorm
        self.norm2 = LayerNorm(cfg["emb_dim"])        # 前馈子层前的 LayerNorm
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 残差分支上的 dropout（正则化）

    def forward(self, x):
        # Shortcut connection for attention block
        # —— 注意力子层（Pre-LN + 残差）——
        shortcut = x            # 保存输入用于残差相加
        x = self.norm1(x)       # 先归一化（Pre-LN）
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]  # 再做注意力
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back  # 残差：加回原始输入

        # Shortcut connection for feed forward block
        # —— 前馈子层（Pre-LN + 残差）——
        shortcut = x            # 更新残差起点为上一子层的输出
        x = self.norm2(x)       # 归一化
        x = self.ff(x)          # 前馈变换
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back  # 残差：加回

        return x
