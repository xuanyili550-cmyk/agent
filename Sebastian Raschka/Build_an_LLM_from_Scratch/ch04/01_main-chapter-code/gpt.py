# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

# ============================================================
# 中文模块说明
# ============================================================
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
# 一书第 2~4 章内容的汇总代码，把书中前几章逐步讲解、逐步搭建的组件整合到一个
# 可以独立运行的脚本中，形成一个"迷你版 GPT-2"模型的完整实现。
#
# 具体来说，本文件包含：
#   1. 第 2 章：文本数据处理相关代码
#      - GPTDatasetV1：使用滑动窗口把长文本切分成 (输入, 目标) 训练样本对的 Dataset
#      - create_dataloader_v1：基于 tiktoken 分词器和上面的 Dataset 构造 DataLoader
#   2. 第 3 章：注意力机制相关代码
#      - MultiHeadAttention：带因果掩码（causal mask）的多头自注意力实现
#   3. 第 4 章：完整 GPT 模型结构相关代码
#      - LayerNorm：层归一化
#      - GELU：GELU 激活函数（这里使用的是原始 GPT-2 论文中的近似实现）
#      - FeedForward：Transformer 中的前馈网络（MLP）子层
#      - TransformerBlock：把注意力子层和前馈子层组合起来的 Transformer 块
#      - GPTModel：堆叠多个 TransformerBlock 构成的完整 GPT 模型
#      - generate_text_simple：最简单的自回归文本生成函数（贪心解码，每步取概率最大的 token）
#
# 学习本文件时，建议按照"数据 -> 注意力 -> 模型结构 -> 生成"的顺序阅读，
# 这也正是原书 2~4 章的讲解顺序，能帮助你理解一个 GPT 模型是如何从零一步步搭建起来的。
# ============================================================

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
    第 2 章介绍的自定义数据集类，用于把一整段原始文本转换成可供语言模型训练的
    (输入序列, 目标序列) 样本对。

    核心思路是"滑动窗口"：把分词后的 token id 序列，以固定长度 max_length 为窗口、
    stride 为步长切分成很多个互相有重叠（或不重叠，取决于 stride 大小）的片段。
    对于每个片段：
      - 输入 input_chunk 是窗口内的 token 序列
      - 目标 target_chunk 是把输入整体右移一位后的 token 序列（即"预测下一个词"任务）

    参数：
        txt (str): 原始训练文本（一整篇文章/一本书等）。
        tokenizer: 分词器对象（这里传入的是 tiktoken 的 GPT-2 编码器），需要有 encode 方法。
        max_length (int): 每个训练样本的 token 序列长度（即上下文窗口大小）。
        stride (int): 滑动窗口每次移动的步长。stride < max_length 时窗口之间会有重叠，
                      stride == max_length 时窗口不重叠。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 中文：先把整段文本一次性编码成 token id 列表。
        # allowed_special={"<|endoftext|>"} 表示允许文本中出现 GPT-2 的特殊结束符标记，
        # 否则 tiktoken 遇到这个特殊字符串默认会报错。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 中文：用滑动窗口把整本书的 token 序列切分成多个长度为 max_length 的重叠片段。
        # range 的终止条件 len(token_ids) - max_length 保证每个片段都能取到完整的
        # max_length 长度的输入，以及再往后一位的目标序列，不会越界。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            # 目标序列相对输入序列整体右移一位：即"给定前 i 个 token，预测第 i+1 个 token"。
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """中文：返回数据集中样本（切分出的窗口片段）的总数量，供 DataLoader 使用。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        中文：按索引 idx 取出一对训练样本。

        返回：
            (input_ids[idx], target_ids[idx])，两者都是形状为 (max_length,) 的一维张量，
            target 相对 input 整体右移一位。
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    中文说明：
    便捷函数，封装了"创建分词器 -> 构造 GPTDatasetV1 -> 包装成 DataLoader"这一整套流程，
    方便在训练脚本中一行代码拿到可迭代的批次数据。

    参数：
        txt (str): 原始训练文本。
        batch_size (int): 每个批次包含的样本数。
        max_length (int): 每个样本的 token 序列长度（上下文窗口大小）。
        stride (int): 滑动窗口步长，控制样本之间的重叠程度。
        shuffle (bool): 是否在每个 epoch 打乱样本顺序。
        drop_last (bool): 若最后一个批次样本数不足 batch_size，是否丢弃该批次
                           （训练时通常设为 True，以保证每个批次形状一致，避免损失值突变）。
        num_workers (int): 数据加载使用的子进程数量。

    返回：
        torch.utils.data.DataLoader：每次迭代产出一个批次的 (input_ids, target_ids)，
        形状均为 (batch_size, max_length)。
    """
    # Initialize the tokenizer
    # 中文：使用 GPT-2 的 BPE 分词器（tiktoken 提供的高效实现）。
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    # 中文：用上面的分词器和滑动窗口切分逻辑，构造出所有 (输入, 目标) 样本对。
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # 中文：DataLoader 负责按 batch_size 打包样本、是否打乱顺序、是否丢弃不完整批次等。
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """
    中文说明：
    第 3 章实现的"多头自注意力"（Multi-Head Self-Attention）模块，
    这是 Transformer / GPT 模型最核心的组件，负责让序列中的每个 token
    能够"看到"（关注）其之前的所有 token，并根据相关性加权聚合信息。

    这里实现的是**因果自注意力**（causal self-attention）：通过一个上三角掩码矩阵，
    保证第 t 个位置只能关注第 1..t 个位置（不能看到未来的 token），
    这正是 GPT 这种自回归语言模型能够做"预测下一个词"训练的关键设计。

    实现上采用了"权重拆分"（weight split）技巧：先用一个大的线性层一次性计算出
    完整维度 d_out 的 Q/K/V，再通过 reshape 把最后一维拆分成 (num_heads, head_dim)，
    从而并行计算多个注意力头，而不需要为每个头单独定义一套线性层参数。

    参数：
        d_in (int): 输入特征维度（即每个 token 的 embedding 维度）。
        d_out (int): 输出特征维度，同时也是所有注意力头拼接后的总维度。
        context_length (int): 支持的最大序列长度，用于预先构造因果掩码矩阵。
        dropout (float): 注意力权重上应用的 dropout 概率，用于正则化、防止过拟合。
        num_heads (int): 注意力头的数量，d_out 必须能被 num_heads 整除。
        qkv_bias (bool): 计算 Q/K/V 的线性层是否使用偏置项（bias），默认为 False。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 中文：多头注意力要求总输出维度能被头数整除，这样才能把 d_out 均分给每个头。
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：head_dim 是每个注意力头单独处理的维度大小，
        # 例如 d_out=768, num_heads=12 时，head_dim=64。

        # 中文：Q/K/V 三个线性投影层，把输入 x 分别映射为 Query、Key、Value。
        # 注意这里是一次性输出完整的 d_out 维度，之后再拆分成多个头，而不是每个头单独一层。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # 中文：out_proj 用于把多个头拼接后的结果再做一次线性变换，融合各头信息。
        self.dropout = nn.Dropout(dropout)
        # 中文：注册一个不参与梯度更新的缓冲区 mask，是一个上三角矩阵（对角线以上为 1）。
        # torch.triu(..., diagonal=1) 生成的矩阵中，位置 (i, j) 当 j > i 时为 1，代表"未来位置"，
        # 后面会用这个掩码把对应位置的注意力分数设为 -inf，从而实现因果（只能看过去）约束。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        中文：前向传播，计算带因果掩码的多头自注意力输出。

        参数：
            x: 形状 (b, num_tokens, d_in) 的输入张量，b 为批次大小，num_tokens 为序列长度。

        返回：
            context_vec: 形状 (b, num_tokens, d_out) 的注意力输出，
            每个位置融合了它能"看到"的所有历史位置的信息。
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆分成 (num_heads, head_dim)，
        # 相当于把一个大的线性投影结果"切"成 num_heads 份，每份供一个头使用。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度换到第二维，方便后续对每个头独立做矩阵乘法（批量矩阵乘法会
        # 把前面除最后两维之外的维度当作"批次维"并行处理）。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 中文：核心的"缩放点积注意力"计算。queries @ keys.transpose(2, 3) 对每个头分别
        # 计算 (num_tokens, head_dim) @ (head_dim, num_tokens) -> (num_tokens, num_tokens)，
        # 得到每个 query 位置对每个 key 位置的原始注意力分数（越大代表越相关）。
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # attn_scores 形状: (b, num_heads, num_tokens, num_tokens)

        # Original mask truncated to the number of tokens and converted to boolean
        # 中文：因为 mask 是按 context_length（最大长度）预先构造的，这里根据实际序列长度
        # num_tokens 截取需要的部分，并转成布尔类型，True 表示"需要屏蔽的未来位置"。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 中文：把未来位置对应的注意力分数原地填充为负无穷，这样经过 softmax 后这些位置的
        # 权重会趋近于 0，从而保证因果性（当前位置看不到未来 token）。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：对注意力分数做缩放（除以 sqrt(head_dim)）再做 softmax，得到归一化的注意力权重。
        # 缩放的目的是防止 head_dim 较大时点积结果数值过大，导致 softmax 梯度消失。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 中文：对注意力权重做 dropout 正则化

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：用注意力权重对 values 做加权求和，得到每个位置融合了历史信息的上下文向量；
        # 再把 (b, num_heads, num_tokens, head_dim) 转置回 (b, num_tokens, num_heads, head_dim)，
        # 为下一步"拼接多头"做准备。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多个头的输出在最后一维拼接回 d_out（先 contiguous 保证内存连续，view 才能安全使用）。
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：最后再做一次线性变换（输出投影），让模型能够学习如何融合不同头捕获到的信息。

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    中文说明：
    第 4 章实现的层归一化（Layer Normalization）模块。
    与 BatchNorm 不同，LayerNorm 是对每个样本自身在特征维度（最后一维，即 emb_dim）上
    做归一化，使其均值为 0、方差为 1，然后再通过可学习的 scale（缩放）和 shift（平移）
    参数把分布调整到模型认为合适的位置。这样做的好处是不依赖 batch 内其他样本，
    在 NLP 变长序列场景下更稳定，也是 Transformer 结构里的标准做法。

    参数：
        emb_dim (int): 特征（embedding）维度大小，即在最后一维上做归一化。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除以 0 的极小数值稳定项
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 中文：可学习的缩放参数，初始化为全 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 中文：可学习的偏移参数，初始化为全 0

    def forward(self, x):
        """
        中文：对输入 x 在最后一维（emb_dim）上做归一化。

        参数：
            x: 形状 (..., emb_dim) 的张量，通常是 (batch, num_tokens, emb_dim)。

        返回：
            与 x 形状相同的张量，每个样本在 emb_dim 维度上均值为 0、方差为 1，
            再经过 scale/shift 仿射变换。
        """
        mean = x.mean(dim=-1, keepdim=True)
        # 中文：unbiased=False 表示使用有偏方差估计（除以 n 而非 n-1），
        # 这是深度学习框架里 LayerNorm 的标准实现方式。
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    中文说明：
    GELU（Gaussian Error Linear Unit）激活函数的实现，这里使用的是原始 GPT-2 论文
    中采用的 tanh 近似公式（而不是精确的误差函数 erf 形式），计算更快、数值上足够接近。
    相比 ReLU，GELU 是平滑曲线，在负数区域也有非零梯度，实践中常用于 Transformer 的
    前馈网络中，效果通常优于 ReLU。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        中文：GELU 的 tanh 近似公式：
            GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        对输入张量 x 逐元素计算，不改变张量形状。
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    中文说明：
    Transformer 块中的前馈网络（Feed-Forward Network，又称 MLP）子层。
    结构是"线性层升维 -> GELU 激活 -> 线性层降维"，即先把 emb_dim 扩大到 4 倍
    （这是原始 Transformer / GPT 论文中的经验设计，扩大隐藏层维度以增强模型表达能力），
    经过非线性激活后再投影回原始的 emb_dim，方便与残差连接相加。

    参数：
        cfg (dict): 模型配置字典，这里用到 cfg["emb_dim"]（embedding 维度）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 中文：升维，emb_dim -> 4*emb_dim
            GELU(),                                          # 中文：非线性激活
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 中文：降维回 emb_dim，便于残差相加
        )

    def forward(self, x):
        """
        中文：前向传播。

        参数：
            x: 形状 (batch, num_tokens, emb_dim) 的张量。

        返回：
            形状同样为 (batch, num_tokens, emb_dim) 的张量（升维后又降维回原始维度）。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    中文说明：
    单个 Transformer 块（Block），是构成 GPT 模型的基本重复单元。
    每个块内部包含两个子层：
        1. 多头自注意力子层（MultiHeadAttention）+ 残差连接
        2. 前馈网络子层（FeedForward）+ 残差连接
    每个子层前面都先做一次 LayerNorm（这是"Pre-LN"结构，即先归一化再进入子层，
    与原始 Transformer 论文的"Post-LN"结构略有不同，Pre-LN 训练时数值更稳定）。
    子层输出还会经过 dropout 后再与残差（shortcut）相加。

    参数：
        cfg (dict): 模型配置字典，包含 emb_dim、context_length、n_heads、drop_rate、qkv_bias 等。
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
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 中文：注意力子层前的归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 中文：前馈子层前的归一化
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 中文：残差分支上的 dropout

    def forward(self, x):
        """
        中文：前向传播，依次执行"注意力子层 + 残差"和"前馈子层 + 残差"。

        参数：
            x: 形状 (batch, num_tokens, emb_dim) 的输入张量。

        返回：
            形状同为 (batch, num_tokens, emb_dim) 的输出张量。
        """
        # Shortcut connection for attention block
        # 中文：保存输入，作为注意力子层的残差连接（shortcut/skip connection）分支。
        # 残差连接能有效缓解深层网络的梯度消失问题，是训练深层 Transformer 的关键技巧。
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：把注意力子层的输出与原始输入相加，即 x = x + Attention(LayerNorm(x))

        # Shortcut connection for feed-forward block
        # 中文：前馈子层同理，也使用残差连接。
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：x = x + FeedForward(LayerNorm(x))

        return x


class GPTModel(nn.Module):
    """
    中文说明：
    完整的 GPT 模型结构，整合了词嵌入、位置嵌入、多个堆叠的 TransformerBlock、
    最终归一化层以及输出投影层（语言模型头），构成一个可以端到端训练/推理的
    自回归语言模型。

    整体数据流：
        token id 序列
            -> 词嵌入 + 位置嵌入（得到融合了词义和位置信息的向量）
            -> Dropout
            -> N 层 TransformerBlock（逐层提取/融合上下文信息）
            -> 最终 LayerNorm
            -> 线性层输出到词表大小的 logits（每个位置对下一个 token 的预测分数）

    参数：
        cfg (dict): 模型配置字典，需要包含以下键：
            - vocab_size: 词表大小
            - context_length: 支持的最大上下文长度（位置嵌入的行数）
            - emb_dim: 嵌入/隐藏层维度
            - n_heads: 注意力头数
            - n_layers: Transformer 块的堆叠层数
            - drop_rate: dropout 概率
            - qkv_bias: 注意力中 Q/K/V 线性层是否使用偏置
    """
    def __init__(self, cfg):
        super().__init__()
        # 中文：词嵌入表，把每个 token id 映射为一个 emb_dim 维的向量。
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 中文：位置嵌入表，为序列中每个位置（0, 1, 2, ...）学习一个 emb_dim 维的向量，
        # 用来给模型提供"顺序"信息（因为注意力机制本身对位置是不敏感的）。
        # 这里使用的是可学习的绝对位置嵌入（GPT-2 风格），而非正弦位置编码或 RoPE。
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 中文：用 nn.Sequential 堆叠 n_layers 个 TransformerBlock，构成模型的主干（backbone）。
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 中文：输出投影层（也叫"语言模型头" / LM head），把最后的隐藏状态映射到词表维度，
        # 得到每个位置上、对词表中每个 token 的预测分数（logits）。bias=False 是 GPT-2 的常见设置。
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        中文：前向传播，输入 token id 序列，输出下一个 token 的预测 logits。

        参数：
            in_idx: 形状 (batch_size, seq_len) 的整数张量，每个元素是词表中的 token id。

        返回：
            logits: 形状 (batch_size, seq_len, vocab_size) 的张量，
            表示模型在每个位置上对"下一个 token"在整个词表上的预测分数（未归一化，
            需要经过 softmax 才是概率分布）。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文：位置嵌入根据序列长度动态生成 [0, 1, ..., seq_len-1] 的位置索引，
        # 并放到与输入相同的设备上（CPU/GPU），避免设备不匹配报错。
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：词嵌入与位置嵌入直接相加（而不是拼接），是 GPT-2 的标准做法；
        # 这里利用了广播机制，pos_embeds 形状 (seq_len, emb_dim) 会广播到每个 batch。
        x = self.drop_emb(x)
        x = self.trf_blocks(x)  # 中文：依次经过所有 Transformer 块，逐层提取上下文特征
        x = self.final_norm(x)  # 中文：最后再做一次归一化，稳定输出分布
        logits = self.out_head(x)  # 中文：投影到词表维度，得到每个位置的预测 logits
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    中文说明：
    最简单的自回归文本生成函数，采用"贪心解码"（greedy decoding）策略：
    每一步都选择模型预测概率（logits）最高的那个 token 作为下一个生成的 token，
    然后把它拼接到已有序列末尾，重复 max_new_tokens 次，从而逐步"续写"文本。

    注意：这是最基础的生成方式，没有引入随机采样（如温度采样、top-k、top-p），
    因此对同样的输入，生成结果是完全确定（deterministic）的。

    参数：
        model: 已训练好的 GPTModel 实例（推理时通常配合 model.eval() 使用）。
        idx: 形状 (batch, num_tokens) 的张量，当前已有的上下文 token id 序列。
        max_new_tokens (int): 要新生成的 token 数量。
        context_size (int): 模型支持的最大上下文长度（与训练时的 context_length 对应），
                             用于在生成过程中裁剪过长的上下文。

    返回：
        idx: 形状 (batch, num_tokens + max_new_tokens) 的张量，
        即原始上下文后面拼接了新生成的 token。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 中文：由于模型的位置嵌入只支持最多 context_size 长度，若当前序列超过这个长度，
        # 只截取最后 context_size 个 token 作为输入，防止位置嵌入索引越界。
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # 中文：推理阶段不需要计算梯度，用 torch.no_grad() 节省显存、加快速度。
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 中文：语言模型每个位置都会输出对"下一个 token"的预测，但生成时我们只关心
        # 序列最后一个位置的预测结果（因为前面的位置对应的"下一个词"已经是已知的历史 token 了）。
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 中文：贪心解码核心一步——直接取 logits 最大值对应的词表索引，不做采样。
        # keepdim=True 保留维度，便于后面拼接。
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 中文：把新生成的 token 拼接到序列末尾，作为下一轮生成的输入上下文。
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def main():
    """
    中文说明：
    脚本入口函数，演示如何：
        1. 定义一个 GPT-2 (124M 参数规模) 的配置字典；
        2. 实例化 GPTModel 并切换到 eval 模式（关闭 dropout，保证推理结果确定）；
        3. 用 tiktoken 对一段起始文本进行编码；
        4. 调用 generate_text_simple 做自回归生成（此时模型未经训练，是随机初始化权重，
           所以生成的文本在语义上通常是无意义的，这里主要用于验证代码能跑通、
           展示"张量形状是如何变化的"这一工程流程）；
        5. 把生成的 token id 解码回文本并打印结果。
    """
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }
    # 中文：以上是 GPT-2 "小型"版本 (约 124M 参数) 的经典配置，
    # 12 层、12 个注意力头、768 维隐藏层、支持 1024 长度上下文。

    torch.manual_seed(123)  # 中文：固定随机种子，保证模型权重初始化可复现
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout
    # 中文：eval() 会关闭 dropout 等训练专用行为，使推理结果是确定的（不受随机性影响）。

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    # 中文：unsqueeze(0) 在最前面增加一个 batch 维度，把形状从 (num_tokens,) 变成 (1, num_tokens)，
    # 因为模型的 forward 期望输入形状是 (batch_size, seq_len)。
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    # 中文：squeeze(0) 去掉 batch 维度，还原成一维 token 序列，再解码回文本字符串。
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
