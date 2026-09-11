# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

# ============================================================================
# 中文说明（模块级 docstring 补充）
# ----------------------------------------------------------------------------
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
# 一书附录 D（Appendix D：为训练循环添加高级特性，如学习率预热/余弦退火、
# 梯度裁剪等）所依赖的“前置代码合集”。
#
# 附录 D 的正文只演示如何在训练循环中加入学习率调度和梯度裁剪等高级技巧，
# 并不会重新实现前几章已经讲过的基础组件（分词/数据加载、多头注意力、
# LayerNorm、GELU、前馈网络、Transformer Block、完整的 GPT 模型、
# 文本生成函数、损失计算与评估函数等）。因此本文件把 第2~5章 中出现过的
# 这些基础构件原样收集在一起，方便附录 D 的主脚本直接 import 使用，
# 避免重复粘贴代码。
#
# 简单来说，本文件 = 第2章(数据加载) + 第3章(自注意力) + 第4章(GPT模型结构)
# + 第5章(训练/评估/生成辅助函数) 的“合订本”，是构建一个可训练、可推理的
# 简化版 GPT 模型所需的最小代码集合。
# ============================================================================

import tiktoken  # OpenAI 开源的高效 BPE 分词器，这里用于加载 GPT-2 的分词器
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # 用于第5章中绘制训练/验证损失曲线


#####################################
# Chapter 2
#####################################

class GPTDatasetV1(Dataset):
    """
    第2章：用于构建“下一个词预测”训练样本的数据集类。

    核心思想：把一整段长文本，先用分词器编码成 token id 序列，
    然后用一个长度为 max_length 的滑动窗口，以步长 stride 在
    token 序列上滑动切片，每个切片作为一条训练样本：
        输入 (input) : 长度为 max_length 的 token 片段
        目标 (target): 把输入整体向右移动一个位置的 token 片段
    这样模型在训练时的任务就是：给定当前的一串 token，预测下一个 token。

    参数：
        txt (str): 原始训练文本（未分词的字符串）。
        tokenizer: 具备 .encode()/.decode() 接口的分词器对象（如 tiktoken 的 GPT-2 编码器）。
        max_length (int): 每个训练样本的 token 序列长度（即上下文窗口长度）。
        stride (int): 滑动窗口每次移动的步长。stride < max_length 时，
                      相邻样本之间会有重叠部分；stride == max_length 时无重叠。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 中文：先把整段文本一次性编码成 token id 列表；
        # allowed_special 允许文本中出现 GPT-2 的特殊标记 "<|endoftext|>" 而不报错。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 中文：用滑动窗口把长 token 序列切成多个 (input, target) 训练样本对。
        # 注意 target 相比 input 整体右移了一位（即“预测下一个 token”这一自监督目标）。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集中样本（滑动窗口切片）的总数量。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        根据索引 idx 返回一条训练样本。

        返回：
            (input_ids[idx], target_ids[idx])，两者均为形状 (max_length,) 的一维张量。
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    第2章：便捷函数——把原始文本一步到位地转换成 PyTorch 的 DataLoader。

    内部流程：加载 GPT-2 分词器 -> 构造 GPTDatasetV1 -> 包装成 DataLoader，
    方便训练循环里直接按 batch 迭代取数据。

    参数：
        txt (str): 原始训练文本。
        batch_size (int): 每个批次包含多少条样本。
        max_length (int): 每条样本（滑动窗口）的 token 长度，即上下文窗口大小。
        stride (int): 滑动窗口步长。
        shuffle (bool): 是否打乱样本顺序（训练集通常为 True）。
        drop_last (bool): 若最后一个 batch 不足 batch_size 是否丢弃，
                           训练时设为 True 可以保证每个 batch 大小一致，
                           从而稳定 loss 的计算。
        num_workers (int): 数据加载使用的子进程数。

    返回：
        torch.utils.data.DataLoader 对象，每次迭代产出
        (input_batch, target_batch)，形状均为 (batch_size, max_length)。
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################

class MultiHeadAttention(nn.Module):
    """
    第3章：多头因果自注意力（Multi-Head Causal Self-Attention）模块。

    这是 Transformer / GPT 的核心组件。它让序列中的每个 token 都能“关注”
    （加权聚合）它自己以及它之前的所有 token 的信息（因果掩码保证不能看到
    未来的 token，这对自回归语言模型至关重要）。"多头"是指把 d_out 维的
    Q/K/V 拆分成 num_heads 个并行的、维度更小的子空间分别计算注意力，
    最后再拼接起来，这样模型可以同时从多个不同的表示子空间中学习信息。

    参数：
        d_in (int): 输入特征维度（每个 token 的 embedding 维度）。
        d_out (int): 输出特征维度，同时也是 Q/K/V 投影后的总维度。
        context_length (int): 支持的最大序列长度，用于预先构造因果掩码矩阵。
        dropout (float): 注意力权重上的 dropout 概率，用于正则化、防止过拟合。
        num_heads (int): 注意力头的数量，d_out 必须能被 num_heads 整除。
        qkv_bias (bool): Q/K/V 的线性层是否使用偏置项（bias）。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个头分到的维度 = 总输出维度 / 头数，这样多头拼接回去后维度仍等于 d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        # 中文：register_buffer 注册一个“非可训练参数”的张量（不会被优化器更新，
        # 但会随模型一起 .to(device) / 保存到 state_dict）。
        # torch.triu(..., diagonal=1) 生成一个上三角矩阵（对角线以上为1，其余为0），
        # 正是因果掩码所需的形状：True/1 的位置表示“未来位置，需要被屏蔽”。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        前向传播：计算多头因果自注意力。

        参数：
            x: 输入张量，形状 (b, num_tokens, d_in)，
               b=批大小，num_tokens=当前序列长度，d_in=输入特征维度。

        返回：
            context_vec: 注意力输出，形状 (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆成 (num_heads, head_dim) 两维，
        # 相当于把一个大的线性投影结果“切”成多个头各自的子向量，
        # 之后每个头独立计算注意力，从而学习不同的关注模式。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度换到前面，这样可以把 num_heads 当作类似 batch 的维度，
        # 利用张量批量矩阵乘法（batched matmul）并行计算所有头的注意力。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 中文：Q @ K^T，对每个头分别计算“查询-键”相似度打分矩阵。
        # 形状变化：(b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens)
        #          -> (b, num_heads, num_tokens, num_tokens)
        # attn_scores[..., i, j] 表示第 i 个 token 对第 j 个 token 的注意力打分（未归一化）。
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        # 中文：因为掩码是按 context_length（最大长度）预先构造好的，
        # 这里裁剪到当前实际序列长度 num_tokens，并转成布尔类型方便后续 masked_fill。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 中文：因果掩码的核心——把“未来位置”（j > i，即上三角部分）的注意力分数
        # 填成负无穷，这样经过 softmax 后这些位置的权重会变成 0，
        # 从而保证第 i 个 token 只能关注到它自己和它之前的 token（自回归特性）。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：缩放点积注意力（scaled dot-product attention）——
        # 除以 sqrt(head_dim) 是为了防止点积结果随维度增大而数值过大，
        # 导致 softmax 梯度消失，这是 Transformer 论文中提出的标准做法。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：注意力权重 @ V，得到每个 token 融合了上下文信息后的新表示；
        # 再把 num_heads 维度换回到 num_tokens 之前，为后续拼接做准备。
        # 形状：(b, num_heads, num_tokens, num_tokens) @ (b, num_heads, num_tokens, head_dim)
        #      -> (b, num_heads, num_tokens, head_dim) -> transpose -> (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多个头的输出重新拼接（reshape）回 (b, num_tokens, d_out)，
        # 相当于把各头学到的不同子空间信息重新汇总到一起。
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：最后再经过一个线性层做“混合投影”，让不同头之间的信息可以相互组合。

        return context_vec


#####################################
# Chapter 4
#####################################

class LayerNorm(nn.Module):
    """
    第4章：层归一化（Layer Normalization）模块。

    对每个 token 的特征向量（最后一维）分别做均值为0、方差为1的归一化，
    然后引入可学习的缩放参数 scale 和平移参数 shift，让模型有能力
    自行调节归一化后的分布。LayerNorm 有助于稳定深层网络的训练、
    加速收敛，是 Transformer 结构中的标准组件。

    参数：
        emb_dim (int): 特征（embedding）维度，即需要归一化的最后一维大小。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除以0的小常数（数值稳定性）
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习的缩放参数 gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的平移参数 beta

    def forward(self, x):
        """
        参数：
            x: 输入张量，形状 (..., emb_dim)，通常为 (batch, num_tokens, emb_dim)。
        返回：
            归一化并经过 scale/shift 变换后的张量，形状与输入相同。
        """
        # 中文：在最后一维（特征维）上计算均值和方差，对每个 token 独立归一化，
        # 这与 BatchNorm（在 batch 维度上归一化）不同，更适合变长序列任务。
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    第4章：GELU（Gaussian Error Linear Unit）激活函数的近似实现（tanh 近似版）。

    GELU 是一种比 ReLU 更平滑的激活函数，在 GPT 系列等 Transformer 模型的
    前馈网络（FeedForward）中被广泛使用，通常能带来更好的训练效果。
    这里实现的是原始 GPT-2 论文中使用的 tanh 近似公式，而非精确的
    高斯误差函数形式。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        参数：
            x: 任意形状的输入张量。
        返回：
            与 x 形状相同的张量，对每个元素独立应用 GELU 激活。
        """
        # 中文：GELU 的 tanh 近似公式：
        # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        # 相比 ReLU，GELU 在 0 附近更平滑，且对负值不是硬截断为0，
        # 而是允许小的负梯度通过，有助于优化。
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    第4章：Transformer Block 中的前馈网络（Feed-Forward Network, FFN）子模块。

    结构为：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)，
    即先把维度放大4倍，经过非线性激活后再压缩回原维度。这个“先扩后缩”的
    结构让模型在更高维的空间中做非线性变换，从而增强表达能力。

    参数：
        cfg (dict): 模型配置字典，需要包含键 "emb_dim"（embedding 维度）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        参数：
            x: 输入张量，形状 (batch, num_tokens, emb_dim)。
        返回：
            输出张量，形状与输入相同 (batch, num_tokens, emb_dim)。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    第4章：单个 Transformer Block（解码器层），是构成 GPT 模型的基本重复单元。

    结构为 Pre-LayerNorm 风格：
        x -> LayerNorm -> 多头因果自注意力 -> Dropout -> 残差相加
          -> LayerNorm -> 前馈网络(FFN)   -> Dropout -> 残差相加
    残差连接（shortcut/skip connection）能有效缓解深层网络的梯度消失问题，
    使得堆叠很多层 TransformerBlock 依然可以稳定训练。

    参数：
        cfg (dict): 模型配置字典，需包含 "emb_dim"、"context_length"、
                    "n_heads"、"drop_rate"、"qkv_bias" 等键。
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
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        参数：
            x: 输入张量，形状 (batch_size, num_tokens, emb_dim)。
        返回：
            输出张量，形状与输入相同 (batch_size, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        # 中文：注意力子层的残差连接——先保存输入 x 作为“捷径”，
        # 让梯度可以直接绕过注意力和归一化层反向传播，缓解深层网络训练难题。
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        # 中文：前馈网络子层同样使用残差连接，结构与上面对称。
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """
    第4章：完整的 GPT 风格自回归语言模型。

    整体结构：
        token embedding + 位置 embedding -> Dropout
          -> 堆叠 n_layers 个 TransformerBlock
          -> 最终 LayerNorm
          -> 线性输出层（映射到词表大小，得到每个位置对下一个 token 的预测 logits）

    参数：
        cfg (dict): 模型配置字典，通常包含：
            "vocab_size": 词表大小
            "emb_dim": embedding / 隐藏层维度
            "context_length": 支持的最大上下文长度
            "n_heads": 注意力头数
            "n_layers": TransformerBlock 层数
            "drop_rate": dropout 概率
            "qkv_bias": 是否给 Q/K/V 线性层加偏置
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        参数：
            in_idx: 输入的 token id 张量，形状 (batch_size, seq_len)。

        返回：
            logits: 未归一化的预测分数，形状 (batch_size, seq_len, vocab_size)，
                    表示模型在每个位置上对“下一个 token”在整个词表上的打分。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文：这里用的是可学习的“绝对位置编码”（learned absolute positional embedding），
        # 而不是 Transformer 原论文中的正弦/余弦位置编码；
        # torch.arange(seq_len) 生成 [0, 1, ..., seq_len-1] 作为位置索引去查表。
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：token embedding 与位置 embedding 直接相加，融合“词义信息”与“位置信息”。
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        # 中文：最后的线性层把隐藏维度 emb_dim 映射到词表大小 vocab_size，
        # 输出的 logits 经过 softmax 后即为下一个 token 的概率分布（此处未做 softmax）。
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    第4章：最简单的自回归文本生成函数（贪心解码，greedy decoding）。

    每一步都取模型输出的最后一个位置的 logits，选取概率最高（argmax）的
    token 作为下一个 token，拼接到序列末尾，如此循环 max_new_tokens 次。
    这是最基础的生成策略，不涉及温度采样（temperature）、top-k/top-p 等
    随机性技巧，因此生成结果是确定性的。

    参数：
        model: GPTModel（或兼容接口的模型），调用 model(idx) 返回 logits。
        idx: 初始上下文的 token id 张量，形状 (batch, T)，T 为当前长度。
        max_new_tokens (int): 需要新生成的 token 数量。
        context_size (int): 模型支持的最大上下文长度，用于裁剪过长的输入。

    返回：
        idx: 生成结束后的完整 token 序列，形状 (batch, T + max_new_tokens)。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 中文：因为位置编码表大小固定为 context_size，序列一旦超过这个长度就会
        # 索引越界，所以每一步都只保留最近的 context_size 个 token 作为输入。
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # 中文：生成阶段不需要计算梯度，用 torch.no_grad() 节省显存、加速推理。
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 中文：自回归生成只关心“下一个 token”，所以只取序列最后一个位置的 logits。
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 中文：贪心解码——直接取 logits 最大值对应的词表索引，不做随机采样。
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 中文：把新生成的 token 拼接到序列末尾，作为下一轮预测的上下文。
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# Chapter 5
####################################


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    第5章：计算单个 batch 的交叉熵损失（下一个 token 预测任务）。

    参数：
        input_batch: 输入 token id，形状 (batch_size, seq_len)。
        target_batch: 目标 token id（即输入右移一位），形状 (batch_size, seq_len)。
        model: GPTModel。
        device: 计算设备（"cpu"/"cuda"/"mps" 等）。

    返回：
        loss: 标量张量，该 batch 的平均交叉熵损失。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # 中文：flatten(0, 1) 把 (batch, seq_len, vocab_size) 的前两维合并成一维，
    # 得到 (batch*seq_len, vocab_size)；target_batch.flatten() 同理变为
    # (batch*seq_len,)。这样就把“逐位置多分类问题”整体喂给交叉熵损失函数，
    # 一次性计算所有 token 位置上的预测损失并取平均。
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    第5章：在整个（或部分）DataLoader 上计算平均损失，用于评估模型在
    训练集/验证集上的表现。

    参数：
        data_loader: 产出 (input_batch, target_batch) 的 DataLoader。
        model: GPTModel。
        device: 计算设备。
        num_batches (int, optional): 只用前 num_batches 个 batch 来估算损失
            （常用于训练过程中做快速评估，避免每次都跑完整个验证集）；
            为 None 时使用全部 batch。

    返回：
        float: 平均损失；若 data_loader 为空则返回 float("nan")。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 中文：防止传入的 num_batches 超过 data_loader 实际的 batch 数量。
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # 中文：.item() 取出标量数值，避免累积计算图占用显存
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    第5章：在训练过程中周期性地评估模型在训练集和验证集上的损失。

    参数：
        model: GPTModel。
        train_loader: 训练集 DataLoader。
        val_loader: 验证集 DataLoader。
        device: 计算设备。
        eval_iter (int): 评估时各自使用的 batch 数量（用于加速评估，
                          不必跑完整个数据集）。

    返回：
        (train_loss, val_loss): 两者均为 float 标量。
    """
    # 中文：切换到 eval 模式，关闭 dropout 等训练专用的随机行为，
    # 保证评估结果稳定、可复现。
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    # 中文：评估结束后切回 train 模式，恢复 dropout 等，继续训练。
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    第5章：训练过程中的“抽样检查”辅助函数——用当前模型生成一段文本并打印，
    直观观察模型的学习进展（定性评估）。

    参数：
        model: GPTModel。
        tokenizer: 分词器，需支持 .encode()/.decode()。
        device: 计算设备。
        start_context (str): 用作生成起点的提示文本（prompt）。

    返回：
        无返回值，直接打印生成的文本。
    """
    model.eval()
    # 中文：从位置 embedding 层的权重形状里读出模型支持的最大上下文长度
    # （pos_emb 的第0维大小就是 context_length）。
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
        # 中文：把换行符替换成空格，避免打印结果占用过多行，保持输出紧凑。
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    第5章：绘制训练/验证损失随训练轮数（epoch）变化的曲线图，
    同时在上方叠加一个以“已见 token 数”为刻度的第二 x 轴，
    方便从两个不同角度（epoch 数 / token 吞吐量）观察训练进度。

    参数：
        epochs_seen: 记录每次评估时对应的（可能是小数的）epoch 数，序列。
        tokens_seen: 记录每次评估时累计处理过的 token 数量，序列。
        train_losses: 对应的训练集损失值序列。
        val_losses: 对应的验证集损失值序列。

    返回：
        无返回值；函数内部通过 matplotlib 绘图（默认不调用 plt.show()，
        由调用方决定何时展示/保存图像）。
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    # 中文：主 x 轴（下方）以 epoch 数为刻度，画出训练/验证损失曲线。
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    # 中文：twiny() 创建一个共享 y 轴、独立 x 轴的“双生”坐标轴，
    # 用来在图像上方额外标注“已训练的 token 数量”这一维度信息。
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    # 中文：这里画一条透明（alpha=0）的曲线，目的只是让 matplotlib
    # 根据 tokens_seen 的数值范围自动生成上方 x 轴的刻度，本身不会显示出来。
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def text_to_token_ids(text, tokenizer):
    """
    第5章：把原始字符串文本编码成模型可接受的 token id 张量。

    参数：
        text (str): 输入文本。
        tokenizer: 分词器，需支持 .encode()。

    返回：
        encoded_tensor: 形状 (1, seq_len) 的张量（batch 维度为1）。
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    # 中文：unsqueeze(0) 在最前面加一维，把一维序列 (seq_len,)
    # 变成模型期望的批处理形状 (1, seq_len)，即 batch_size=1。
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    第5章：把模型生成的 token id 张量解码回可读的字符串文本。

    参数：
        token_ids: 形状 (1, seq_len) 的张量（batch 维度为1）。
        tokenizer: 分词器，需支持 .decode()。

    返回：
        str: 解码后的文本字符串。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    # 中文：squeeze(0) 去掉 batch 维，恢复成一维 token 序列，
    # 再用 .tolist() 转成 Python list 交给分词器解码为字符串。
    return tokenizer.decode(flat.tolist())
