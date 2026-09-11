# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-5.

"""
模块级中文说明:
本文件汇总了第 2~5 章中已经实现过的核心代码,供第 5 章「08_memory_efficient_weight_loading
(内存高效权重加载)」小节复用,避免重复定义。

主要内容包括:
    - MultiHeadAttention: 第 3 章实现的多头因果自注意力模块;
    - LayerNorm: 第 4 章实现的层归一化模块;
    - GELU: 第 4 章实现的 GELU 激活函数(使用 tanh 近似公式);
    - FeedForward: 第 4 章实现的前馈神经网络(MLP)模块;
    - TransformerBlock: 第 4 章实现的 Transformer 块(包含注意力子层与前馈子层,
      均带有残差连接与层归一化);
    - GPTModel: 第 4 章实现的完整 GPT 模型(词嵌入 + 位置嵌入 + 多层 Transformer 块 +
      最终归一化 + 输出投影层)。

本文件中原有的英文注释均予以保留,仅在此基础上补充详细的中文注释。
"""


import torch
import torch.nn as nn

#####################################
# Chapter 3
#####################################


class MultiHeadAttention(nn.Module):
    """
    多头因果自注意力(Multi-Head Causal Self-Attention)模块。

    该模块将输入序列投影为 Query、Key、Value 三组张量,并将其在最后一维上拆分为
    多个注意力头分别计算缩放点积注意力,再将各头输出拼接后经过一次线性投影得到最终结果。
    通过上三角掩码(causal mask)保证每个位置只能关注它自身及之前的位置(即因果关系),
    从而支持自回归语言建模。

    参数:
        d_in (int): 输入特征维度(即输入张量最后一维的大小)。
        d_out (int): 输出特征维度,同时也是 Q/K/V 投影后的总维度,
            必须能被 num_heads 整除。
        context_length (int): 支持的最大上下文长度(序列长度上限),
            用于预先构建因果掩码矩阵。
        dropout (float): 应用在注意力权重上的 dropout 比例。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, 可选): 是否在 Q/K/V 的线性层中使用偏置项,默认为 False。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 每个注意力头的维度 = 总输出维度 / 头数,保证多头拼接后维度与 d_out 一致

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 生成 Query 的线性层
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 生成 Key 的线性层
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 生成 Value 的线性层
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # 用于将多头拼接后的上下文向量再做一次线性变换(输出投影)
        self.dropout = nn.Dropout(dropout)  # 应用于注意力权重的 dropout 层
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        # 注册一个不参与梯度更新、但会随模型移动设备的缓冲区:
        # 上三角(不含对角线)为 1 的矩阵,用作因果掩码,屏蔽“未来”位置

    def forward(self, x):
        """
        前向传播:对输入序列执行多头因果自注意力计算。

        参数:
            x (torch.Tensor): 输入张量,形状为 (b, num_tokens, d_in),
                其中 b 为批大小,num_tokens 为序列长度,d_in 为输入特征维度。

        返回:
            torch.Tensor: 经过多头自注意力与输出投影后的上下文张量,
                形状为 (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape  # 解析输入形状: 批大小、序列长度、输入维度

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)  # 形状同上: (b, num_tokens, d_out)
        values = self.W_value(x)  # 形状同上: (b, num_tokens, d_out)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 将最后一维 d_out 拆分为 (num_heads, head_dim),从而隐式地把矩阵切分给各个头

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # 交换维度,使 num_heads 排在 num_tokens 之前,便于对每个头独立做批量矩阵乘法

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 计算每个头内 Query 与 Key 的点积,得到注意力分数,
        # 形状为 (b, num_heads, num_tokens, num_tokens)

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 将预先构建的掩码矩阵截取到当前序列长度,并转换为布尔类型

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 将掩码中为 True(即未来位置)对应的分数原地填充为 -inf,
        # 这样在做 softmax 时这些位置的权重会趋近于 0,实现因果掩蔽

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 对分数按最后一维做缩放(除以 head_dim 的平方根)后再做 softmax,得到注意力权重
        attn_weights = self.dropout(attn_weights)  # 对注意力权重应用 dropout 以增强泛化能力

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 用注意力权重对 Value 加权求和得到各头的上下文向量,
        # 再将 num_heads 与 num_tokens 维度换回来

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        # 将多个头的输出在最后一维拼接起来,恢复为 (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 对拼接后的上下文向量做一次可选的线性投影,融合各头信息

        return context_vec  # 返回最终的注意力输出, 形状 (b, num_tokens, d_out)


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    层归一化(Layer Normalization)模块。

    对输入张量的最后一维(特征维度)做归一化处理,使其均值为 0、方差为 1,
    然后使用可学习的缩放参数 scale 与偏移参数 shift 对归一化结果做仿射变换。

    参数:
        emb_dim (int): 需要归一化的特征维度大小(即嵌入维度)。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的极小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数,初始化为全 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数,初始化为全 0

    def forward(self, x):
        """
        前向传播:对输入的最后一维做归一化并施加可学习的仿射变换。

        参数:
            x (torch.Tensor): 输入张量,形状为 (..., emb_dim),
                最后一维为需要归一化的特征维度。

        返回:
            torch.Tensor: 归一化并仿射变换后的张量,形状与输入 x 相同。
        """
        mean = x.mean(dim=-1, keepdim=True)  # 沿最后一维计算均值,保留维度以便广播
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 沿最后一维计算方差(有偏估计)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化: 减均值除以标准差
        return self.scale * norm_x + self.shift  # 施加可学习的缩放与偏移,返回结果


class GELU(nn.Module):
    """
    GELU(高斯误差线性单元, Gaussian Error Linear Unit)激活函数模块。

    使用 GELU 的 tanh 近似公式实现,是 GPT 等 Transformer 模型中常用的激活函数,
    相较 ReLU 更加平滑。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        前向传播:对输入张量逐元素应用 GELU 激活函数(tanh 近似版本)。

        参数:
            x (torch.Tensor): 任意形状的输入张量。

        返回:
            torch.Tensor: 与输入形状相同的张量,每个元素均经过 GELU 激活。
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        # 上式为 GELU 的近似公式:
        # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))


class FeedForward(nn.Module):
    """
    前馈神经网络(Feed Forward Network, FFN)模块,即 Transformer 块中的 MLP 子层。

    结构为: Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim),
    先将特征维度放大 4 倍再压缩回原始维度,中间使用 GELU 激活函数引入非线性。

    参数:
        cfg (dict): 模型配置字典,需包含键 "emb_dim"(嵌入维度)。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 升维: emb_dim -> 4*emb_dim
            GELU(),  # 非线性激活
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 降维: 4*emb_dim -> emb_dim
        )

    def forward(self, x):
        """
        前向传播:依次经过升维线性层、GELU 激活、降维线性层。

        参数:
            x (torch.Tensor): 输入张量,形状为 (batch_size, num_tokens, emb_dim)。

        返回:
            torch.Tensor: 输出张量,形状与输入相同,为 (batch_size, num_tokens, emb_dim)。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Transformer 块(解码器风格,GPT 使用的基本构建单元)。

    包含两个子层: 多头因果自注意力子层与前馈网络子层,
    每个子层前均先做层归一化(Pre-LayerNorm),子层输出经 dropout 后再与
    子层输入(shortcut/残差)相加,形成残差连接。

    参数:
        cfg (dict): 模型配置字典,需包含 "emb_dim"、"context_length"、"n_heads"、
            "drop_rate"、"qkv_bias" 等键,用于构建内部的注意力层与前馈层。
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
        # 多头因果自注意力子层,输入输出维度均为 emb_dim
        self.ff = FeedForward(cfg)  # 前馈网络子层
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 注意力子层前的层归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 前馈子层前的层归一化
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # 应用于子层输出、残差相加前的 dropout

    def forward(self, x):
        """
        前向传播:先后经过“注意力子层 + 残差连接”和“前馈子层 + 残差连接”。

        参数:
            x (torch.Tensor): 输入张量,形状为 (batch_size, num_tokens, emb_dim)。

        返回:
            torch.Tensor: 输出张量,形状与输入相同,为 (batch_size, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        shortcut = x  # 保存输入,用作注意力子层的残差连接分支
        x = self.norm1(x)  # 先做层归一化(Pre-LN 结构)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        # 经过多头因果自注意力,形状保持 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)  # 对注意力输出应用 dropout
        x = x + shortcut  # Add the original input back
        # 残差相加: 将原始输入加回,缓解深层网络的梯度消失问题

        # Shortcut connection for feed-forward block
        shortcut = x  # 保存当前结果,用作前馈子层的残差连接分支
        x = self.norm2(x)  # 前馈子层前的层归一化
        x = self.ff(x)  # 经过前馈网络(MLP)
        x = self.drop_shortcut(x)  # 对前馈输出应用 dropout
        x = x + shortcut  # Add the original input back
        # 残差相加: 将前馈子层输入加回

        return x  # 返回该 Transformer 块的最终输出


class GPTModel(nn.Module):
    """
    完整的 GPT 语言模型。

    结构为: 词嵌入(token embedding) + 位置嵌入(positional embedding) ->
    嵌入层 dropout -> 多个堆叠的 TransformerBlock -> 最终层归一化 ->
    输出线性层(将隐藏状态映射到词表大小,得到每个位置的下一词预测 logits)。

    参数:
        cfg (dict): 模型配置字典,需包含以下键:
            - "vocab_size" (int): 词表大小;
            - "emb_dim" (int): 嵌入维度;
            - "context_length" (int): 支持的最大上下文长度;
            - "drop_rate" (float): dropout 比例;
            - "n_layers" (int): 堆叠的 Transformer 块数量;
            - "n_heads" (int): 注意力头数量;
            - "qkv_bias" (bool): Q/K/V 线性层是否使用偏置。
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词元嵌入表: vocab_size -> emb_dim
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入表: context_length -> emb_dim
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 应用于词嵌入+位置嵌入之和的 dropout

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # 按配置堆叠 n_layers 个 TransformerBlock,顺序执行

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 所有 Transformer 块之后的最终层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 输出投影层,将隐藏状态映射为词表大小的 logits,不使用偏置项

    def forward(self, in_idx):
        """
        前向传播:将输入的词元 ID 序列转换为下一词预测的 logits。

        参数:
            in_idx (torch.Tensor): 输入的词元 ID 张量,形状为 (batch_size, seq_len),
                每个元素为词表中的整数索引。

        返回:
            torch.Tensor: 预测的 logits 张量,形状为
                (batch_size, seq_len, vocab_size)。
        """
        batch_size, seq_len = in_idx.shape  # 解析输入形状: 批大小与序列长度
        tok_embeds = self.tok_emb(in_idx)  # 查表得到词元嵌入, 形状 (batch_size, seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 生成 [0, 1, ..., seq_len-1] 的位置索引并查表得到位置嵌入, 形状 (seq_len, emb_dim)
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 词嵌入与位置嵌入相加(广播机制),得到最终输入嵌入
        x = self.drop_emb(x)  # 对嵌入结果应用 dropout
        x = self.trf_blocks(x)  # 依次通过所有堆叠的 Transformer 块
        x = self.final_norm(x)  # 最终层归一化
        logits = self.out_head(x)  # 线性投影得到词表大小的 logits
        return logits  # 返回形状为 (batch_size, seq_len, vocab_size) 的预测结果
