# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块对应《从零构建大语言模型》第 3 章：注意力机制（Attention Mechanisms）。

模块内容概览（由简单到复杂，层层递进）：
    1. SelfAttention_v1        —— 用可训练的原始参数矩阵 (nn.Parameter) 手工实现的
                                    最基础的自注意力（无掩码、无多头）。
    2. SelfAttention_v2        —— 用 nn.Linear 层替代手写参数矩阵实现的自注意力，
                                    写法更接近实际工程实现，可选是否带偏置项。
    3. CausalAttention         —— 在自注意力基础上加入“因果掩码”（causal mask），
                                    保证第 t 个位置只能看到 <= t 的位置（防止“偷看”未来
                                    的 token），并支持批次维度（batch）与 Dropout。
    4. MultiHeadAttentionWrapper —— 通过简单地并行堆叠多个 CausalAttention 实例，
                                    并在输出维度上拼接（concat）来实现“多头”注意力。
                                    实现直观但效率较低（每个头都要重复做线性变换）。
    5. MultiHeadAttention      —— 更高效的多头注意力实现：只用一组大的 W_query /
                                    W_key / W_value 线性层做一次投影，再通过
                                    reshape + transpose 把最后一维“拆分”成
                                    (num_heads, head_dim)，实现多头并行计算，
                                    最后再合并（合并）各头结果并做输出投影。
                                    这是实际大模型中常见的高效写法。
    6. PyTorchMultiHeadAttention（Bonus）—— 使用 PyTorch 内置的高度优化算子
                                    `F.scaled_dot_product_attention`（可自动使用
                                    Flash Attention 等加速实现）来实现多头注意力，
                                    速度和显存效率通常优于手写版本。

核心概念贯穿全文：
    - Q（Query，查询）/ K（Key，键）/ V（Value，值）：由输入 x 分别经过三个线性变换
      得到，用于计算“每个 token 应该关注其他哪些 token”。
    - 缩放点积注意力（Scaled Dot-Product Attention）：
      attn_scores = Q @ K^T ，再除以 sqrt(head_dim) 做缩放，防止点积值过大导致
      softmax 梯度消失，最后经过 softmax 得到注意力权重 attn_weights，
      与 V 加权求和得到上下文向量 context_vec。
    - 因果掩码（Causal Mask）：将注意力分数矩阵中“未来位置”对应的分数设为 -inf，
      使得 softmax 后对应权重趋近于 0，从而实现“只能看过去，不能看未来”的
      自回归（autoregressive）特性。
    - 多头拆分与合并：将维度为 d_out 的 Q/K/V 在最后一维上拆分为
      (num_heads, head_dim)，每个头独立计算注意力（可以关注输入的不同子空间/
      不同模式），最后再把各头的输出拼接回 d_out 维度，通过一个线性层
      （out_proj）融合各头信息。
"""

import torch
import torch.nn as nn


class SelfAttention_v1(nn.Module):
    """最基础的自注意力实现（版本 1）。

    直接用可训练的 `nn.Parameter` 权重矩阵（而非 nn.Linear 层）手工实现
    Q/K/V 的线性变换，便于直观理解自注意力的底层数学运算。
    不包含因果掩码、多头拆分、Dropout、批次维度处理等高级特性，
    仅适用于单个序列（无 batch 维）的场景，是理解注意力机制的“教学基线”。

    参数：
        d_in (int): 输入 token 向量的维度（embedding 维度）。
        d_out (int): 输出的 Q/K/V 向量维度（也是 context 向量的维度）。
    """

    def __init__(self, d_in, d_out):
        super().__init__()
        # 三个可训练权重矩阵，形状均为 (d_in, d_out)。
        # 用 nn.Parameter 包裹的随机张量作为“手写版线性层”的权重，
        # 分别用于将输入 x 投影为 Query / Key / Value。
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        """前向传播：计算自注意力输出的上下文向量。

        参数：
            x (Tensor): 输入序列，形状 (num_tokens, d_in)（无 batch 维）。

        返回：
            Tensor: 上下文向量 context_vec，形状 (num_tokens, d_out)。
        """
        # 分别用输入 x 与三个权重矩阵相乘，得到 Key / Query / Value。
        # x: (num_tokens, d_in) @ W_*: (d_in, d_out) -> (num_tokens, d_out)
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # 计算注意力分数（原始点积，未缩放）：queries @ keys 的转置。
        # queries: (num_tokens, d_out) @ keys.T: (d_out, num_tokens)
        # -> attn_scores: (num_tokens, num_tokens)，即每个 query token
        # 对每个 key token 的“相关性”打分（也叫 omega，源自论文中的符号）。
        attn_scores = queries @ keys.T # omega
        # 缩放点积注意力：除以 sqrt(d_k)（这里用 keys 最后一维大小的平方根）
        # 防止点积数值过大导致 softmax 进入饱和区（梯度消失），
        # 再沿最后一维（每行，即每个 query 对所有 key）做 softmax 归一化，
        # 得到注意力权重（每行和为 1）。
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        # 用注意力权重对 Value 做加权求和，得到每个 token 的上下文向量。
        # attn_weights: (num_tokens, num_tokens) @ values: (num_tokens, d_out)
        # -> context_vec: (num_tokens, d_out)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """自注意力实现（版本 2）：用 nn.Linear 替代手写权重矩阵。

    与 SelfAttention_v1 数学上等价，但使用 PyTorch 标准的 `nn.Linear` 层
    来实现 Q/K/V 投影，代码更简洁、更符合工程实践，并支持可选的偏置项
    （qkv_bias）。这是更贴近真实项目/论文实现的写法。

    参数：
        d_in (int): 输入 token 向量维度。
        d_out (int): Q/K/V 及输出 context 向量的维度。
        qkv_bias (bool): 是否为 Q/K/V 的线性层添加偏置项，默认 False。
    """

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # 用三个独立的线性层分别实现 Q/K/V 投影，权重由 nn.Linear 内部
        # 自动初始化和管理（等价于 v1 中的 W_query/W_key/W_value，
        # 但写法更规范，且可选择是否带 bias）。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """前向传播：与 v1 逻辑一致，仅 Q/K/V 计算方式不同。

        参数：
            x (Tensor): 输入序列，形状 (num_tokens, d_in)。

        返回：
            Tensor: 上下文向量，形状 (num_tokens, d_out)。
        """
        # 通过线性层得到 Key / Query / Value，形状均为 (num_tokens, d_out)。
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算注意力分数：(num_tokens, d_out) @ (d_out, num_tokens)
        # -> (num_tokens, num_tokens)
        attn_scores = queries @ keys.T
        # 缩放点积 + softmax，得到归一化的注意力权重。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # 注意力权重对 Value 加权求和，得到上下文向量。
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    """带因果掩码（causal mask）的自注意力，支持批次维度和 Dropout。

    在 SelfAttention_v2 的基础上新增：
        1. 批次（batch）维度支持：输入形状变为 (b, num_tokens, d_in)。
        2. 因果掩码：保证位置 i 的输出只依赖于位置 <= i 的输入（防止模型
           在训练时“偷看”未来的 token，这是自回归语言模型的核心约束）。
        3. 注意力权重上的 Dropout：训练时随机丢弃部分注意力权重，缓解过拟合。

    参数：
        d_in (int): 输入向量维度。
        d_out (int): Q/K/V 及输出向量维度。
        context_length (int): 支持的最大序列长度（用于预先构造掩码矩阵）。
        dropout (float): 应用于注意力权重的 Dropout 概率。
        qkv_bias (bool): 是否为 Q/K/V 线性层添加偏置，默认 False。
    """

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # New
        # 预先构造一个上三角掩码矩阵（对角线以上为 1，其余为 0），
        # 形状为 (context_length, context_length)。
        # torch.triu(..., diagonal=1) 表示主对角线上方（不含对角线）的元素为 1，
        # 这些位置对应“未来”的 token，之后会在 forward 中把这些位置的注意力分数
        # 设为 -inf，从而屏蔽掉“看到未来”的信息。
        # 用 register_buffer 注册为缓冲区（而非参数），使其能随 .to(device)
        # 一起移动设备，但不会被优化器更新、也不会被视为模型参数。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        """前向传播：计算带因果掩码的自注意力输出。

        参数：
            x (Tensor): 输入序列，形状 (b, num_tokens, d_in)，
                        其中 b 为批次大小（batch size）。

        返回：
            Tensor: 上下文向量，形状 (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape  # New batch dimension b
        # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
        # in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
        # do not exceed `context_length` before reaching this forward method.
        # 通过线性层分别计算 Key / Query / Value，
        # 形状均为 (b, num_tokens, d_out)。
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算注意力分数：对最后两维做批量矩阵乘法（batched matmul）。
        # queries: (b, num_tokens, d_out) @ keys.transpose(1,2): (b, d_out, num_tokens)
        # -> attn_scores: (b, num_tokens, num_tokens)
        # 注意这里用 transpose(1, 2) 而不是像单样本版本那样用 .T，
        # 是因为多了 batch 维度，需要保留第 0 维（batch）不变，
        # 只转置第 1、2 维（序列长度维 与 d_out 维）。
        attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose
        # 应用因果掩码：把 mask 中值为 1（即“未来”位置）对应的注意力分数
        # 原地（in-place，函数名以下划线结尾）填充为 -inf。
        # 截取 [:num_tokens, :num_tokens] 是为了兼容当前序列长度小于
        # context_length 的情况（例如推理时序列还没有填满整个上下文窗口）。
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        # 缩放点积 + softmax：-inf 位置经过 softmax 后权重趋近于 0，
        # 从而实现“看不到未来 token”的效果。
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        # 对注意力权重做 Dropout（仅在训练模式下生效），随机丢弃部分连接，
        # 增强模型的泛化能力，缓解过拟合。
        attn_weights = self.dropout(attn_weights)  # New

        # 注意力权重与 Value 做加权求和，得到上下文向量。
        # attn_weights: (b, num_tokens, num_tokens) @ values: (b, num_tokens, d_out)
        # -> context_vec: (b, num_tokens, d_out)
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """多头注意力的“简单堆叠”实现：并行运行多个独立的 CausalAttention 头。

    每个头都是一个完整、独立的 CausalAttention 实例（各自拥有独立的
    W_query/W_key/W_value 参数），多个头并行计算后，将各自输出的
    context 向量沿最后一维拼接起来，得到最终的多头输出。

    优点：实现简单直观，容易理解“多头”的概念。
    缺点：效率较低——每个头都要对输入 x 重新做一次完整的线性变换，
          计算存在冗余（相比之下 MultiHeadAttention 类用一次大的线性层
          实现所有头的投影，效率更高）。

    参数：
        d_in (int): 输入向量维度。
        d_out (int): 每个注意力头输出的向量维度（注意：这里是“单头”的
                     输出维度，最终拼接后总维度为 d_out * num_heads）。
        context_length (int): 支持的最大序列长度。
        dropout (float): 注意力权重的 Dropout 概率。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool): 是否为 Q/K/V 线性层添加偏置，默认 False。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 用 ModuleList 存放 num_heads 个独立的 CausalAttention 子模块，
        # 每个头都各自维护一套 Q/K/V 参数和因果掩码。
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        """前向传播：并行运行所有头，再沿最后一维拼接输出。

        参数：
            x (Tensor): 输入序列，形状 (b, num_tokens, d_in)。

        返回：
            Tensor: 拼接后的多头输出，形状 (b, num_tokens, d_out * num_heads)。
        """
        # 依次（逻辑上是并行）调用每个头对同一份输入 x 计算注意力，
        # 每个头输出形状为 (b, num_tokens, d_out)；
        # 再沿最后一维（特征维）拼接所有头的输出，
        # 最终形状为 (b, num_tokens, d_out * num_heads)。
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """高效的多头注意力实现：一次线性投影 + 张量 reshape 拆分多头。

    与 MultiHeadAttentionWrapper 不同，本实现只用一组较大的
    W_query/W_key/W_value 线性层（输出维度为总的 d_out）对输入做一次投影，
    然后通过 view/reshape 将最后一维拆分为 (num_heads, head_dim)，
    并把 num_heads 提到 batch 维之后，从而让多个头可以在一次批量矩阵乘法
    中并行计算注意力，避免了重复的线性变换，计算效率更高。
    这也是实际工业界 Transformer 实现（如 GPT 系列）常用的写法。

    参数：
        d_in (int): 输入向量维度。
        d_out (int): 多头注意力的总输出维度（会被均分给每个头，
                     必须能被 num_heads 整除）。
        context_length (int): 支持的最大序列长度，用于构造因果掩码。
        dropout (float): 注意力权重的 Dropout 概率。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool): 是否为 Q/K/V 线性层添加偏置，默认 False。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # d_out 必须能被 num_heads 整除，才能均匀拆分为多个头，
        # 每个头的维度为 head_dim = d_out // num_heads。
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        # 用一组线性层一次性完成“所有头”的 Q/K/V 投影，
        # 输出维度为总的 d_out（后续再拆分成 num_heads 个 head_dim）。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 输出投影层：将拼接后的多头输出再做一次线性变换，
        # 融合各个头学到的信息（常见于 Transformer 的多头注意力设计）。
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        # 与 CausalAttention 相同，预先构造上三角因果掩码矩阵，
        # 注册为 buffer（非可训练参数，但会随模型一起搬到 GPU/CPU）。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """前向传播：一次投影 -> 拆分多头 -> 并行计算注意力 -> 合并多头。

        参数：
            x (Tensor): 输入序列，形状 (b, num_tokens, d_in)。

        返回：
            Tensor: 多头注意力输出，形状 (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape

        # 一次性对全部头做线性投影，得到完整维度 d_out 的 Q/K/V。
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 通过 view 将最后一维 d_out “隐式拆分”为 (num_heads, head_dim) 两维，
        # 也就是把一次大的线性投影结果，按头切分成多组更小的向量，
        # 每一组分别对应一个注意力头要处理的子空间。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 交换 num_tokens 和 num_heads 两个维度，使得 num_heads 紧跟在 batch
        # 维度之后。这样一来，后续的矩阵乘法（在最后两维 num_tokens 与
        # head_dim 上进行）会自动对 (b, num_heads) 这两个维度做批量处理，
        # 相当于对每个 batch 中的每个头分别、并行地计算注意力。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 计算每个头内部的注意力分数：
        # queries: (b, num_heads, num_tokens, head_dim)
        # @ keys.transpose(2,3): (b, num_heads, head_dim, num_tokens)
        # -> attn_scores: (b, num_heads, num_tokens, num_tokens)
        # 即对每个 batch、每个头，都得到一个 (num_tokens, num_tokens) 的
        # 注意力分数矩阵。
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        # 截取掩码到当前序列长度，并转换为布尔类型，
        # True 表示对应位置需要被屏蔽（即“未来”位置）。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 利用广播机制，将 mask_bool 应用到形状为
        # (b, num_heads, num_tokens, num_tokens) 的 attn_scores 上，
        # 把未来位置的分数原地填充为 -inf，实现因果掩码（对所有头、
        # 所有 batch 样本同时生效）。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放点积 + softmax：注意这里除以 sqrt(head_dim)（单个头的维度），
        # 而不是总维度 d_out，这是多头注意力缩放因子的标准做法。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 对注意力权重做 Dropout。
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 用注意力权重对 Value 加权求和：
        # attn_weights: (b, num_heads, num_tokens, num_tokens)
        # @ values: (b, num_heads, num_tokens, head_dim)
        # -> (b, num_heads, num_tokens, head_dim)
        # 再 transpose(1, 2) 把 num_heads 换回到 num_tokens 之后，
        # 得到形状 (b, num_tokens, num_heads, head_dim)，
        # 方便下一步把各头结果在最后两维上合并（拼接回 d_out）。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 将最后两维 (num_heads, head_dim) 合并（reshape）为一维 d_out，
        # 即把各个头的输出拼接在一起，恢复到与输入相同量级的特征维度。
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        # 最后经过一个输出投影线性层，融合各头信息，得到最终的多头注意力输出。
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


######################
# Bonus
######################


class PyTorchMultiHeadAttention(nn.Module):
    """使用 PyTorch 内置高效算子实现的多头注意力（Bonus，性能优化版本）。

    与手写版 MultiHeadAttention 逻辑等价，但有两点关键差异：
        1. 用一个线性层 `self.qkv` 一次性同时算出 Q、K、V 三者
           （输出维度为 3 * d_out），进一步减少线性层调用次数。
        2. 核心的“缩放点积注意力 + 因果掩码 + Dropout”计算，
           直接调用 PyTorch 提供的高度优化算子
           `nn.functional.scaled_dot_product_attention`（可自动利用
           FlashAttention 等底层加速实现），无需手动构造掩码矩阵、
           手动做 softmax 和 masked_fill，速度和显存效率通常更优。

    参数：
        d_in (int): 输入向量维度。
        d_out (int): 多头注意力总输出维度（需能被 num_heads 整除）。
        num_heads (int): 注意力头数量。
        dropout (float): 注意力权重的 Dropout 概率，默认 0.0。
        qkv_bias (bool): 是否为合并的 qkv 线性层添加偏置，默认 False。
    """
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        # 用一个线性层同时算出 Q、K、V（输出维度为 3 * d_out），
        # 之后再切分成三份，相当于把三次矩阵乘法合并为一次，效率更高。
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        # 多头输出合并后的投影层。
        self.proj = nn.Linear(d_out, d_out)
        # 这里只保存 dropout 概率（float），实际调用在 forward 中
        # 直接作为参数传给 scaled_dot_product_attention。
        self.dropout = dropout

    def forward(self, x):
        """前向传播：合并 QKV 投影 -> 调用高效注意力算子 -> 合并多头输出。

        参数：
            x (Tensor): 输入序列，形状 (batch_size, num_tokens, embed_dim)。

        返回：
            Tensor: 多头注意力输出，形状 (batch_size, num_tokens, d_out)。
        """
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        # 一次线性变换同时得到 Q、K、V 拼接后的结果。
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        # 把最后一维 3 * embed_dim 拆分为 (3, num_heads, head_dim)，
        # 其中 3 分别对应 Q、K、V。
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        # 调整维度顺序，把代表 Q/K/V 的维度放到最前面，方便下一步直接
        # 解包成三个独立张量；同时把 num_heads 放到 batch 维之后，
        # 便于对每个头做批量并行的注意力计算。
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        # 沿第 0 维解包，得到形状均为 (b, num_heads, num_tokens, head_dim)
        # 的 queries、keys、values 三个张量。
        queries, keys, values = qkv

        # 仅在训练模式下才启用 Dropout，评估/推理模式下 dropout 概率设为 0。
        use_dropout = 0. if not self.training else self.dropout

        # 调用 PyTorch 内置的缩放点积注意力算子：
        # 该算子内部自动完成 Q@K^T 缩放、因果掩码（is_causal=True 时自动
        # 生成下三角掩码，效果等价于前面手写版本中的 mask 操作）、
        # softmax、dropout、以及与 V 的加权求和，并且在支持的硬件上
        # 会自动使用 FlashAttention 等高效实现，速度更快、显存占用更低。
        # 输出形状与输入的 queries 相同：(b, num_heads, num_tokens, head_dim)。
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 将 num_heads 维度换回到 num_tokens 之前（transpose(1, 2)），
        # 得到 (b, num_tokens, num_heads, head_dim)；
        # .contiguous() 确保内存连续，以便安全调用 .view()；
        # 最后 reshape 为 (b, num_tokens, d_out)，即把各头输出拼接合并。
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        # 输出投影层，融合各头信息，得到最终多头注意力输出。
        context_vec = self.proj(context_vec)

        return context_vec
