# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

# ============================================================================
# 中文模块说明：
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
# 一书第 5 章的"前置代码汇总文件"，把第 2~4 章中已经讲解并实现过的核心组件
# 原样搬到这里复用，方便第 5 章（预训练 GPT 模型）直接调用，而不用重复粘贴：
#   - 第 2 章：文本 -> token 的数据集与 DataLoader 构建（滑动窗口采样）
#   - 第 3 章：多头自注意力机制（Multi-Head Self-Attention），含因果掩码
#   - 第 4 章：LayerNorm、GELU 激活、前馈网络、Transformer Block、完整 GPTModel，
#              以及最简单的贪心解码文本生成函数 generate_text_simple
# 该文件也可以作为独立脚本运行：会构建一个 124M 参数规模的 GPT 配置，
# 用随机初始化的权重对一段起始文本做贪心生成，验证模型前向/生成流程是否走通。
# ============================================================================

import tiktoken  # OpenAI 开源的 BPE 分词器库，这里用它的 gpt2 编码方案做文本<->token 转换
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################
# 第 2 章：构建训练用的数据集（滑动窗口切分 token 序列）


class GPTDatasetV1(Dataset):
    """
    自定义 PyTorch Dataset：将一整段原始文本 token 化后，
    用固定窗口大小 max_length、步长 stride 做滑动窗口切分，
    生成用于"预测下一个 token"任务的 (输入序列, 目标序列) 样本对。

    参数：
        txt (str): 原始训练文本（例如一整本书的内容）。
        tokenizer: 分词器对象，需支持 .encode() 方法（这里通常传入 tiktoken 的 gpt2 编码器）。
        max_length (int): 每个训练样本的 token 序列长度（即上下文窗口大小）。
        stride (int): 相邻两个样本起始位置之间的步长；stride < max_length 时样本间会有重叠。

    每个样本：
        input_chunk:  长度为 max_length 的 token 序列，形状 (max_length,)
        target_chunk: 相对 input_chunk 整体右移一位的 token 序列，形状 (max_length,)
                       即 target_chunk[i] = input_chunk[i+1]，用于自回归语言模型的
                       "根据前 i 个 token 预测第 i+1 个 token" 训练目标。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 中文：先把整段文本一次性编码为 token id 列表；
        # allowed_special={"<|endoftext|>"} 允许文本中出现的 <|endoftext|> 特殊标记被正常编码
        # （而不是被当作普通文本报错或误处理），常用于拼接多篇文档时的分隔符。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 中文：用滑动窗口把整本书的 token 序列切成多个长度为 max_length、可能相互重叠的子序列。
        # 循环范围 range(0, len(token_ids) - max_length, stride)：
        #   - 保证每次切片 [i : i+max_length] 和 [i+1 : i+max_length+1] 都不会越界；
        #   - stride 越小，样本重叠越多，数据利用率越高但样本间相关性也越强。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列整体比输入序列右移 1 位
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 中文：数据集大小 = 切出来的样本（窗口）数量
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 中文：按索引返回一对 (输入 token 序列, 目标 token 序列)，均为 1D LongTensor，形状 (max_length,)
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    便捷函数：给定原始文本，一步完成 分词器初始化 -> Dataset 构建 -> DataLoader 封装。

    参数：
        txt (str): 原始训练文本。
        batch_size (int): 每个 mini-batch 的样本数。
        max_length (int): 每个样本的上下文长度（token 数）。
        stride (int): 滑动窗口步长，控制样本重叠程度。
        shuffle (bool): 是否在每个 epoch 打乱样本顺序。
        drop_last (bool): 是否丢弃最后一个不满 batch_size 的不完整批次
                          （训练时常设为 True，避免最后一个 batch 过小导致梯度不稳定）。
        num_workers (int): 数据加载的子进程数量。

    返回：
        torch.utils.data.DataLoader：迭代时每次产出
            (input_batch, target_batch)，形状均为 (batch_size, max_length)。
    """
    # Initialize the tokenizer
    # 中文：使用 GPT-2 的 BPE 分词方案（词表大小 50257）
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # 中文：DataLoader 负责按 batch_size 打包样本、是否打乱顺序、多进程并行读取等
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
# 第 3 章：多头因果自注意力机制（Multi-Head Causal Self-Attention）
class MultiHeadAttention(nn.Module):
    """
    多头自注意力模块，是 Transformer/GPT 的核心组件。
    通过"分头并行计算缩放点积注意力 + 因果掩码"实现：
        1) 每个位置只能看到自己及之前的位置（自回归特性，靠上三角掩码实现）；
        2) 将 Q/K/V 拆分成多个头并行计算，增强模型捕捉不同子空间信息的能力。

    参数：
        d_in (int): 输入特征维度。
        d_out (int): 输出特征维度（同时也是 Q/K/V 投影后的总维度）。
        context_length (int): 支持的最大序列长度，用于预先构建因果掩码矩阵。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头数，要求 d_out 能被 num_heads 整除。
        qkv_bias (bool): Q/K/V 线性层是否使用偏置项。

    输入形状：x -> (batch_size, num_tokens, d_in)
    输出形状：-> (batch_size, num_tokens, d_out)
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：每个注意力头分到的维度 = 总输出维度 / 头数，
        # 这样多头拼接回去正好还原成 d_out，不增加总参数量/计算量的量级。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        # 中文：register_buffer 注册的 mask 不是可训练参数，但会随模型一起 to(device)/保存加载；
        # torch.triu(..., diagonal=1) 生成一个严格上三角为 1、其余为 0 的矩阵，
        # 后面会用它标记"未来位置"（当前 token 看不到的位置），实现因果（causal）注意力。
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # b: batch size, num_tokens: 序列长度, d_in: 输入维度

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)
        # 中文：分别用三个独立的线性层把输入 x 投影为 Query、Key、Value，
        # 三者形状均为 (b, num_tokens, d_out)。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文：把最后一维 d_out 拆分成 (num_heads, head_dim)，
        # 相当于"隐式地"把一个大投影矩阵切成 num_heads 份，为并行多头计算做准备。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文：把 num_heads 维度换到 batch 之后、num_tokens 之前，
        # 这样后续矩阵乘法可以把 (b, num_heads) 当作"批量维度"，对每个头独立并行做注意力计算。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 中文：Q @ K^T，对每个头分别计算"查询-键"相似度得分（未缩放前的注意力分数）。
        # queries: (b, num_heads, num_tokens, head_dim)
        # keys.transpose(2,3): (b, num_heads, head_dim, num_tokens)
        # attn_scores: (b, num_heads, num_tokens, num_tokens)  —— 每个 (i, j) 表示第 i 个 token 对第 j 个 token 的注意力原始得分
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        # 中文：因为 mask 是按 context_length 预先构建好的最大尺寸矩阵，
        # 这里裁剪到当前实际序列长度 num_tokens，并转成 bool 类型方便 masked_fill_ 使用。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 中文：把"未来位置"（上三角部分，mask_bool 为 True 处）的注意力分数填为 -inf，
        # 这样经过 softmax 后这些位置的权重会趋近于 0，从而保证每个位置只能"看到"自己及之前的 token
        # ——这就是自回归语言模型必须的因果（causal）掩码。
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文：对注意力分数做缩放（除以 sqrt(head_dim)，即 keys.shape[-1] 的平方根），
        # 这是"缩放点积注意力"(Scaled Dot-Product Attention) 中的关键一步，
        # 目的是防止点积结果随维度增大而过大，导致 softmax 梯度消失。
        # 最后在最后一维（key/token 维）上做 softmax，得到归一化的注意力权重。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 中文：对注意力权重做 dropout 正则化，缓解过拟合

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文：用注意力权重对 Value 加权求和，得到每个头的上下文向量；
        # attn_weights: (b, num_heads, num_tokens, num_tokens)
        # values:       (b, num_heads, num_tokens, head_dim)
        # 相乘结果:      (b, num_heads, num_tokens, head_dim)
        # 再 transpose(1,2) 把 num_heads 换回中间维度，方便下一步合并多头。
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多头结果重新拼接（reshape）回 (b, num_tokens, d_out)，
        # 相当于把每个 token 在各个头上算出的上下文向量首尾相连拼在一起。
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：最后再经过一个输出投影层，让多头信息进一步融合（可选但常用的做法）。

        return context_vec


#####################################
# Chapter 4
#####################################
# 第 4 章：LayerNorm、GELU、前馈网络、Transformer Block、完整 GPT 模型
class LayerNorm(nn.Module):
    """
    自定义层归一化（Layer Normalization）模块。
    对每个样本、每个 token 在特征维（最后一维）上做归一化，
    使其均值为 0、方差为 1，再通过可学习的 scale（缩放）和 shift（平移）参数
    恢复模型需要的表达能力。有助于稳定深层网络的训练。

    参数：
        emb_dim (int): 特征（嵌入）维度大小，即归一化作用的最后一维长度。

    输入/输出形状：(..., emb_dim) -> (..., emb_dim)，形状不变。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止除以 0 的极小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习缩放参数，初始化为全 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习平移参数，初始化为全 0

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 中文：在最后一维（特征维）上求均值，保留维度便于广播
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 中文：unbiased=False 表示使用有偏方差估计（除以 N 而不是 N-1），
        # 与常见深度学习框架中 LayerNorm 的实现保持一致。
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 中文：标准化，得到均值0方差1的归一化结果
        return self.scale * norm_x + self.shift  # 中文：仿射变换，恢复模型所需的表达能力


class GELU(nn.Module):
    """
    GELU（Gaussian Error Linear Unit）激活函数的近似实现（tanh 近似版本）。
    相比 ReLU，GELU 在 0 附近更平滑，常用于 Transformer 类模型的前馈网络中。

    输入/输出形状保持不变，逐元素计算。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 中文：GELU 的 tanh 近似公式：
        # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        # 这是原始 GELU 论文中给出的高精度近似式，避免直接计算高斯误差函数 erf 带来的开销。
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Transformer Block 中的逐位置前馈网络（Position-wise Feed-Forward Network）。
    结构为：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)，
    先升维再降维，增强模型的非线性表达能力（4 倍扩展是原始 Transformer/GPT 的常见设计）。

    参数：
        cfg (dict): 模型配置字典，需包含 "emb_dim" 键。

    输入/输出形状：(batch_size, num_tokens, emb_dim) -> 形状不变。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 升维：emb_dim -> 4*emb_dim
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 降维：4*emb_dim -> emb_dim
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    一个完整的 Transformer 解码器层（GPT 风格，Pre-LayerNorm 结构）：
        x -> LN -> 多头自注意力 -> Dropout -> 残差相加
          -> LN -> 前馈网络     -> Dropout -> 残差相加

    参数：
        cfg (dict): 模型配置字典，需包含 emb_dim / context_length / n_heads /
                    drop_rate / qkv_bias 等键。

    输入/输出形状：(batch_size, num_tokens, emb_dim) -> 形状不变，可多层堆叠。
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
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        # 中文：先保存输入，用于后面的残差连接（shortcut/skip connection），
        # 残差连接能有效缓解深层网络的梯度消失问题，是训练深层 Transformer 的关键设计。
        shortcut = x
        x = self.norm1(x)  # 中文：Pre-LN 结构——先归一化，再进入子层（比 Post-LN 训练更稳定）
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back  中文：残差相加，恢复原始信息通路

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back  中文：第二次残差相加

        return x


class GPTModel(nn.Module):
    """
    完整的 GPT 风格自回归语言模型。
    结构：token 嵌入 + 位置嵌入 -> Dropout -> N 层 TransformerBlock 堆叠
          -> 最终 LayerNorm -> 线性输出头（映射到词表大小，得到 logits）。

    参数：
        cfg (dict): 模型配置字典，需包含：
            vocab_size (int): 词表大小
            context_length (int): 支持的最大上下文长度（也是位置嵌入表大小）
            emb_dim (int): 嵌入/隐藏维度
            n_heads (int): 注意力头数
            n_layers (int): Transformer 层数
            drop_rate (float): dropout 概率
            qkv_bias (bool): 注意力中 Q/K/V 线性层是否使用偏置

    输入：in_idx，形状 (batch_size, seq_len)，元素为 token id（LongTensor）
    输出：logits，形状 (batch_size, seq_len, vocab_size)，每个位置对下一个 token 的预测分布（未归一化）
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])       # 中文：词元（token）嵌入表
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 中文：可学习的绝对位置嵌入表
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # 中文：堆叠 n_layers 个相同结构（但参数各自独立）的 TransformerBlock

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 中文：输出前的最终归一化层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 中文：输出投影头，把隐藏状态映射到词表维度上的 logits；不使用偏置项（GPT-2 的常见做法）

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 中文：(batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 中文：位置嵌入只依赖序列长度 seq_len，与 batch 无关；
        # torch.arange(seq_len) 生成 [0, 1, ..., seq_len-1] 作为位置索引，
        # device=in_idx.device 保证位置索引和输入张量在同一设备（CPU/GPU）上。
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：pos_embeds 形状 (seq_len, emb_dim) 会被广播加到 (batch_size, seq_len, emb_dim) 上，
        # 得到"token 语义信息 + 位置信息"融合后的输入表示。
        x = self.drop_emb(x)
        x = self.trf_blocks(x)     # 中文：依次通过所有 Transformer 层
        x = self.final_norm(x)
        logits = self.out_head(x)  # 中文：(batch_size, seq_len, emb_dim) -> (batch_size, seq_len, vocab_size)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    最简单的自回归文本生成函数：贪心解码（greedy decoding）。
    每一步都取模型输出概率最大（logits 最大）的 token 作为下一个 token，
    不做采样、不做 top-k/top-p 等多样性处理，是理解生成流程的最小实现。

    参数：
        model: 训练好的（或用于演示的）GPTModel 实例。
        idx (Tensor): 初始的 token id 序列，形状 (batch_size, num_tokens)。
        max_new_tokens (int): 需要新生成的 token 数量。
        context_size (int): 模型支持的最大上下文长度，用于裁剪过长的输入序列。

    返回：
        Tensor: 拼接了新生成 token 后的完整序列，
                形状 (batch_size, num_tokens + max_new_tokens)。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):  # 中文：循环生成，每次只产出一个新 token（自回归特性）

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 中文：若当前序列长度超过模型支持的最大上下文长度 context_size，
        # 只保留最近的 context_size 个 token 作为输入（位置嵌入表大小是固定的，不能越界）。
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # 中文：生成阶段不需要计算梯度，用 torch.no_grad() 节省显存/加速推理
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 中文：语言模型在每个位置都会输出"下一个 token"的预测，
        # 但生成新 token 时只需要关注序列最后一个位置的预测结果。
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 中文：贪心解码——直接取概率（logits）最大的词表索引作为下一个 token，
        # keepdim=True 保持结果为 (batch, 1) 形状，方便后续拼接。
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 中文：把新生成的 token 拼接到序列末尾，作为下一轮生成的输入上下文
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


if __name__ == "__main__":
    # 中文：以下为脚本独立运行时的示例代码——
    # 构建一个 GPT-2 small（124M 参数量级）规模的配置，随机初始化模型权重，
    # 对给定起始文本做贪心生成，验证整个前向传播 + 生成流程能够跑通。
    # 注意：由于权重是随机初始化的（未经训练），生成的文本内容不具有语义意义。

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size  中文：词表大小，对应 GPT-2 的 BPE 词表
        "context_length": 1024,  # Context length  中文：最大上下文长度
        "emb_dim": 768,          # Embedding dimension  中文：嵌入/隐藏层维度
        "n_heads": 12,           # Number of attention heads  中文：注意力头数
        "n_layers": 12,          # Number of layers  中文：Transformer 层数
        "drop_rate": 0.1,        # Dropout rate  中文：dropout 概率
        "qkv_bias": False        # Query-Key-Value bias  中文：Q/K/V 线性层是否带偏置
    }

    torch.manual_seed(123)  # 中文：固定随机种子，保证权重初始化和结果可复现
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout  中文：切换为评估模式，关闭 dropout，保证生成结果确定性

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)  # 中文：将起始文本编码为 token id 列表
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # 中文：unsqueeze(0) 在最前面增加一个 batch 维度，形状从 (num_tokens,) 变为 (1, num_tokens)，
    # 因为模型的输入约定是 (batch_size, seq_len)。

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
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    # 中文：squeeze(0) 去掉 batch 维度，.tolist() 转成 python list，再用分词器解码回可读文本。

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)
