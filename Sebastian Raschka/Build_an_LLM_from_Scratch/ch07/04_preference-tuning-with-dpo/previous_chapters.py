# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-6.
# This file can be run as a standalone script.

"""
本文件是第 2~6 章代码的汇总（"previous_chapters.py"），供第 7 章
「04_preference-tuning-with-dpo」（DPO 偏好微调）小节复用，避免在每个
子章节里重复粘贴之前写过的代码。

内容概览：
- 第 2 章：GPT 数据集 / DataLoader（滑动窗口切分文本为「输入-目标」对）。
- 第 3 章：多头自注意力（MultiHeadAttention），含因果掩码（causal mask）。
- 第 4 章：LayerNorm、GELU 激活、前馈网络（FeedForward）、
  Transformer Block、完整的 GPTModel。
- 第 5 章：文本生成函数（贪心解码 generate_text_simple、
  带温度采样 + top-k 的 generate）、训练循环 train_model_simple、
  损失计算、OpenAI 预训练权重加载 load_weights_into_gpt、
  以及画训练曲线的 plot_losses。

本文件可以作为独立脚本运行（虽然这里主要是被其他脚本 import 使用）。

注释说明（中文注释小节，非原作者内容）：
- 本次仅在原代码基础上新增中文注释（模块级 docstring、类/函数 docstring、
  关键行内注释），不改变原有逻辑，除非明确标注为「bug 修复」。
- 发现的确定性 bug 已修复，并在修复处以「原代码 / 为什么是bug」的形式
  注明；无法确定或涉及跨版本/环境差异的风险点，仅标注，不修改。
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
    第 2 章：GPT 预训练用的数据集类。

    用「滑动窗口」的方式把一整段长文本切成若干个长度为 max_length 的
    (输入片段, 目标片段) 样本对，目标片段是输入片段整体右移一位
    （即经典的「预测下一个 token」自回归训练目标）。

    参数：
        txt (str): 原始的完整文本（例如一整本书的内容）。
        tokenizer: 具备 .encode()/.decode() 接口的分词器（此处用 tiktoken 的 gpt2 编码）。
        max_length (int): 每个训练样本的 token 序列长度（等价于模型的上下文窗口大小）。
        stride (int): 滑动窗口每次移动的步长；stride < max_length 时窗口之间有重叠。

    属性：
        self.input_ids (List[torch.Tensor]): 每个元素形状为 (max_length,) 的输入 token id 序列。
        self.target_ids (List[torch.Tensor]): 每个元素形状为 (max_length,) 的目标 token id 序列
            （相对 input_ids 整体右移一位）。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 对整段文本一次性编码为 token id 列表；允许特殊 token "<|endoftext|>" 出现在文本中
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 用滑动窗口把 token 序列切分成多个长度为 max_length、可能相互重叠的片段
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            # 目标序列整体比输入序列右移一个 token，即“预测下一个词”的监督信号
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集中样本（输入-目标片段对）的总数。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        按索引取出一个训练样本。

        参数：
            idx (int): 样本索引。
        返回：
            (input_ids, target_ids): 一对形状均为 (max_length,) 的 LongTensor。
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    第 2 章：创建用于 GPT 预训练的 DataLoader。

    内部固定使用 tiktoken 的 "gpt2" 编码器对文本分词，然后用
    GPTDatasetV1 做滑动窗口切分，最后包装成标准的 PyTorch DataLoader。

    参数：
        txt (str): 原始文本。
        batch_size (int): 每个 batch 的样本数。
        max_length (int): 每个样本的序列长度（模型上下文窗口大小）。
        stride (int): 滑动窗口步长。
        shuffle (bool): 是否打乱样本顺序。
        drop_last (bool): 若最后一个 batch 样本数不足 batch_size 是否丢弃
            （训练时通常设为 True，避免最后一个不完整 batch 导致 loss 波动）。
        num_workers (int): 数据加载使用的子进程数。

    返回：
        torch.utils.data.DataLoader: 产出 (input_batch, target_batch) 的迭代器，
            两者形状均为 (batch_size, max_length)。
    """
    # Initialize the tokenizer
    # 初始化 GPT-2 使用的 BPE 分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    # 创建数据集（内部完成分词 + 滑动窗口切分）
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # 用标准 DataLoader 包装数据集，得到可迭代的批次数据
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """
    第 3 章：多头因果自注意力（Multi-Head Causal Self-Attention）模块。

    将输入投影到 Q/K/V 三组向量，拆分成多个头分别计算带因果掩码的
    缩放点积注意力，再把各头的输出拼接、投影回 d_out 维。

    参数：
        d_in (int): 输入特征维度。
        d_out (int): 输出特征维度（也是 Q/K/V 投影后的总维度）。
        context_length (int): 支持的最大序列长度，用于预先构造因果掩码矩阵。
        dropout (float): 注意力权重上使用的 dropout 概率。
        num_heads (int): 注意力头数，要求 d_out 能被 num_heads 整除。
        qkv_bias (bool): Q/K/V 线性层是否使用偏置项。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 每个头的维度 = 总输出维度 / 头数

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # 用于把多头拼接后的结果再做一次线性变换（合并各头信息）
        self.dropout = nn.Dropout(dropout)
        # 预先构造上三角掩码（对角线以上为 1），用于实现“看不到未来 token”的因果注意力
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        前向传播。

        参数：
            x (torch.Tensor): 形状 (b, num_tokens, d_in)，b 为 batch size，
                num_tokens 为当前序列长度。

        返回：
            torch.Tensor: 形状 (b, num_tokens, d_out) 的上下文向量。
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 把最后一维 d_out 拆分成 (num_heads, head_dim)，相当于隐式地切分成多个头
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 交换维度，使 num_heads 排到 batch 维之后，便于对每个头独立做矩阵乘法
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # attn_scores 形状: (b, num_heads, num_tokens, num_tokens)

        # Original mask truncated to the number of tokens and converted to boolean
        # 将掩码截取到当前实际序列长度，并转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 把掩码中为 True（即“未来位置”）的注意力分数填充为负无穷，
        # 这样 softmax 后这些位置的权重会趋近于 0，实现因果（只看过去）注意力
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 缩放点积注意力：除以 sqrt(head_dim) 防止数值过大导致 softmax 梯度消失
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 用注意力权重对 values 加权求和，再把 num_heads 维换回到 num_tokens 之前
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 把多头的输出重新拼接（reshape）回单一的 d_out 维
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 对拼接后的多头结果再做一次线性投影，融合各头信息

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    第 4 章：层归一化（Layer Normalization）模块。

    对最后一维（特征维）做归一化，再用可学习的缩放（scale）和平移（shift）
    参数进行仿射变换，用于稳定深层网络的训练。

    参数：
        emb_dim (int): 特征（嵌入）维度大小，即归一化作用的最后一维长度。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除以 0 的极小值
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习缩放参数，初始为 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习平移参数，初始为 0

    def forward(self, x):
        """
        参数：
            x (torch.Tensor): 形状 (..., emb_dim)，对最后一维做归一化。
        返回：
            torch.Tensor: 与输入同形状，归一化并仿射变换后的结果。
        """
        mean = x.mean(dim=-1, keepdim=True)
        # 使用有偏方差估计（unbiased=False，即除以 N 而非 N-1），与 GPT-2 官方实现保持一致
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    第 4 章：GELU 激活函数（GPT-2 使用的 tanh 近似版本）。

    公式：0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    这是 GELU 精确形式（基于误差函数 erf）的近似实现，计算更快。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        参数：
            x (torch.Tensor): 任意形状。
        返回：
            torch.Tensor: 与输入同形状，逐元素应用 GELU 激活后的结果。
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    第 4 章：Transformer Block 中的逐位置前馈网络（Position-wise FFN）。

    结构：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)。
    先升维再降维，是 Transformer 中常见的“扩展-压缩”设计，用于增强模型的非线性表达能力。

    参数：
        cfg (dict): 配置字典，需包含 "emb_dim" 键（嵌入维度）。
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
            x (torch.Tensor): 形状 (b, num_tokens, emb_dim)。
        返回：
            torch.Tensor: 形状 (b, num_tokens, emb_dim)，与输入相同。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    第 4 章：单个 Transformer 解码器块（Pre-LayerNorm 结构）。

    结构：
        x -> LayerNorm -> 多头自注意力 -> Dropout -> 残差相加
          -> LayerNorm -> 前馈网络     -> Dropout -> 残差相加

    参数：
        cfg (dict): 模型配置字典，需包含以下键：
            "emb_dim": 嵌入维度
            "context_length": 上下文窗口大小
            "n_heads": 注意力头数
            "drop_rate": dropout 概率
            "qkv_bias": 是否给 QKV 线性层加偏置
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
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        参数：
            x (torch.Tensor): 形状 (batch_size, num_tokens, emb_dim)。
        返回：
            torch.Tensor: 形状 (batch_size, num_tokens, emb_dim)，与输入相同。
        """
        # Shortcut connection for attention block
        # 注意力子层的残差连接：先保存输入，稍后加回去
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        # 前馈网络子层的残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """
    第 4 章：完整的 GPT 模型（仅解码器结构）。

    结构：token 嵌入 + 位置嵌入 -> Dropout -> N 层 TransformerBlock 堆叠
        -> 最终 LayerNorm -> 输出线性层（映射到词表大小，得到 logits）。

    参数：
        cfg (dict): 模型配置字典，需包含：
            "vocab_size": 词表大小
            "emb_dim": 嵌入维度
            "context_length": 支持的最大上下文长度
            "drop_rate": embedding 层 dropout 概率
            "n_layers": TransformerBlock 堆叠层数
            以及 TransformerBlock 所需的其他键（n_heads, qkv_bias 等）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 按配置堆叠 n_layers 个 TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 输出头：把最终隐藏状态映射到词表维度，得到每个位置对下一个 token 的预测分布（未归一化 logits）

    def forward(self, in_idx):
        """
        参数：
            in_idx (torch.Tensor): 形状 (batch_size, seq_len)，输入的 token id 序列（LongTensor）。
        返回：
            torch.Tensor: 形状 (batch_size, seq_len, vocab_size)，每个位置对词表中每个 token 的预测 logits。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 位置嵌入：为当前序列长度生成 [0, 1, ..., seq_len-1] 的位置索引，再查表得到位置向量
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # token 嵌入与位置嵌入相加，得到输入表示（广播：pos_embeds 无 batch 维，自动广播到每个样本）
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    第 4 章：最简单的贪心解码（greedy decoding）文本生成函数。

    每一步取当前 logits 分布中概率最高（argmax）的 token 作为下一个 token，
    不涉及采样随机性，因此对同一输入总是生成相同的结果（确定性生成）。

    参数：
        model (GPTModel): 训练好的（或正在训练的）GPT 模型。
        idx (torch.Tensor): 形状 (B, T)，当前已有的 token id 序列（B=batch size，T=当前长度）。
        max_new_tokens (int): 要新生成的 token 数量。
        context_size (int): 模型支持的最大上下文长度，用于裁剪过长的输入。

    返回：
        torch.Tensor: 形状 (B, T + max_new_tokens)，原序列拼接上新生成 token 后的结果。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 若当前序列超过模型支持的上下文长度，只保留最后 context_size 个 token 作为输入
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 只关心序列最后一个位置的预测（即“下一个 token”的分布）
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 取概率（logits）最大的词表索引作为下一个 token —— 贪心解码，无随机性
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 把新生成的 token 拼接到序列末尾，供下一轮迭代使用
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# Chapter 5
#####################################
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    第 5 章：支持温度采样（temperature scaling）与 top-k 过滤的文本生成函数。

    相比 generate_text_simple 的纯贪心解码，本函数额外支持：
    - top_k：只在概率最高的 k 个候选 token 中采样，其余置为 -inf。
    - temperature：温度缩放 logits 后再做随机采样（temperature 越大，分布越平滑/越随机；
      temperature=0 时退化为贪心解码）。
    - eos_id：遇到指定的结束符 token 时提前停止生成。

    参数：
        model (GPTModel): GPT 模型。
        idx (torch.Tensor): 形状 (batch_size, T)，初始 token 序列。
        max_new_tokens (int): 最多生成的新 token 数。
        context_size (int): 模型支持的最大上下文长度。
        temperature (float): 温度系数；<=0 时使用贪心解码（argmax），>0 时按采样分布随机采样。
        top_k (int or None): 若指定，仅保留 logits 最大的 top_k 个候选参与采样/argmax。
        eos_id (int or None): 结束符 token id；生成到该 token 时提前 break。

    返回：
        torch.Tensor: 形状 (batch_size, T + 实际生成的新 token 数)。
            注意：一旦触发 eos_id 提前停止，会跳过 “把该 token 追加进序列” 这一步，
            直接结束整个循环（详见下方 break 处的说明）。
    """

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            # 只保留 logits 最大的 top_k 个值，其余全部置为 -inf（softmax 后概率趋近 0）
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # 温度缩放：temperature < 1 会让分布更尖锐（更接近贪心），> 1 则更平滑（更随机）

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            #logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            #probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            # ------------------------------------------------------------------
            # 【bug 修复】原代码：probs = torch.log_softmax(logits, dim=-1)
            # 为什么是 bug：torch.multinomial 要求输入是「非负的、可以不必归一化的概率权重」，
            # 而 log_softmax 的输出是对数概率，取值恒 <= 0（只有概率恰好为 1 时才等于 0）。
            # 直接把负数权重传给 torch.multinomial 会在运行期必现报错：
            #   RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
            # 已用实际调用验证复现（temperature > 0 的采样分支必炸），因此判定为确定性 bug。
            # 修复方式：改回使用 torch.softmax 得到真正的概率分布（与上面注释掉的写法一致，
            # 也是原书里的写法），再交给 multinomial 采样。
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            # ------------------------------------------------------------------

            # Sample from the distribution
            # 按概率分布做随机采样（而不是像贪心解码那样总选概率最大的那个）
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            # temperature <= 0 时退化为贪心解码
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            # 注意（风险点，未修改）：这里用 `idx_next == eos_id` 做真值判断，
            # 当 batch_size > 1 时 idx_next 是形状 (batch_size, 1) 的张量，
            # 该比较会得到多元素张量，在 if 中的真值语义依赖 PyTorch 版本/元素数，
            # 只在“批大小为 1”时行为明确可靠；这是原书代码的已知简化写法（非本次引入的问题），
            # 按要求仅标注、不修改。
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    第 5 章：简单的预训练/微调训练循环。

    标准的训练流程：按 epoch 遍历训练数据，前向计算交叉熵损失、反向传播、
    更新参数；每隔 eval_freq 步在训练/验证集上评估一次损失；每个 epoch
    结束后用当前模型生成一段示例文本，直观查看训练效果。

    参数：
        model (GPTModel): 待训练的模型。
        train_loader / val_loader (DataLoader): 训练 / 验证数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device): 运行设备（cpu/cuda/mps）。
        num_epochs (int): 训练轮数。
        eval_freq (int): 每多少个 step 做一次评估。
        eval_iter (int): 评估时使用的 batch 数量（避免遍历整个验证集，加快评估速度）。
        start_context (str): 用于每个 epoch 末尾生成示例文本的起始提示词。
        tokenizer: 分词器，用于编码 start_context 及解码生成结果。

    返回：
        (train_losses, val_losses, track_tokens_seen): 三个等长的 list，
            分别记录每次评估时的训练损失、验证损失，以及累计处理过的 token 数，
            可用于后续绘制训练曲线（见 plot_losses）。
    """
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # 切换到训练模式（启用 Dropout 等训练专属行为）

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()  # 累计已处理的 token 总数（用于横轴对齐）
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        # 每个 epoch 结束后，用当前模型生成一段文本样例，直观感受训练进展
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    第 5 章：在训练集和验证集上分别计算平均损失（用于训练过程中的监控）。

    参数：
        model (GPTModel): 待评估的模型。
        train_loader / val_loader (DataLoader): 训练 / 验证数据加载器。
        device (torch.device): 运行设备。
        eval_iter (int): 每个数据集上最多评估的 batch 数（限制评估开销）。

    返回：
        (train_loss, val_loss): 两个 float，分别为训练集、验证集上的平均交叉熵损失。
    """
    model.eval()  # 切换到评估模式（关闭 Dropout 等）
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 评估结束后切回训练模式，避免影响外层训练循环
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    第 5 章：用当前模型基于给定起始文本生成一段续写，并打印出来（用于训练过程中直观检查效果）。

    参数：
        model (GPTModel): 当前模型。
        tokenizer: 分词器，用于编码起始文本 / 解码生成的 token 序列。
        device (torch.device): 运行设备。
        start_context (str): 生成用的起始提示文本。
    返回：
        None（直接打印结果，不返回值）。
    """
    model.eval()
    # 从模型的位置嵌入层权重形状读取上下文窗口大小（context_length）
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        # 这里使用最简单的贪心解码（无随机性），固定生成 50 个新 token
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
        # 把换行替换为空格，避免打印的样例文本跨越多行、不便阅读
    model.train()


def assign(left, right):
    """
    第 5 章：辅助函数，用于把 numpy 数组 right 的值赋给 nn.Parameter left（形状必须一致）。

    主要用于 load_weights_into_gpt 中，将 OpenAI 官方发布的 GPT-2 权重
    （numpy 数组）加载进本项目自己实现的 GPTModel 对应的参数张量中。

    参数：
        left (torch.nn.Parameter): 目标参数（提供期望的 shape 校验基准）。
        right (np.ndarray): 待赋值的源数据（通常来自预训练权重文件）。

    返回：
        torch.nn.Parameter: 用 right 的数值包装出的新 Parameter（与 left 同形状）。

    抛出：
        ValueError: 当 left 与 right 的形状不一致时。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    第 5 章：把 OpenAI 官方发布的 GPT-2 预训练权重（params，嵌套字典/numpy 数组结构）
    加载到本项目自定义实现的 GPTModel 实例（gpt）中。

    OpenAI 原始权重里 QKV 是合并存放在一个 c_attn 矩阵里的（沿最后一维拼接），
    这里需要手动按 3 等分切分成 Q/K/V 三部分；同时 OpenAI 的 Linear 权重存储
    习惯与 PyTorch nn.Linear 的权重存储方式是「转置」关系，所以很多地方要 .T。

    参数：
        gpt (GPTModel): 待写入权重的模型实例（结构需与 params 对应，如层数一致）。
        params (dict): 从 OpenAI GPT-2 checkpoint 解析出的嵌套字典，
            包含 "wpe"（位置嵌入）、"wte"（token 嵌入/输出头共享权重）、
            "blocks"（每层的注意力/前馈/LayerNorm 参数）、"g"/"b"（最终 LayerNorm 的缩放/平移）。

    返回：
        None（原地修改 gpt 的参数）。
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # c_attn 的权重把 Q/K/V 三部分拼接在最后一维，这里按最后一维等分为 3 份
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        # 注意 .T：OpenAI 权重是 (in, out) 布局，而 nn.Linear.weight 需要 (out, in)，故转置
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 偏置项同样按 Q/K/V 三等分切分（偏置是一维向量，无需转置）
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 注意力输出投影层（out_proj）对应 OpenAI 的 c_proj
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 前馈网络第一层（升维）对应 OpenAI 的 mlp.c_fc
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        # 前馈网络第二层（降维）对应 OpenAI 的 mlp.c_proj；
        # 注意 layers[1] 是 GELU（无参数），所以第二个 Linear 是 layers[2]
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 两个 LayerNorm 分别对应 OpenAI 的 ln_1（注意力前）与 ln_2（前馈前）
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

    # 最终 LayerNorm 的缩放/平移参数
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # GPT-2 中输出头（out_head）与 token 嵌入层权重共享（weight tying），
    # 所以这里再次用 "wte" 给 out_head.weight 赋值
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    """
    第 5 章：把字符串文本编码为模型可用的 token id 张量。

    参数：
        text (str): 原始文本。
        tokenizer: 分词器（需支持 .encode()）。

    返回：
        torch.Tensor: 形状 (1, seq_len) 的 LongTensor（新增一个 batch 维，batch_size=1）。
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    # unsqueeze(0)：在最前面插入 batch 维，使其符合模型期望的 (batch, seq_len) 输入格式
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    第 5 章：把模型输出的 token id 张量解码回可读字符串。

    参数：
        token_ids (torch.Tensor): 形状 (1, seq_len) 的 token id 张量（batch_size 固定假设为 1）。
        tokenizer: 分词器（需支持 .decode()）。

    返回：
        str: 解码后的文本。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    # squeeze(0)：去掉 batch 维，得到一维的 token id 列表，再交给 tokenizer 解码
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    第 5 章：计算单个 batch 的交叉熵损失（next-token prediction loss）。

    参数：
        input_batch (torch.Tensor): 形状 (batch_size, seq_len)，模型输入 token id。
        target_batch (torch.Tensor): 形状 (batch_size, seq_len)，对应的目标 token id
            （相对 input_batch 整体右移一位）。
        model (GPTModel): 模型。
        device (torch.device): 运行设备。

    返回：
        torch.Tensor: 标量张量，当前 batch 的平均交叉熵损失。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # logits 形状 (batch_size, seq_len, vocab_size) -> flatten(0,1) 后变为 (batch_size*seq_len, vocab_size)
    # target_batch flatten 后变为 (batch_size*seq_len,)，两者刚好匹配交叉熵损失的输入要求
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    第 5 章：在给定 DataLoader 上计算若干个 batch 的平均损失。

    参数：
        data_loader (DataLoader): 数据加载器。
        model (GPTModel): 模型。
        device (torch.device): 运行设备。
        num_batches (int or None): 最多评估的 batch 数；None 表示遍历整个 data_loader。

    返回：
        float: 平均损失；若 data_loader 为空则返回 float("nan")。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # 若指定的 num_batches 超过了 data_loader 实际的 batch 数，取二者较小值，避免越界
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, label="loss"):
    """
    第 5 章：绘制训练/验证损失随 epoch（及累计 token 数）变化的曲线图。

    图中使用两个共享 y 轴、不同 x 轴的坐标系：
    - 下方 x 轴：已训练的 epoch 数。
    - 上方 x 轴：累计处理过的 token 数（与下方 epoch 轴对齐，便于对比训练进度）。

    参数：
        epochs_seen (Sequence[float]): 每次记录点对应的 epoch 数（可为小数，如 0.5 表示半个 epoch）。
        tokens_seen (Sequence[int]): 每次记录点对应的累计 token 数。
        train_losses (Sequence[float]): 对应的训练损失序列。
        val_losses (Sequence[float]): 对应的验证损失序列。
        label (str): 纵轴/图例中使用的名称（默认 "loss"），并作为保存文件名的一部分。

    返回：
        None。函数会调用 plt.show() 展示图像，并把图保存为 "{label}-plot.pdf"。
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    # 主坐标轴：以 epoch 为横轴，画训练/验证损失曲线
    ax1.plot(epochs_seen, train_losses, label=f"Training {label}")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis
    # 强制 x 轴刻度只显示整数（epoch 数通常按整数展示更直观）

    # Create a second x-axis for tokens seen
    # 创建共享同一 y 轴、但拥有独立（顶部）x 轴的第二个坐标轴，用于展示 token 数
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    # alpha=0：这条曲线本身不可见，画它只是为了让 matplotlib 按 tokens_seen 的范围对齐上方 x 轴刻度
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()
