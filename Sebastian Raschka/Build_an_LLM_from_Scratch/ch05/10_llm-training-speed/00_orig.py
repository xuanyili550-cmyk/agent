# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本文件中文说明(模块级 docstring,仅为新增注释,不影响任何可执行逻辑):

这是《Build a Large Language Model From Scratch》(从零构建大语言模型)一书代码仓库中,
第 5 章 "LLM 训练速度优化" 系列示例(ch05/10_llm-training-speed)的 **基线(未优化)版本** —— 00_orig.py。
后续同目录下的其他脚本会在此版本基础上逐步引入混合精度、torch.compile、Flash Attention、
数据加载优化等手段,并与本文件的运行速度做对比。

文件结构概览:
    - Chapter 2 部分:实现 GPTDatasetV1 数据集(用滑动窗口把长文本切成 (输入, 目标) token 序列对)
      以及 create_dataloader_v1 工厂函数,用于构建 PyTorch DataLoader。
    - Chapter 3 部分:实现 MultiHeadAttention 多头自注意力模块(带因果掩码的缩放点积注意力)。
    - Chapter 4 部分:实现 LayerNorm、GELU 激活函数、FeedForward 前馈网络、TransformerBlock
      以及完整的 GPTModel;并提供一个简单的贪心解码生成函数 generate_text_simple。
    - Chapter 5 部分:训练相关工具函数(token 与文本互转、损失计算、模型评估、生成样例文本打印),
      核心是 train_model_simple_with_timing —— 在标准训练循环基础上,额外统计并打印
      每秒处理的 token 数(tokens/sec,含单区间与累计平均两种指标)以及 GPU 显存占用情况,
      用于衡量和对比不同训练速度优化手段的效果。
    - 主函数 main():下载/读取《米德尔马契》(Middlemarch)英文文本数据,构建 GPT-124M 配置的模型,
      划分训练/验证集并构建 DataLoader,调用训练函数完成训练。
    - 脚本入口 (__main__):定义 GPT_CONFIG_124M 模型配置与 OTHER_SETTINGS 训练超参数,
      启动训练并绘制/保存训练与验证损失曲线。

重要提示:本次改动仅新增中文注释与 docstring,未修改任何变量名、函数签名、控制流或缩进等可执行代码,
原有英文注释均予以保留。
"""


import os
import time
import urllib.request

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    """GPT 训练用数据集(第 2 章)。

    使用滑动窗口(sliding window)将整段文本切分成若干个长度为 max_length 的重叠 token 序列,
    每个样本的目标序列(target)是输入序列(input)整体右移一位的结果,用于"预测下一个 token"
    这一自回归语言建模训练目标。

    参数:
        txt (str): 原始训练文本(未分词的字符串)。
        tokenizer: 具备 .encode() 方法的分词器实例(此处为 tiktoken 的 gpt2 编码器)。
        max_length (int): 每个训练样本的 token 序列长度(即上下文窗口大小)。
        stride (int): 滑动窗口每次移动的步长;stride < max_length 时相邻样本会有重叠,
            stride == max_length 时样本之间无重叠。

    属性:
        input_ids (List[torch.Tensor]): 每个元素形状为 (max_length,) 的输入 token 序列。
        target_ids (List[torch.Tensor]): 每个元素形状为 (max_length,) 的目标 token 序列
            (相对于对应的 input_ids 整体右移一个 token)。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 中文:先把整段文本一次性编码为 token id 列表(允许出现特殊 token "<|endoftext|>")
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 中文:用滑动窗口把长 token 序列切分成多个长度为 max_length 的(输入, 目标)样本对;
        # 目标序列是输入序列整体右移一位,即每个位置的标签是"下一个 token",这是自回归语言模型的训练方式
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集中样本(即切分出的序列对)的总数量。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """按索引取出一个训练样本。

        参数:
            idx (int): 样本索引。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: (输入 token 序列, 目标 token 序列),
                两者形状均为 (max_length,)。
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """构建 GPT 训练/验证用的 PyTorch DataLoader。

    内部会先用 tiktoken 的 gpt2 编码器对文本分词,再基于 GPTDatasetV1 做滑动窗口切分,
    最后包装成 DataLoader 以支持批量(batch)读取、打乱(shuffle)与多进程加载。

    参数:
        txt (str): 原始训练/验证文本。
        batch_size (int): 每个 batch 包含的样本数。
        max_length (int): 每个样本的 token 序列长度(等同于模型的上下文长度)。
        stride (int): 滑动窗口步长。
        shuffle (bool): 是否在每个 epoch 打乱样本顺序。
        drop_last (bool): 若最后一个 batch 样本数不足 batch_size,是否丢弃。
        num_workers (int): 数据加载使用的子进程数(用于并行预取,提升数据读取吞吐)。

    返回:
        torch.utils.data.DataLoader: 可迭代产生 (input_batch, target_batch) 的数据加载器,
            其中每个 batch 的张量形状均为 (batch_size, max_length)。
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # 中文:num_workers > 0 时使用多进程并行加载数据,可减少训练时 CPU 数据准备成为瓶颈的情况,
    # 这是本系列"训练速度优化"要考察的因素之一
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """多头自注意力模块(第 3 章),带因果掩码(causal mask)的缩放点积注意力。

    将输入的 Q/K/V 投影结果拆分为 num_heads 个头并行计算注意力,再把各头的输出拼接后
    经过一次线性投影(out_proj)得到最终输出,是 Transformer 编码/解码块的核心组件之一。

    参数:
        d_in (int): 输入特征维度。
        d_out (int): 输出特征维度(必须能被 num_heads 整除)。
        context_length (int): 支持的最大上下文长度,用于预先构建因果掩码矩阵。
        dropout (float): 注意力权重上使用的 dropout 概率。
        num_heads (int): 注意力头数。
        qkv_bias (bool): Q/K/V 线性层是否使用偏置项。
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文:每个注意力头的维度 = 总输出维度 / 头数,保证多头拼接后维度仍等于 d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        # 中文:注册为 buffer(非可训练参数),torch.triu(..., diagonal=1) 生成上三角(不含对角线)为 1 的矩阵,
        # 用作因果掩码——确保每个位置只能看到自己及之前的位置,不能"偷看"未来的 token
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """前向传播,计算带因果掩码的多头自注意力输出。

        参数:
            x (torch.Tensor): 输入张量,形状为 (b, num_tokens, d_in),
                其中 b 为 batch 大小,num_tokens 为序列长度。

        返回:
            torch.Tensor: 注意力输出,形状为 (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 中文:把最后一维 d_out 拆分成 (num_heads, head_dim) 两维,为并行计算多个头做准备
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 中文:把 num_heads 维提前,便于后续在最后两维(num_tokens, head_dim)上做批量矩阵乘法
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文:queries 与 keys 的转置做矩阵乘法,得到每个 query 位置对每个 key 位置的原始注意力分数,
        # 形状为 (b, num_heads, num_tokens, num_tokens)

        # Original mask truncated to the number of tokens and converted to boolean
        # 中文:因果掩码矩阵是按 context_length 预先构建的,这里截取到当前实际序列长度 num_tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 中文:将掩码中为 True(即未来位置)的注意力分数填充为 -inf,softmax 后这些位置权重趋近于 0,
        # 从而实现"只能看过去,不能看未来"的因果约束
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 中文:除以 sqrt(head_dim) 做缩放(scaled dot-product),避免维度较大时点积数值过大导致 softmax 梯度消失
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 中文:对注意力权重做 dropout,起正则化作用

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 中文:注意力权重与 values 加权求和得到上下文向量,再把 num_heads 维换回到 num_tokens 之后,方便后续拼接
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文:把多头拼接回单一维度 d_out
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化(Layer Normalization)模块(第 4 章的手写实现)。

    对输入张量最后一维(特征维)做归一化,并引入可学习的缩放(scale)与偏移(shift)参数,
    有助于稳定深层 Transformer 网络的训练。

    参数:
        emb_dim (int): 输入的特征维度(嵌入维度),scale/shift 参数形状均为 (emb_dim,)。
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文:数值稳定项,防止除以 0
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """对输入最后一维做归一化并施加可学习的缩放与偏移。

        参数:
            x (torch.Tensor): 输入张量,形状为 (..., emb_dim)。

        返回:
            torch.Tensor: 与输入形状相同的归一化结果,形状为 (..., emb_dim)。
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 中文:unbiased=False 表示使用有偏方差估计(除以 N 而非 N-1),与常见 LayerNorm 实现保持一致
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU 激活函数的 tanh 近似实现(第 4 章)。

    对应 GPT-2 等模型中常用的 GELU 激活的近似公式,而非 PyTorch 内置的精确 erf 版本。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """对输入逐元素应用 GELU(tanh 近似版)激活。

        参数:
            x (torch.Tensor): 任意形状的输入张量。

        返回:
            torch.Tensor: 与输入形状相同的激活结果。
        """
        # 中文:GELU 的 tanh 近似公式:0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer Block 中的前馈网络(FFN,第 4 章)。

    结构为:线性升维(emb_dim -> 4*emb_dim) -> GELU 激活 -> 线性降维(4*emb_dim -> emb_dim),
    是 Transformer 中在注意力子层之后的逐位置(position-wise)非线性变换模块。

    参数:
        cfg (dict): 模型配置字典,需包含键 "emb_dim"(嵌入维度)。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """前向传播。

        参数:
            x (torch.Tensor): 输入张量,形状为 (b, num_tokens, emb_dim)。

        返回:
            torch.Tensor: 输出张量,形状为 (b, num_tokens, emb_dim)(与输入相同)。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """单个 Transformer 解码器块(第 4 章),GPT 模型的基本重复单元。

    采用 Pre-LayerNorm 结构:先归一化再进入子层(注意力/前馈),并通过残差连接(shortcut)
    将子层输入与子层输出相加,缓解深层网络的梯度消失/爆炸问题。

    参数:
        cfg (dict): 模型配置字典,需包含 "emb_dim"、"context_length"、"n_heads"、
            "drop_rate"、"qkv_bias" 等键。
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
        """前向传播:依次经过"归一化 + 注意力 + 残差"与"归一化 + 前馈 + 残差"两个子层。

        参数:
            x (torch.Tensor): 输入张量,形状为 (batch_size, num_tokens, emb_dim)。

        返回:
            torch.Tensor: 输出张量,形状为 (batch_size, num_tokens, emb_dim)(与输入相同)。
        """
        # Shortcut connection for attention block
        # 中文:保存注意力子层的输入,用于残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        # 中文:保存前馈子层的输入,用于残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的 GPT 风格自回归语言模型(第 4 章)。

    结构:token 嵌入 + 位置嵌入 -> Dropout -> 堆叠 n_layers 个 TransformerBlock ->
    最终 LayerNorm -> 线性输出头(映射到词表大小,得到每个位置在词表上的 logits)。

    参数:
        cfg (dict): 模型配置字典,需包含 "vocab_size"、"emb_dim"、"context_length"、
            "drop_rate"、"n_layers" 等键(TransformerBlock 所需的键也一并包含在 cfg 中)。
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
        """前向传播,计算下一个 token 的 logits。

        参数:
            in_idx (torch.Tensor): 输入 token id 序列,形状为 (batch_size, seq_len),
                dtype 为 long/int64。

        返回:
            torch.Tensor: 每个位置在词表上的 logits,形状为 (batch_size, seq_len, vocab_size)。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文:位置嵌入按 0..seq_len-1 的位置索引查表,与 token 嵌入相加得到"内容+位置"的联合表示
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """基于贪心解码(greedy decoding)的简单自回归文本生成函数(第 4 章)。

    每一步都取模型在当前上下文最后一个位置输出 logits 中概率最大(argmax)的 token 作为下一个 token,
    并拼接到序列末尾,循环 max_new_tokens 次。

    参数:
        model (nn.Module): 训练好的(或训练中的)GPTModel 实例。
        idx (torch.Tensor): 初始上下文 token id,形状为 (B, T)。
        max_new_tokens (int): 需要生成的新 token 数量。
        context_size (int): 模型支持的最大上下文长度,用于裁剪过长的上下文。

    返回:
        torch.Tensor: 拼接了新生成 token 后的完整序列,形状为 (B, T + max_new_tokens)。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 中文:只保留最近 context_size 个 token 作为模型输入,避免超出模型支持的最大上下文长度
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # 中文:生成阶段不需要计算梯度,用 torch.no_grad() 节省显存并加速前向计算
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 中文:只关心序列最后一个位置的预测结果,因为这才是"下一个 token"的预测
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 中文:贪心解码——直接取概率(logits)最大的词表索引,而非按分布采样
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

#####################################
# Chapter 5
#####################################


def text_to_token_ids(text, tokenizer):
    """将文本字符串编码为带 batch 维度的 token id 张量。

    参数:
        text (str): 待编码的文本。
        tokenizer: 具备 .encode() 方法的分词器(如 tiktoken 的 gpt2 编码器)。

    返回:
        torch.Tensor: 形状为 (1, seq_len) 的 token id 张量(增加了 batch 维)。
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将带 batch 维度的 token id 张量解码回文本字符串。

    参数:
        token_ids (torch.Tensor): 形状为 (1, seq_len) 的 token id 张量。
        tokenizer: 具备 .decode() 方法的分词器。

    返回:
        str: 解码后的文本。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个 batch 的交叉熵损失(语言模型的"预测下一个 token"损失)。

    参数:
        input_batch (torch.Tensor): 输入 token id,形状为 (batch_size, seq_len)。
        target_batch (torch.Tensor): 目标 token id,形状为 (batch_size, seq_len)。
        model (nn.Module): GPTModel 实例。
        device (torch.device): 计算设备(cpu/cuda)。

    返回:
        torch.Tensor: 标量损失值(0 维张量)。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # 中文:把 (batch_size, seq_len, vocab_size) 展平前两维为 (batch_size*seq_len, vocab_size),
    # 目标同样展平为 (batch_size*seq_len,),再计算逐 token 的交叉熵损失(内部自动做 softmax)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """在给定的 DataLoader 上计算平均损失(可指定只评估前若干个 batch,加速评估过程)。

    参数:
        data_loader (DataLoader): 训练或验证用的数据加载器。
        model (nn.Module): GPTModel 实例。
        device (torch.device): 计算设备。
        num_batches (int, optional): 最多评估的 batch 数;为 None 时评估整个 data_loader。

    返回:
        float: 平均损失值;若 data_loader 为空则返回 float("nan")。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 中文:num_batches 不能超过 data_loader 实际的 batch 总数
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """在训练/验证集上分别评估模型损失(临时切换到 eval 模式,评估后恢复 train 模式)。

    参数:
        model (nn.Module): GPTModel 实例。
        train_loader (DataLoader): 训练集数据加载器。
        val_loader (DataLoader): 验证集数据加载器。
        device (torch.device): 计算设备。
        eval_iter (int): 每个数据加载器最多评估的 batch 数。

    返回:
        Tuple[float, float]: (训练集平均损失, 验证集平均损失)。
    """
    model.eval()  # 中文:关闭 dropout 等训练专用行为,保证评估结果确定且不引入额外噪声
    with torch.no_grad():  # 中文:评估阶段不需要梯度,节省显存与计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 中文:评估结束后切回训练模式,恢复 dropout 等行为
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """在训练过程中生成一段样例文本并打印,用于直观观察模型当前的生成效果。

    参数:
        model (nn.Module): GPTModel 实例。
        tokenizer: 分词器实例。
        device (torch.device): 计算设备。
        start_context (str): 用作生成起始上下文的提示文本。

    返回:
        None(直接打印生成结果)。
    """
    model.eval()  # 中文:生成前切换到 eval 模式
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()  # 中文:打印完成后切回训练模式,继续训练


def train_model_simple_with_timing(model, train_loader, val_loader, optimizer, device,
                                   num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """带训练速度(tokens/sec)与显存统计的简单训练循环(本文件的核心/基线实现)。

    这是本"训练速度优化"系列示例的基线版本:标准的前向-反向-优化步骤之外,额外记录:
    - 每个评估区间(eval_freq 个 step)内处理的 token 数与耗时,计算区间 tokens/sec;
    - 从训练开始(跳过第一个区间,因其常包含额外的预热/编译开销)累计的平均 tokens/sec;
    - 每个 epoch 结束后(若使用 CUDA)打印已分配/已保留的 GPU 显存。

    参数:
        model (nn.Module): 待训练的 GPTModel 实例。
        train_loader (DataLoader): 训练集数据加载器。
        val_loader (DataLoader): 验证集数据加载器。
        optimizer (torch.optim.Optimizer): 优化器实例(如 AdamW)。
        device (torch.device): 训练设备(cpu/cuda)。
        num_epochs (int): 训练的 epoch 数。
        eval_freq (int): 每隔多少个训练 step 做一次评估与速度统计。
        eval_iter (int): 每次评估时,在 train/val loader 上各评估的 batch 数。
        start_context (str): 每个 epoch 结束后用于生成样例文本的起始提示。
        tokenizer: 分词器实例,供生成样例文本使用。

    返回:
        Tuple[List[float], List[float], List[int]]:
            (每次评估记录的训练损失列表, 每次评估记录的验证损失列表,
             每次评估时累计已处理的 token 总数列表)。
    """
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    # Variables for cumulative average tokens/sec
    # 中文:用于计算"从训练开始以来(排除首个区间)"的累计平均 tokens/sec
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # CUDA-specific timing setup
    # 中文:GPU 上的操作是异步执行的,普通 time.time() 计时不准确,需用 CUDA Event 配合
    # torch.cuda.synchronize() 才能得到精确的 GPU 端耗时;CPU 上则直接用 time.time() 即可
    use_cuda = device.type == "cuda"
    if use_cuda:
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()  # Ensure all prior CUDA operations are done
        t_start.record()          # Start the timer for the first interval
    else:
        t0 = time.time()          # Start the timer for the first interval

    # Main training loop
    for epoch in range(num_epochs):
        model.train()
        for inp_batch, tgt_batch in train_loader:
            optimizer.zero_grad()  # 中文:清空上一步残留的梯度
            global_step += 1

            # Forward and backward pass
            loss = calc_loss_batch(inp_batch, tgt_batch, model, device)
            loss.backward()  # 中文:反向传播计算梯度
            optimizer.step()  # 中文:根据梯度更新模型参数

            total_tokens += inp_batch.numel()  # 中文:累计已处理的 token 总数(batch_size * seq_len)

            # At evaluation intervals, measure elapsed time and tokens per second
            if global_step % eval_freq == 0:
                # End timing for the current interval
                if use_cuda:
                    t_end.record()
                    torch.cuda.synchronize()  # Wait for all CUDA ops to complete.
                    elapsed = t_start.elapsed_time(t_end) / 1000  # Convert ms to seconds
                    t_start.record()  # Reset timer for the next interval
                else:
                    elapsed = time.time() - t0
                    t0 = time.time()  # Reset timer for the next interval

                # Calculate tokens processed in this interval
                # 中文:本次评估区间内新处理的 token 数 = 当前累计 token 数 - 上次记录时的累计 token 数
                tokens_interval = total_tokens - last_tokens
                last_tokens = total_tokens
                tps = tokens_interval / elapsed if elapsed > 0 else 0  # Tokens per second

                # Update cumulative counters (skip the first evaluation interval)
                # 中文:跳过第一个评估区间不计入累计统计,因为其中可能包含 CUDA 初始化、
                # cudnn 自动调优等一次性预热开销,会拉低平均速度的代表性
                if global_step:  # This is False only when global_step == 0 (first evaluation)
                    cumulative_tokens += tokens_interval
                    cumulative_time += elapsed

                # Compute cumulative average tokens/sec (excluding the first interval)
                avg_tps = cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # Evaluate model performance (this may add overhead)
                # 中文:评估本身会占用额外时间(已在下一区间的计时中体现),用于监控训练/验证损失走势
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)

                print(f"Ep {epoch+1}, Step {global_step:06d}, "
                      f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                      f"Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}")

        generate_and_print_sample(model, tokenizer, device, start_context)

        # Memory stats
        # 中文:每个 epoch 结束后打印 GPU 显存占用,便于对比不同优化手段(如混合精度)对显存的影响
        if torch.cuda.is_available():
            device = torch.cuda.current_device()

            allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB

            print(f"\nAllocated memory: {allocated:.4f} GB")
            print(f"Reserved memory: {reserved:.4f} GB\n")

    return train_losses, val_losses, track_tokens


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制训练/验证损失随 epoch 与已处理 token 数变化的曲线图。

    使用双 x 轴:主 x 轴为 epoch 数,次 x 轴(共享同一 y 轴)为已处理的 token 总数,
    方便同时从"训练轮数"和"数据吞吐量"两个角度观察损失变化。

    参数:
        epochs_seen (torch.Tensor): 与 train_losses 等长,表示每次记录点对应的 epoch 数(可为小数)。
        tokens_seen (List[int]): 与 train_losses 等长,表示每次记录点对应的累计 token 数。
        train_losses (List[float]): 训练损失记录列表。
        val_losses (List[float]): 验证损失记录列表。

    返回:
        None(直接在当前 matplotlib 图形上绘图,调用方负责保存/展示)。
    """
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


#####################################
# Main function calls
#####################################

def main(gpt_config, settings):
    """训练流程主函数:准备数据、初始化模型与优化器、构建 DataLoader,并执行训练。

    参数:
        gpt_config (dict): GPT 模型结构配置(如 GPT_CONFIG_124M),
            包含 vocab_size、context_length、emb_dim、n_heads、n_layers、drop_rate、qkv_bias。
        settings (dict): 训练超参数配置(如 OTHER_SETTINGS),
            包含 learning_rate、num_epochs、batch_size、weight_decay。

    返回:
        Tuple[List[float], List[float], List[int], nn.Module]:
            (训练损失列表, 验证损失列表, 累计 token 数列表, 训练完成后的模型实例)。
    """

    torch.manual_seed(123)  # 中文:固定随机种子,保证模型初始化、shuffle 等具有可复现性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print()

    ##############################
    # Download data if necessary
    ##############################

    file_path = "middlemarch.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"

    # 中文:若本地不存在训练语料文件,则从古登堡计划(Project Gutenberg)下载《米德尔马契》全文并缓存到本地;
    # 若已存在则直接读取本地文件,避免重复下载
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    # 中文:训练集 DataLoader,stride == context_length 表示样本之间不重叠,
    # num_workers=4 使用 4 个子进程并行加载数据以提升吞吐
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=4
    )

    # 中文:验证集 DataLoader,不丢弃最后一个不完整 batch(drop_last=False),也不打乱顺序
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=4
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    # 中文:调用本文件的核心训练函数,eval_freq=15 表示每 15 个 step 评估一次并统计速度,
    # eval_iter=1 表示每次评估仅用 1 个 batch(快速评估,减少评估本身对训练速度的干扰)
    train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],
        eval_freq=15,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # 中文:GPT-124M(约 1.24 亿参数)模型的结构配置,与 GPT-2 small 规格一致
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Input tokens per training example
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-key-value bias
    }

    # 中文:训练超参数配置
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 15,
        "batch_size": 8,
        "weight_decay": 0.1
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    # Plot results
    # 中文:构造与 train_losses 等长的、均匀分布在 [0, num_epochs] 区间的 epoch 坐标,用于绘图 x 轴
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    # torch.save(model.state_dict(), "model.pth")
    # model = GPTModel(GPT_CONFIG_124M)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))
