# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-5.
# This file can be run as a standalone script.

"""
模块级中文说明
==============
本文件汇总了《从零构建大语言模型》第 2~5 章已经讲过的核心代码，
供第 6 章 IMDB 情感分类 bonus 示例复用，可作为独立脚本运行。

具体包含：
- 第 2 章：GPT 预训练用的滑动窗口数据集 `GPTDatasetV1` 及其 DataLoader 构造函数。
- 第 3 章：带因果掩码（causal mask）的多头自注意力 `MultiHeadAttention`。
- 第 4 章：LayerNorm、GELU 激活函数、前馈网络 FeedForward、
  Transformer 块 TransformerBlock，以及完整的 `GPTModel`，
  外加贪心解码函数 `generate_text_simple`。
- 第 5 章：将 OpenAI 官方 GPT-2 预训练权重（TensorFlow 格式，经转换后的
  numpy 字典）加载进本仓库自建的 `GPTModel` 的 `load_weights_into_gpt`，
  以及文本 <-> token id 互转的辅助函数。

【核对说明】经与官方仓库 rasbt/LLMs-from-scratch 同名文件逐行比对，
本文件内容完全一致，未发现确定性的数值/逻辑 bug；所有张量形状变换、
GPT-2 权重切分与转置均与标准 GPT-2 结构吻合。因此本次仅补充详细中文
注释，不改动任何代码行为。文中如有风险点/跨版本兼容点，会以
「风险标注」的形式指出，但不做代码改动。
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
    GPT 预训练用的滑动窗口数据集（第 2 章）。

    将一整段文本 token 化后，用固定长度、固定步长的滑动窗口切分成
    多个 (输入序列, 目标序列) 样本对，目标序列相对输入序列整体右移一位，
    用于「预测下一个 token」的自回归语言模型训练。

    参数:
        txt (str): 原始文本（未分词）。
        tokenizer: 具备 `.encode()` 方法的分词器实例（本文件中通常是 tiktoken 的 gpt2 编码器）。
        max_length (int): 每个训练样本的 token 序列长度（即上下文窗口大小）。
        stride (int): 滑动窗口每次移动的 token 步数；stride < max_length 时窗口会重叠。

    属性:
        input_ids (List[Tensor]): 每个元素形状为 (max_length,)，是输入 token 序列。
        target_ids (List[Tensor]): 每个元素形状为 (max_length,)，是相对 input 右移一位的目标序列。
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 将整段文本一次性编码为 token id 列表；允许出现特殊 token "<|endoftext|>"
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 用滑动窗口把长文本切成多个可能重叠的定长子序列
        # 注意：range 的终点是 len(token_ids) - max_length，是为了保证
        # target_chunk 取到 i + max_length（即最后一个输入 token 的下一个 token）时不会越界
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]          # 输入片段：[i, i+max_length)
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标片段：整体右移一位，即"下一个token"标签
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集中样本（滑动窗口切片）的总数。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        按索引取出一个训练样本。

        返回:
            (input_ids[idx], target_ids[idx])，两者均为形状 (max_length,) 的 LongTensor。
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    构造基于 GPTDatasetV1 的 PyTorch DataLoader（第 2 章）。

    参数:
        txt (str): 原始文本。
        batch_size (int): 每个 batch 的样本数。
        max_length (int): 每个样本的 token 序列长度。
        stride (int): 滑动窗口步长。
        shuffle (bool): 是否打乱样本顺序。
        drop_last (bool): 若最后一个 batch 样本数不足 batch_size 是否丢弃（预训练时常设 True 以保证每个 batch 形状一致）。
        num_workers (int): DataLoader 使用的子进程数。

    返回:
        torch.utils.data.DataLoader: 每次迭代产出 (input_ids, target_ids) 两个
        形状均为 (batch_size, max_length) 的张量。
    """
    # Initialize the tokenizer
    # 固定使用 GPT-2 的 BPE 分词器（tiktoken 实现），与后续加载的 GPT-2 预训练权重保持词表一致
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
    带因果掩码（causal mask）的多头自注意力模块（第 3 章）。

    通过单次线性投影得到完整的 Q/K/V，再在张量维度上"拆分"成多个头
    （而不是为每个头单独建立线性层），是效率更高的多头注意力实现方式。

    参数:
        d_in (int): 输入特征维度。
        d_out (int): 输出特征维度（同时也是 Q/K/V 投影后的总维度），必须能被 num_heads 整除。
        context_length (int): 支持的最大序列长度，用于预先构造因果掩码矩阵。
        dropout (float): 注意力权重上的 dropout 概率。
        num_heads (int): 注意力头数。
        qkv_bias (bool): Q/K/V 线性层是否使用偏置项，默认为 False（与 GPT-2 的 QKV 投影习惯一致时需设为 True）。
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 每个注意力头的维度 = 总输出维度 / 头数

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # out_proj：多头拼接后的输出再做一次线性变换（融合各头信息）
        self.dropout = nn.Dropout(dropout)
        # 预先构造一个 (context_length, context_length) 的上三角掩码（对角线上方为1），
        # 用 register_buffer 注册为非训练参数但会随模型一起 to(device)/save
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        前向传播。

        参数:
            x (Tensor): 形状 (b, num_tokens, d_in)，b 为 batch size，num_tokens 为当前序列长度。

        返回:
            Tensor: 形状 (b, num_tokens, d_out)，融合多头注意力后的上下文向量。
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 把最后一维 d_out 拆成 (num_heads, head_dim)，即"隐式"地把大矩阵切分给各个头
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 把 num_heads 维度换到前面，方便把 num_heads 当作"批量维"参与矩阵乘法
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # 计算注意力打分：Q @ K^T，对最后两维做矩阵乘法，结果形状 (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        # 掩码矩阵是按 context_length 构造的，这里裁剪到当前实际序列长度 num_tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        # 因果掩码：把"未来"位置（上三角部分）的注意力分数填为 -inf，softmax 后趋近于 0，
        # 从而保证第 i 个 token 只能看到 <= i 的 token（自回归特性）
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 按 sqrt(head_dim) 缩放后做 softmax，缓解点积数值过大导致梯度消失的问题
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        # 用注意力权重对 V 加权求和，再把 num_heads 维度换回到 num_tokens 前面
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 把各头的输出在最后一维拼接回 d_out（reshape 要求内存连续，transpose 后需要用 reshape 而非 view）
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection
        # 最后再过一层线性层，融合各头信息（GPT-2 中对应 attn.c_proj）

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    层归一化模块（第 4 章，GPT-2 风格 LayerNorm 的手写实现）。

    对最后一维（特征维）做均值/方差归一化，再用可学习的缩放和偏移参数
    还原表达能力，能稳定深层网络训练。

    参数:
        emb_dim (int): 归一化的特征维度大小，用于初始化 scale/shift 参数形状。
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除以 0 的极小值
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习缩放参数 γ，初始为全 1
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习偏移参数 β，初始为全 0

    def forward(self, x):
        """
        参数:
            x (Tensor): 形状 (..., emb_dim)，最后一维会被归一化。

        返回:
            Tensor: 与输入同形状 (..., emb_dim)。
        """
        mean = x.mean(dim=-1, keepdim=True)
        # 注意：unbiased=False 表示用有偏方差估计（除以 N 而非 N-1），与 GPT-2 官方实现保持一致
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    GELU 激活函数（第 4 章，GPT-2 所用的 tanh 近似版本，而非精确的高斯误差函数）。

    与 nn.GELU(approximate='tanh') 等价，历史上 GPT-2 官方实现即采用该近似公式。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        参数:
            x (Tensor): 任意形状。

        返回:
            Tensor: 与输入同形状，逐元素施加 GELU 激活。
        """
        # GELU 的 tanh 近似公式：
        # 0.5x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Transformer 块中的逐位置前馈网络（第 4 章）。

    结构为 Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)，
    先升维再降维（升维系数 4 与 GPT-2 一致），对序列中每个位置独立、相同地作用。

    参数:
        cfg (dict): 配置字典，需包含键 "emb_dim"。
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
        参数:
            x (Tensor): 形状 (b, num_tokens, emb_dim)。

        返回:
            Tensor: 形状 (b, num_tokens, emb_dim)，与输入相同。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    单个 Transformer 块（第 4 章）：多头自注意力子层 + 前馈网络子层，
    均采用 Pre-LayerNorm 结构（先归一化再计算，再残差相加）。

    参数:
        cfg (dict): 配置字典，需包含 "emb_dim"、"context_length"、"n_heads"、
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
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 注意力子层前的 LayerNorm
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 前馈子层前的 LayerNorm
        self.drop_resid = nn.Dropout(cfg["drop_rate"])
        # 注：此处两个残差分支共用同一个 Dropout 实例；由于 Dropout 无可学习参数，
        # 每次前向调用仍会独立重新采样掩码，因此与分别定义两个 Dropout 实例效果等价，不是 bug。

    def forward(self, x):
        """
        参数:
            x (Tensor): 形状 (b, num_tokens, emb_dim)。

        返回:
            Tensor: 形状 (b, num_tokens, emb_dim)，与输入相同（Transformer 块保持形状不变）。
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back
        # 残差连接：保留原始输入信号，缓解深层网络梯度消失问题

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """
    完整的 GPT 模型（第 4 章）：token 嵌入 + 位置嵌入 -> N 层 TransformerBlock
    -> 最终 LayerNorm -> 线性输出头（映射到词表维度的 logits）。

    参数:
        cfg (dict): 配置字典，需包含 "vocab_size"、"emb_dim"、"context_length"、
            "drop_rate"、"n_layers" 等键（其余键会在构造 TransformerBlock 时用到）。
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])       # token 嵌入表
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])   # 可学习的绝对位置嵌入表
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # 堆叠 n_layers 个结构相同但参数独立的 Transformer 块

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 输出头：把隐藏状态映射回词表大小的 logits，无偏置项（与 GPT-2 一致）

    def forward(self, in_idx):
        """
        参数:
            in_idx (LongTensor): 形状 (batch_size, seq_len)，输入 token id 序列。

        返回:
            Tensor: 形状 (batch_size, seq_len, vocab_size)，每个位置对下一个 token 的预测 logits。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # (batch_size, seq_len, emb_dim)
        # 位置嵌入按 [0, 1, ..., seq_len-1] 取值，广播加到每个样本上
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    贪心解码：逐步生成新 token（第 4 章，最简单的自回归生成方式，无采样、无温度）。

    参数:
        model (GPTModel): 已训练/加载权重的 GPT 模型。
        idx (LongTensor): 形状 (B, T)，当前已有的上下文 token 序列。
        max_new_tokens (int): 需要新生成的 token 数量。
        context_size (int): 模型支持的最大上下文长度，用于裁剪过长的输入。

    返回:
        LongTensor: 形状 (B, T + max_new_tokens)，在原序列后追加了新生成的 token。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 若当前序列长度超过模型支持的上下文窗口，只保留最近 context_size 个 token
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # 推理阶段不需要计算梯度，用 no_grad 节省显存/加速
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 只关心最后一个位置的预测（即"下一个 token"的分布）
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 贪心策略：直接取概率（logits）最大的词表 id，不做采样
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 把新生成的 token 拼接到序列末尾，作为下一轮生成的输入
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# Chapter 5
#####################################
def assign(left, right):
    """
    将 numpy 权重 `right` 转成 `torch.nn.Parameter` 并赋值给 `left`（第 5 章，
    用于把 OpenAI 官方 GPT-2 权重加载进本仓库自建模型）。

    参数:
        left (Tensor or Parameter): 模型中原有的参数张量，仅用于校验形状。
        right (np.ndarray): 待加载的 GPT-2 官方权重数组，形状需与 left 完全一致。

    返回:
        torch.nn.Parameter: 包裹 right 数据的新参数张量，可直接赋值给模型对应属性。

    异常:
        ValueError: 当 left 与 right 形状不一致时抛出，避免静默加载错误权重。

    风险标注（不改动代码）：这里用 torch.tensor(right) 而非 torch.as_tensor(right)，
    如果 right 本身已经是 torch.Tensor 会触发 "UserWarning: To copy construct from
    a tensor..." 警告；但在本文件的实际调用场景中 right 均为 numpy.ndarray，
    不会触发该警告，因此不属于需要修复的确定性 bug，仅作跨版本/跨用法提示。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    将 OpenAI 官方 GPT-2 预训练权重（已转换为嵌套 dict 的 numpy 数组）
    逐一拷贝进本仓库自建的 GPTModel 对应参数中（第 5 章）。

    参数:
        gpt (GPTModel): 待加载权重的目标模型实例（结构需与 cfg 匹配，即 n_layers/emb_dim 等一致）。
        params (dict): GPT-2 官方权重字典，典型结构：
            {
                "wpe": (context_length, emb_dim)  位置嵌入,
                "wte": (vocab_size, emb_dim)      token 嵌入（与输出头权重共享/绑定）,
                "g", "b": 最终 LayerNorm 的 scale/shift,
                "blocks": [
                    {
                        "attn": {
                            "c_attn": {"w": (emb_dim, 3*emb_dim), "b": (3*emb_dim,)},  # QKV 合并权重
                            "c_proj": {"w": (emb_dim, emb_dim), "b": (emb_dim,)},       # 注意力输出投影
                        },
                        "mlp": {
                            "c_fc":   {"w": (emb_dim, 4*emb_dim), "b": (4*emb_dim,)},
                            "c_proj": {"w": (4*emb_dim, emb_dim), "b": (emb_dim,)},
                        },
                        "ln_1": {"g": (emb_dim,), "b": (emb_dim,)},
                        "ln_2": {"g": (emb_dim,), "b": (emb_dim,)},
                    },
                    ... (每层一个 dict)
                ]
            }

    返回:
        None（原地修改 gpt 的各层参数）。

    说明: GPT-2 官方权重使用的是 Conv1D（形状为 (in, out)），而 PyTorch 的
    nn.Linear 权重形状是 (out, in)，因此下面凡是权重矩阵都需要做 `.T` 转置；
    偏置是一维向量，转置无意义故不做 `.T`。
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # c_attn 是 Q/K/V 三个投影矩阵在最后一维拼接后的结果，需按最后一维切成三份
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)  # Conv1D(in,out) -> Linear(out,in) 需转置
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 偏置同样按最后一维切成 Q/K/V 三份，一维向量无需转置
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 注意力输出投影（对应 attn.c_proj）
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 前馈网络第一层（升维，对应 mlp.c_fc）
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        # 前馈网络第三层（layers[1] 是 GELU，无参数；layers[2] 是降维层，对应 mlp.c_proj）
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 两个子层各自的 LayerNorm 缩放/偏移参数
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

    # 模型最外层的最终 LayerNorm
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 输出头权重与 token 嵌入权重共享（weight tying），这里直接复用 wte
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    """
    把字符串文本编码为模型输入所需的 token id 张量。

    参数:
        text (str): 原始文本。
        tokenizer: 具备 `.encode()` 方法的分词器。

    返回:
        LongTensor: 形状 (1, seq_len)，已增加 batch 维（batch_size=1）。
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    把 token id 张量解码回字符串文本。

    参数:
        token_ids (LongTensor): 形状 (1, seq_len)，带 batch 维（batch_size=1）。
        tokenizer: 具备 `.decode()` 方法的分词器。

    返回:
        str: 解码后的文本。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
