# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
第 4 章：从零实现一个 GPT 模型（GPT-like LLM）。

本模块是全书第 4 章的核心代码，把前面章节实现的多头自注意力（MultiHeadAttention，
定义在 ch03.py 中）组装成一个完整的、可训练/可推理的类 GPT 架构，并提供最朴素的
贪心解码（greedy decoding）文本生成函数。

主要内容（按依赖顺序）：
    - LayerNorm：手写的层归一化（Layer Normalization），对每个 token 的
      特征维度做标准化，并配有可学习的缩放（scale）和平移（shift）参数。
    - GELU：手写的 GELU 激活函数（使用 tanh 近似公式），比 ReLU 更平滑，
      是 GPT-2 等模型前馈网络中常用的激活函数。
    - FeedForward：Transformer Block 中的前馈子层（Position-wise
      Feed-Forward Network），先把维度扩大 4 倍再投影回原维度，中间插入 GELU。
    - TransformerBlock：把「多头自注意力 + 前馈网络」通过「层归一化 + 残差
      连接（shortcut connection）+ Dropout」组织起来的标准 Transformer 块
      （这里使用的是 Pre-LayerNorm 结构，即先归一化再进入子层）。
    - GPTModel：完整的 GPT 模型，由 token 嵌入、位置嵌入、若干个堆叠的
      TransformerBlock、最终层归一化以及输出线性层（语言模型头）组成。
    - generate_text_simple：最简单的自回归文本生成函数，每一步都贪心地
      （greedy）选取概率最大的下一个 token，不使用采样、温度或 top-k/top-p。

文件末尾的 "Bonus" 部分提供了功能等价但使用 PyTorch 内置算子（nn.LayerNorm、
nn.GELU、scaled_dot_product_attention）实现的加速版本（FeedForwardFast、
TransformerBlockFast、GPTModelFast），在 GPU 上可以借助 FlashAttention 等
底层优化获得更快的训练速度，但数学上与上面的手写版本是等价的。

注意：本文件仅在原始英文注释基础上新增中文注释与 docstring，未修改任何
可执行代码（变量名、函数签名、逻辑顺序、缩进、字符串字面量、import 均保持原样）。
"""

from .ch03 import MultiHeadAttention, PyTorchMultiHeadAttention
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）模块。

    与 BatchNorm 不同，LayerNorm 是对**每个样本、每个 token** 在其
    特征维度（emb_dim，即最后一维）上做均值/方差归一化，不依赖 batch 内
    其他样本，因此在处理变长序列、小 batch 甚至 batch_size=1 时都很稳定，
    这也是 Transformer 类模型普遍采用 LayerNorm 而不是 BatchNorm 的原因。

    归一化之后会乘以可学习的 scale（初始化为全 1）并加上可学习的 shift
    （初始化为全 0），让网络有能力在训练中自己学出「要不要归一化、归一化
    到什么程度」，而不是被强行限制在标准正态分布上。

    Args:
        emb_dim (int): 嵌入维度大小，即输入张量最后一维的大小。
    """
    def __init__(self, emb_dim):
        super().__init__()
        # 防止除以 0 的极小常数（数值稳定性）
        self.eps = 1e-5
        # 可学习的缩放参数，形状为 (emb_dim,)，初始化为全 1（相当于初始不缩放）
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 可学习的平移参数，形状为 (emb_dim,)，初始化为全 0（相当于初始不平移）
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """前向传播：对最后一维（emb_dim）做归一化。

        Args:
            x (torch.Tensor): 形状为 (batch_size, num_tokens, emb_dim) 的输入张量。

        Returns:
            torch.Tensor: 与输入形状相同 (batch_size, num_tokens, emb_dim)，
                每个 token 的特征已在 emb_dim 维度上被归一化，再经过
                scale/shift 仿射变换。
        """
        # 沿最后一维（特征维度）求均值，keepdim=True 保持维度便于广播
        # mean 形状: (batch_size, num_tokens, 1)
        mean = x.mean(dim=-1, keepdim=True)
        # 沿最后一维求方差；unbiased=False 表示用有偏估计（除以 N 而非 N-1），
        # 与深度学习框架中 LayerNorm 的标准实现保持一致
        # var 形状: (batch_size, num_tokens, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 标准化：(x - 均值) / 标准差，得到均值为 0、方差为 1 的分布
        # norm_x 形状: (batch_size, num_tokens, emb_dim)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 仿射变换：乘以可学习的 scale 再加上可学习的 shift，
        # scale/shift 形状为 (emb_dim,)，会广播到 (batch_size, num_tokens, emb_dim)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """手写实现的 GELU（Gaussian Error Linear Unit）激活函数。

    这里使用的是 GELU 的 tanh 近似公式（与 GPT-2 官方实现一致），
    相比精确的 GELU（用高斯误差函数 erf 计算）计算更快，且数值上非常接近：

        GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))

    相比 ReLU，GELU 是平滑、处处可导的激活函数，在 x 为负值时不会完全置零，
    而是有一个小的负值输出，这有助于梯度流动，是 GPT 系列前馈网络的标准选择。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """计算 GELU 激活值。

        Args:
            x (torch.Tensor): 任意形状的输入张量。

        Returns:
            torch.Tensor: 与输入形状相同，逐元素应用 GELU 后的结果。
        """
        # tanh 近似版 GELU 公式：
        # 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Transformer Block 中的前馈子层（Position-wise Feed-Forward Network）。

    结构为：Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)。
    先把特征维度放大 4 倍（这是 GPT-2/Transformer 论文中的经验设定），
    在更高维空间中通过非线性激活提取特征，再投影回原始的 emb_dim，
    使得该子层的输入输出形状保持一致，便于与残差连接（shortcut）相加。

    Args:
        cfg (dict): 模型配置字典，需包含键 "emb_dim"（嵌入维度）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # 升维：emb_dim -> 4 * emb_dim
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # 非线性激活（手写版 GELU）
            GELU(),
            # 降维：4 * emb_dim -> emb_dim，恢复原始维度
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 形状为 (batch_size, num_tokens, emb_dim)。

        Returns:
            torch.Tensor: 形状与输入相同 (batch_size, num_tokens, emb_dim)。
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """标准的 GPT 风格 Transformer 块（Pre-LayerNorm 结构）。

    每个 TransformerBlock 包含两个子层：
        1. 多头自注意力子层（Multi-Head Self-Attention）
        2. 前馈网络子层（Feed-Forward Network）

    每个子层都遵循「先归一化，再进子层，再 Dropout，再残差相加」的模式，
    即：x = x + Dropout(SubLayer(LayerNorm(x)))，这被称为 Pre-LN
    （Pre-LayerNorm）结构，相比原始 Transformer 论文的 Post-LN 结构，
    在训练更深的网络时更稳定，梯度更容易传播（这也是残差连接/shortcut
    connection 的核心作用：让梯度可以直接沿着恒等路径回传，缓解梯度消失）。

    Args:
        cfg (dict): 模型配置字典，需包含 "emb_dim"、"context_length"、
            "n_heads"、"drop_rate"、"qkv_bias" 等键。
    """
    def __init__(self, cfg):
        super().__init__()
        # 多头自注意力层，来自 ch03.py 的手写实现
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        # 前馈网络子层
        self.ff = FeedForward(cfg)
        # 两个独立的 LayerNorm：一个用在注意力子层之前，一个用在前馈子层之前
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 残差分支上的 Dropout，用于正则化、缓解过拟合
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """前向传播：依次经过注意力子层和前馈子层，各自带残差连接。

        Args:
            x (torch.Tensor): 形状为 (batch_size, num_tokens, emb_dim)。

        Returns:
            torch.Tensor: 形状与输入相同 (batch_size, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        # 保存原始输入，用于之后的残差相加（shortcut connection）
        shortcut = x
        # 先做层归一化（Pre-LN），再送入注意力层
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        # 对注意力子层的输出做 Dropout 正则化
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back
        # 残差相加：把子层输出与归一化前的原始输入相加，形成恒等映射的捷径，
        # 使梯度可以绕过非线性变换直接回传，缓解深层网络的梯度消失问题

        # Shortcut connection for feed-forward block
        # 同样的模式：保存输入 -> 归一化 -> 前馈子层 -> Dropout -> 残差相加
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的 GPT 模型（类 GPT-2 架构）。

    整体流程：
        输入 token id -> token 嵌入 + 位置嵌入 -> Dropout ->
        堆叠 N 个 TransformerBlock -> 最终 LayerNorm ->
        线性输出层（语言模型头，输出词表大小的 logits）

    Args:
        cfg (dict): 模型配置字典，需包含：
            - "vocab_size": 词表大小
            - "emb_dim": 嵌入维度
            - "context_length": 支持的最大上下文长度（用于位置嵌入表大小）
            - "drop_rate": Dropout 概率
            - "n_layers": 堆叠的 TransformerBlock 数量
    """
    def __init__(self, cfg):
        super().__init__()
        # token 嵌入表：把每个 token id 映射为 emb_dim 维向量
        # 参数形状: (vocab_size, emb_dim)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入表：为序列中每个位置（0 到 context_length-1）学习一个向量，
        # 用于向模型注入 token 的顺序信息（因为自注意力本身不区分位置）
        # 参数形状: (context_length, emb_dim)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 嵌入层之后的 Dropout，用于正则化
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠 n_layers 个 TransformerBlock，构成模型的主干（backbone）
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 最终的层归一化，在进入输出层之前对特征再做一次归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 输出层（语言模型头 / LM head）：把 emb_dim 维特征投影到词表大小，
        # bias=False 是 GPT-2 等模型的常见设定
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """前向传播：从 token id 序列计算下一个 token 的 logits 分布。

        Args:
            in_idx (torch.Tensor): 形状为 (batch_size, seq_len) 的整型张量，
                每个元素是词表中的 token id。

        Returns:
            torch.Tensor: 形状为 (batch_size, seq_len, vocab_size) 的 logits，
                表示模型在每个位置上对「下一个 token」的未归一化预测分数。
        """
        # batch_size: 批大小；seq_len: 当前输入序列长度（token 数）
        batch_size, seq_len = in_idx.shape
        # 查表得到每个 token 的嵌入向量
        # tok_embeds 形状: (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        # 生成位置索引 [0, 1, ..., seq_len-1]，查表得到位置嵌入
        # pos_embeds 形状: (seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 将 token 嵌入与位置嵌入相加（pos_embeds 在 batch 维度上广播），
        # 得到同时包含「内容信息」和「位置信息」的输入表示
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 对嵌入结果做 Dropout
        x = self.drop_emb(x)
        # 依次通过所有堆叠的 TransformerBlock，形状始终保持
        # (batch_size, seq_len, emb_dim)
        x = self.trf_blocks(x)
        # 最终层归一化
        x = self.final_norm(x)
        # 投影到词表维度，得到每个位置上对下一个 token 的预测 logits
        # logits 形状: (batch_size, seq_len, vocab_size)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """最朴素的自回归文本生成函数（贪心解码，greedy decoding）。

    每一步都：
        1. 把当前序列截断到模型支持的最大上下文长度；
        2. 用模型前向计算得到 logits；
        3. 只取最后一个时间步的 logits（代表「下一个 token」的预测）；
        4. 用 argmax 贪心地选出概率最大的 token id（不做采样/温度/top-k）；
        5. 把新 token 拼接到序列末尾，重复上述过程直到生成 max_new_tokens 个新 token。

    Args:
        model (nn.Module): 已训练（或未训练）的 GPT 模型，调用后返回 logits。
        idx (torch.Tensor): 形状为 (batch_size, num_tokens) 的初始 token id 序列
            （即当前上下文）。
        max_new_tokens (int): 要生成的新 token 数量。
        context_size (int): 模型支持的最大上下文长度，超出部分会被裁剪掉。

    Returns:
        torch.Tensor: 形状为 (batch_size, num_tokens + max_new_tokens) 的
            token id 序列，即原始上下文加上新生成的 token。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 只保留最近 context_size 个 token 作为模型输入，防止超出位置嵌入表范围
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # 推理阶段不需要计算梯度，节省显存并加速
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 只关心序列最后一个位置的预测结果，因为它对应「下一个 token」的分布
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 贪心选择：直接取 logits 最大值对应的词表索引，不做随机采样
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 把新生成的 token 拼接到序列末尾，作为下一轮迭代的输入
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

######################
# Bonus
######################
# 以下为「加速版」实现：功能与上面的手写版完全等价，
# 但改用 PyTorch 内置的 nn.LayerNorm、nn.GELU 以及
# scaled_dot_product_attention（在 ch03.py 的 PyTorchMultiHeadAttention 中调用），
# 从而可以利用底层优化的 CUDA 核（如 FlashAttention），在 GPU 上显著提速。


class FeedForwardFast(nn.Module):
    """FeedForward 的加速版本，使用 PyTorch 内置的 nn.GELU（tanh 近似）。

    结构与 FeedForward 完全相同：Linear -> GELU -> Linear，
    唯一区别是这里用 nn.GELU(approximate="tanh") 代替手写的 GELU 类，
    数值上与手写版等价，但通常执行效率更高。

    Args:
        cfg (dict): 模型配置字典，需包含键 "emb_dim"。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # 升维：emb_dim -> 4 * emb_dim
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # 使用 PyTorch 内置 GELU 的 tanh 近似实现，等价于上面手写的 GELU 类
            nn.GELU(approximate="tanh"),
            # 降维：4 * emb_dim -> emb_dim
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """前向传播，输入输出形状均为 (batch_size, num_tokens, emb_dim)。"""
        return self.layers(x)


class TransformerBlockFast(nn.Module):
    """TransformerBlock 的加速版本。

    与 TransformerBlock 的结构完全一致（Pre-LN + 残差连接），区别在于：
        - 注意力层改用 PyTorchMultiHeadAttention（内部调用
          torch.nn.functional.scaled_dot_product_attention，可自动启用
          FlashAttention 等高效实现）；
        - 前馈网络改用 FeedForwardFast（内置 GELU）；
        - LayerNorm 改用 PyTorch 内置的 nn.LayerNorm。

    Args:
        cfg (dict): 模型配置字典，需包含 "emb_dim"、"n_heads"、
            "drop_rate"、"qkv_bias" 等键。
    """
    def __init__(self, cfg):
        super().__init__()
        # 使用基于 scaled_dot_product_attention 的高效多头注意力实现
        self.att = PyTorchMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        # 使用内置 GELU 的前馈网络
        self.ff = FeedForwardFast(cfg)
        # 使用 PyTorch 内置的 LayerNorm 实现（数值上与手写 LayerNorm 类等价）
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        # 残差分支上的 Dropout
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """前向传播，结构与 TransformerBlock.forward 完全一致。

        Args:
            x (torch.Tensor): 形状为 (batch_size, num_tokens, emb_dim)。

        Returns:
            torch.Tensor: 形状与输入相同。
        """
        # Shortcut connection for attention block
        # 保存原始输入用于残差相加
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModelFast(nn.Module):
    """
    A faster variant of GPTModel optimized for training speed.

    This version is only marginally faster on CPU (~1.02x) but significantly
    faster on GPU (~2.05x) during training, thanks to optimized CUDA kernels
    and FlashAttention support.

    Key differences from the original GPTModel:
    1. Uses PyTorch's built-in LayerNorm instead of a custom implementation.
    2. Uses PyTorch's built-in GELU instead of a custom implementation.
    3. Uses PyTorch's scaled_dot_product_attention instead of a custom MultiHeadAttention.
    4. Automatically enables FlashAttention on compatible GPUs.

    中文说明：
        这是 GPTModel 的加速版本，架构与前向计算逻辑与 GPTModel 完全等价
        （token 嵌入 + 位置嵌入 -> Dropout -> 堆叠 TransformerBlockFast ->
        最终 LayerNorm -> 输出线性层），只是把手写的 LayerNorm/GELU/
        多头注意力替换为 PyTorch 内置的高效实现，因此在 CPU 上提速有限
        （约 1.02 倍），但在支持 FlashAttention 的 GPU 上可获得明显加速
        （约 2.05 倍），非常适合实际训练场景。
    """
    def __init__(self, cfg):
        super().__init__()
        # token 嵌入表，形状 (vocab_size, emb_dim)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入表，形状 (context_length, emb_dim)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠 n_layers 个加速版 TransformerBlockFast
        self.trf_blocks = nn.Sequential(
            *[TransformerBlockFast(cfg) for _ in range(cfg["n_layers"])])

        # 使用内置 LayerNorm 作为最终归一化层
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        # 输出线性层（语言模型头），无偏置
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """前向传播，逻辑与 GPTModel.forward 完全一致。

        Args:
            in_idx (torch.Tensor): 形状为 (batch_size, seq_len) 的 token id 序列。

        Returns:
            torch.Tensor: 形状为 (batch_size, seq_len, vocab_size) 的 logits。
        """
        # batch_size: 批大小；seq_len: 序列长度
        batch_size, seq_len = in_idx.shape
        # token 嵌入，形状 (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        # 位置嵌入，形状 (seq_len, emb_dim)，与 tok_embeds 相加时在 batch 维广播
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 融合内容信息与位置信息
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        # 依次通过所有加速版 TransformerBlock
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # 投影到词表大小，得到预测 logits
        logits = self.out_head(x)
        return logits
