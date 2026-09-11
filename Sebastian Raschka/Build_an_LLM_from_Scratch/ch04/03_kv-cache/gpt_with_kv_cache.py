# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

# ============================================================
# 中文说明（模块级 docstring）
# ------------------------------------------------------------
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
# 一书中第 3~4 章代码的汇总版本，并在其基础上加入了 **KV 缓存（KV Cache）**
# 这一推理加速技巧（对应 ch04/03_kv-cache 目录）。
#
# 文件中包含的核心内容：
#   1. Chapter 3：多头自注意力机制 MultiHeadAttention（因果掩码 + 缩放点积注意力）。
#   2. Chapter 4：LayerNorm、GELU 激活函数、FeedForward 前馈网络、
#      TransformerBlock（残差连接 + 注意力 + 前馈网络）、GPTModel（完整的 GPT 结构）。
#   3. 文本生成函数：
#        - generate_text_simple：朴素的逐 token 自回归生成（每步都要把
#          全部历史 token 重新过一遍模型，效率较低）。
#        - generate_text_simple_cached：使用 KV 缓存加速的生成函数，
#          每一步只需要把“新产生的这一个 token”喂给模型，历史的 Key/Value
##         张量被缓存下来复用，从而把自回归生成的时间复杂度从 O(n^2) 降到 O(n)。
#
# 关于 KV 缓存的核心思想：
#   在自回归生成时，每一步都要计算 Query 与所有历史 token 的 Key 做注意力，
#   并加权求和所有历史 token 的 Value。如果不做缓存，每生成一个新 token，
#   就要把“历史 + 新 token”整体重新输入模型，重复计算所有历史 token 的
#   Key/Value 投影，浪费大量算力。KV 缓存的做法是：把每一层注意力模块中
#   已经算好的 Key、Value 张量保存（缓存）起来，后续步骤只计算“新 token”
#   对应的 Key/Value，并将其拼接（concat）到缓存后面，从而避免重复计算。
#   本文件中，代码里标注了 "NEW" 的部分就是相对于普通（无缓存）版本新增的
#   KV 缓存实现逻辑，读者可以对照普通版本理解其中的差异。
# ============================================================

import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """多头自注意力（Multi-Head Self-Attention）模块，支持可选的 KV 缓存。

    这是 Transformer 的核心组件：将输入序列通过线性层投影为 Query/Key/Value，
    拆分成多个“头”（head）分别做缩放点积注意力（并施加因果掩码，保证每个
    位置只能看到自己及之前的 token），再把各头的输出拼接、投影回原始维度。

    参数：
        d_in (int): 输入特征维度（即输入 token 向量的维度）。
        d_out (int): 输出特征维度，同时也是所有头拼接后的总维度。
        context_length (int): 支持的最大上下文长度，用于构造因果掩码矩阵。
        dropout (float): 注意力权重上的 dropout 比例。
        num_heads (int): 注意力头的数量，d_out 必须能被 num_heads 整除。
        qkv_bias (bool): Query/Key/Value 线性层是否使用偏置项。

    关键张量形状（b=batch_size, num_tokens=当前序列长度）：
        输入 x: (b, num_tokens, d_in)
        输出 context_vec: (b, num_tokens, d_out)
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 中文：把总输出维度 d_out 平均分给每个头，每个头只负责 head_dim 维的子空间，
        # 这样多个头可以并行学习到不同的注意力模式（相当于“多套视角”）。

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )
        # 中文：mask 是一个上三角矩阵（对角线以上为 1），用来实现“因果掩码”
        # （causal mask）——即第 i 个位置只能看到 <= i 的位置，看不到未来的 token。
        # persistent=False 表示这个 buffer 不会被保存进 state_dict（它是可以按需
        # 重新生成的固定值，不属于模型学到的参数）。

        ####################################################
        # NEW
        # 中文：以下三个属性是为 KV 缓存新增的状态变量。
        # cache_k / cache_v：分别缓存历史所有 token 计算出来的 Key / Value 张量。
        # 初始为 None，表示还没有缓存内容（对应"提示词/prompt"还没输入的状态）。
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        # ptr_current_pos：记录当前已经处理到序列的第几个位置，用于在使用缓存时
        # 从 self.mask 中裁剪出正确的因果掩码切片（因为此时 Query 的位置不再从 0
        # 开始，而是接着上一次缓存的位置继续往后走）。
        self.ptr_current_pos = 0
        ####################################################

    def forward(self, x, use_cache=False):
        """前向传播：计算带因果掩码的多头自注意力输出。

        参数：
            x (Tensor): 输入张量，形状 (b, num_tokens, d_in)。
                - 若 use_cache=False：num_tokens 通常是整段序列长度。
                - 若 use_cache=True：num_tokens 可以只是"新增的 token"数量
                  （比如生成阶段每步只传入 1 个新 token）。
            use_cache (bool): 是否启用 KV 缓存机制。

        返回：
            context_vec (Tensor): 注意力输出，形状 (b, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)
        # 中文：这里的 keys_new / values_new 只是“本次前向传播新算出来的”
        # Key/Value，注意变量名带 "_new" 后缀，是为了和后面拼接后的完整
        # keys/values（可能包含历史缓存）区分开。

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 中文：把最后一维 d_out 拆分成 (num_heads, head_dim) 两维，
        # 这样每个头可以独立地在自己的子空间里计算注意力。

        ####################################################
        # NEW
        # 中文：KV 缓存的核心逻辑——
        # 如果启用缓存且这是第一次调用（cache_k 为 None，通常对应"喂入 prompt"这一步），
        # 就直接把这次算出来的 keys_new/values_new 作为缓存的初始值；
        # 否则（已经有缓存了，通常对应"生成第 N 个新 token"这一步），
        # 就把新算出来的 Key/Value 沿着序列长度维度（dim=1）拼接到旧缓存后面，
        # 这样 self.cache_k/self.cache_v 里始终保存着"迄今为止全部 token"的 Key/Value，
        # 而不需要重新计算历史 token 的投影，节省了大量重复计算。
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            # 不使用缓存时，keys/values 就是本次输入序列算出来的全部 Key/Value，
            # 每次调用都要重新计算，不做任何复用（这是"朴素/无缓存"版本的行为）。
            keys, values = keys_new, values_new
        ####################################################

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # 中文：把 num_heads 维提到前面，方便后续按批量矩阵乘法（bmm）的方式
        # 对每个头分别做注意力计算（PyTorch 的 @ 运算符会对最后两维做矩阵乘法，
        # 并在前面的维度上自动广播/批处理）。

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # 中文：queries 形状 (b, num_heads, num_tokens_Q, head_dim)，
        # keys.transpose(2,3) 形状 (b, num_heads, head_dim, num_tokens_K)，
        # 相乘后 attn_scores 形状为 (b, num_heads, num_tokens_Q, num_tokens_K)，
        # 表示每个 Query 位置对每个 Key 位置的注意力得分（未归一化）。
        # 注意：当使用 KV 缓存时，num_tokens_K（键的数量，等于历史+新 token 总数）
        # 通常会大于 num_tokens_Q（本次前向传播只有新 token 作为 Query）。

        ####################################################
        # NEW
        # 中文：根据是否使用缓存，选取不同方式裁剪因果掩码 mask_bool。
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        if use_cache:
            # 使用缓存时，Query 对应的是"新 token"，它们在整个序列中的绝对位置
            # 并不是从 0 开始，而是从 self.ptr_current_pos 开始（即之前已经处理
            # 过的 token 数量）。因此要从完整的因果掩码矩阵中，取出
            # [ptr_current_pos : ptr_current_pos+num_tokens_Q] 行、
            # [0 : num_tokens_K] 列这一小块，才能保证"新 token 只能看到自己
            # 及之前所有 token（含缓存的历史 token）"这一因果约束依然成立。
            mask_bool = self.mask.bool()[
                self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
            ]
            # 处理完这一批新 token 后，把指针往后移动，为下一次调用做准备。
            self.ptr_current_pos += num_tokens_Q
        ####################################################
        # Original mask truncated to the number of tokens and converted to boolean
        else:
            # 不使用缓存时，Query 和 Key 都是同一段完整序列，位置从 0 开始，
            # 直接取左上角 (num_tokens_Q, num_tokens_K) 的子矩阵即可。
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 中文：把掩码中为 True（即"未来"位置，不应该被看到）的注意力得分
        # 设为负无穷，这样后续 softmax 后这些位置的权重会变成 0，
        # 从而实现"因果性"——每个位置只能关注它自己和它之前的 token。

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 中文：除以 sqrt(head_dim) 做缩放（scaled dot-product attention 中的
        # "scaled"），避免点积数值过大导致 softmax 梯度消失；
        # 沿最后一维（Key 的维度）做 softmax，得到归一化的注意力权重。
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 中文：attn_weights (b, num_heads, num_tokens_Q, num_tokens_K) 与
        # values (b, num_heads, num_tokens_K, head_dim) 相乘，得到
        # (b, num_heads, num_tokens_Q, head_dim)，即每个 Query 位置对所有
        # Value 的加权求和结果；再转置回 (b, num_tokens_Q, num_heads, head_dim)。

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 中文：把各个头的输出重新拼接（reshape）回 (b, num_tokens, d_out)，
        # 相当于把多个"视角"的信息合并到一起。
        context_vec = self.out_proj(context_vec)  # optional projection
        # 中文：再经过一个线性层做一次"融合"投影，让不同头的信息可以互相交流。

        return context_vec

    ####################################################
    # NEW
    def reset_cache(self):
        """清空 KV 缓存并重置位置指针。

        中文说明：在开始处理一个全新的序列（比如新的一次生成任务）之前，
        必须调用该方法把上一次生成留下的缓存（cache_k/cache_v）和位置指针
        （ptr_current_pos）清空，否则新序列的注意力计算会错误地"看到"
        上一次生成任务残留的历史 Key/Value。
        """
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0
    ####################################################


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）模块。

    对每个 token 向量（最后一维，即 emb_dim 维）分别做归一化，使其均值为 0、
    方差为 1，再通过可学习的缩放参数 scale 和平移参数 shift 做仿射变换。
    这样可以稳定训练、加速收敛。

    参数：
        emb_dim (int): 需要归一化的特征维度（嵌入维度）。

    输入/输出形状均为 (..., emb_dim)，例如 (batch, seq_len, emb_dim)。
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 中文：防止分母为 0 的小常数（数值稳定性）
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 中文：可学习的缩放参数 γ
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 中文：可学习的平移参数 β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 中文：沿着最后一维（emb_dim）计算均值和方差，
        # unbiased=False 表示使用有偏估计（除以 N 而非 N-1），
        # 这与深度学习框架中常见的 LayerNorm 实现保持一致。
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 中文：标准化：减均值除以标准差，使得每个 token 向量在特征维度上
        # 均值为 0、方差为 1。
        return self.scale * norm_x + self.shift
        # 中文：再做一次仿射变换（缩放+平移），让模型有能力学习恢复
        # 归一化前的分布（如果这对任务更有利的话）。


class GELU(nn.Module):
    """GELU（Gaussian Error Linear Unit）激活函数，使用 tanh 近似实现。

    相比 ReLU，GELU 是平滑、非单调的激活函数，在 Transformer / GPT 系列
    模型中被广泛使用。这里使用的是论文中给出的 tanh 近似公式，而非
    精确的高斯误差函数形式。

    输入/输出形状相同，例如 (batch, seq_len, hidden_dim)。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        # 中文：GELU 的 tanh 近似公式：
        # GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
        # 该公式在数值上非常接近真实的 GELU（基于标准正态分布累积分布函数），
        # 但计算效率更高，常用于工程实现中。


class FeedForward(nn.Module):
    """前馈网络（Position-wise Feed-Forward Network）。

    Transformer Block 中的第二个子层：对每个位置（token）独立地做
    "升维 -> 激活 -> 降维" 的两层全连接变换，用于增强模型的非线性表达能力。

    参数：
        cfg (dict): 配置字典，需要包含 "emb_dim" 键（嵌入维度）。

    形状变化：(b, seq_len, emb_dim) -> (b, seq_len, 4*emb_dim) -> (b, seq_len, emb_dim)
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 中文：升维到 4 倍，扩大模型容量
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 中文：再降维回原始 emb_dim
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer 块：一个完整的"注意力子层 + 前馈子层"，均带残差连接和前置归一化。

    结构（Pre-LayerNorm 风格）：
        x -> LayerNorm -> 多头注意力 -> Dropout -> 残差相加
          -> LayerNorm -> 前馈网络   -> Dropout -> 残差相加

    参数：
        cfg (dict): 配置字典，需包含 emb_dim / context_length / n_heads /
            drop_rate / qkv_bias 等键。

    输入/输出形状均为 (batch_size, num_tokens, emb_dim)。
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

    def forward(self, x, use_cache=False):
        """前向传播。

        参数：
            x (Tensor): 输入，形状 (batch_size, num_tokens, emb_dim)。
            use_cache (bool): 是否启用注意力层的 KV 缓存（会透传给内部的
                MultiHeadAttention）。

        返回：
            x (Tensor): 输出，形状与输入相同 (batch_size, num_tokens, emb_dim)。
        """
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        # 中文：Pre-LN 结构——先归一化，再进入注意力子层，
        # 这样比 Post-LN（先算子层再归一化）在深层网络中训练更稳定。

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        # NEW
        # 中文：把 use_cache 参数透传给多头注意力层，
        # 由它决定是否读写 KV 缓存。
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：残差连接（Residual Connection）——把注意力子层的输出加回
        # 原始输入，缓解深层网络中的梯度消失问题，也让模型可以"选择性"地
        # 利用新计算出的信息。

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 中文：前馈子层同样采用"归一化 -> 变换 -> Dropout -> 残差相加"的模式。

        return x


class GPTModel(nn.Module):
    """完整的 GPT 模型：词嵌入 + 位置嵌入 + 多层 TransformerBlock + 输出头。

    参数：
        cfg (dict): 模型配置，需包含以下键：
            vocab_size (int): 词表大小。
            context_length (int): 支持的最大上下文长度。
            emb_dim (int): 嵌入维度。
            n_heads (int): 注意力头数。
            n_layers (int): TransformerBlock 层数。
            drop_rate (float): dropout 比例。
            qkv_bias (bool): QKV 线性层是否使用偏置。

    输入：in_idx，形状 (batch_size, seq_len)，元素为词表中的 token id。
    输出：logits，形状 (batch_size, seq_len, vocab_size)，
        表示每个位置对词表中每个 token 的预测得分（未做 softmax）。
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 中文：tok_emb 把 token id 映射为向量（词嵌入）；
        # pos_emb 是可学习的绝对位置嵌入，为模型提供"顺序"信息
        # （因为自注意力机制本身不具备处理序列顺序的能力）。
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        # NEW
        # 中文：原版用 nn.Sequential 顺序堆叠各层，但 nn.Sequential.forward()
        # 只支持单一输入，无法传递 use_cache 这个额外参数。
        # 这里改用 nn.ModuleList，手动在 forward 里用 for 循环遍历每一层，
        # 从而可以把 use_cache 参数逐层透传下去。
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # current_pos：记录当前已经生成/处理到序列中的第几个绝对位置，
        # 用于在使用 KV 缓存生成时，为"新 token"计算正确的位置嵌入
        # （因为新 token 在整个序列中的位置不是从 0 开始的）。
        self.current_pos = 0
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 中文：最终归一化后，用一个线性层把 emb_dim 维的隐藏状态映射到
        # vocab_size 维的 logits，用于预测下一个 token 的概率分布。

    def forward(self, in_idx, use_cache=False):
        """前向传播，计算下一个 token 的预测 logits。

        参数：
            in_idx (Tensor): 输入 token id 序列，形状 (batch_size, seq_len)。
                - use_cache=False 时通常是完整序列；
                - use_cache=True 时可以只传入"新 token"（比如逐 token 生成时
                  每步只传 1 个 token）。
            use_cache (bool): 是否启用 KV 缓存（会逐层透传给 TransformerBlock）。

        返回：
            logits (Tensor): 形状 (batch_size, seq_len, vocab_size)。
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 中文：tok_embeds 形状 (batch_size, seq_len, emb_dim)。

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        # NEW
        # 中文：位置编码需要区分"是否使用缓存"两种情况——
        if use_cache:
            # 使用缓存时，这次传入的 seq_len 个 token 是接在历史 token 之后的，
            # 它们的绝对位置应该是 [current_pos, current_pos+seq_len)，
            # 而不是从 0 开始，否则位置编码会和历史 token 重复/错乱。
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            # 处理完这批 token 后，把全局位置指针向前推进 seq_len，
            # 供下一次调用（比如生成下一个 token 时）使用。
            self.current_pos += seq_len
        else:
            # 不使用缓存时，每次都是从头开始处理完整序列，位置固定从 0 开始。
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        # 中文：pos_embeds 形状先是 (seq_len, emb_dim)，unsqueeze(0) 后变成
        # (1, seq_len, emb_dim)，以便与 tok_embeds 的 (batch_size, seq_len, emb_dim)
        # 通过广播（broadcasting）相加。
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 中文：词嵌入 + 位置嵌入，得到每个 token 的初始表示。
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # NEW
        # 中文：手动逐层遍历 ModuleList 中的每个 TransformerBlock，
        # 并把 use_cache 参数透传下去，让每一层的注意力模块都能正确地
        # 读写自己独立的 KV 缓存（每一层的 cache_k/cache_v 是各自独立的）。
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        # 中文：logits 形状 (batch_size, seq_len, vocab_size)，
        # 沿最后一维做 argmax/softmax 即可得到每个位置预测的下一个 token。
        return logits

    ####################################################
    # NEW
    def reset_kv_cache(self):
        """重置模型中所有层的 KV 缓存，以及全局位置指针 current_pos。

        中文说明：在开始一次全新的生成任务之前必须调用此方法，
        否则残留的历史缓存和位置指针会污染新序列的注意力计算结果
        （比如错误地"看到"上一次生成任务的 token，或者位置编码从错误的
        位置开始）。
        """
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """朴素的自回归文本生成函数（不使用 KV 缓存）。

    每生成一个新 token，都要把"截断后的完整历史序列"重新输入模型做一次
    完整的前向传播，因此计算复杂度会随生成长度的增加而显著上升
    （历史越长，重复计算的开销越大）。这里是与 generate_text_simple_cached
    做效率对比的基线版本。

    参数：
        model: GPTModel 实例。
        idx (Tensor): 初始上下文的 token id，形状 (batch_size, T)。
        max_new_tokens (int): 要生成的新 token 数量。
        context_size (int): 模型支持的最大上下文长度，用于裁剪过长的历史。

    返回：
        idx (Tensor): 拼接了新生成 token 之后的完整序列，
            形状 (batch_size, T + max_new_tokens)。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        # 中文：只保留最近 context_size 个 token 作为上下文输入，
        # 防止序列长度超出模型支持的最大上下文长度。

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        # 中文：每次都对"完整的（截断后的）历史序列"做一次前向传播，
        # 没有复用任何之前计算过的 Key/Value，因此存在大量重复计算。

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]
        # 中文：只关心序列最后一个位置的输出，因为这才是"预测下一个 token"
        # 所需要的 logits。

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        # 中文：贪心采样（greedy decoding）——直接取概率（logits）最大的
        # token 作为下一个预测结果，不做随机采样。

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
        # 中文：把新生成的 token 拼接到序列末尾，供下一轮迭代使用。

    return idx


####################################################
# NEW
def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    """使用 KV 缓存加速的自回归文本生成函数。

    与 generate_text_simple 的核心区别：
        - 先用完整的 prompt 做一次前向传播，把每一层的 Key/Value 缓存下来；
        - 之后每生成一个新 token，只需要把"这一个新 token"喂给模型
          （而不是整个历史序列），模型内部会自动把新 token 的 Key/Value
          拼接到缓存后面，从而避免了对历史 token 的重复计算。
    这使得生成的时间复杂度从"朴素版本"的 O(n^2)（每步都要重新处理全部历史）
    降低到接近 O(n)（每步只处理 1 个新 token），显著提升长序列生成速度。

    参数：
        model: GPTModel 实例。
        idx (Tensor): 初始上下文（prompt）的 token id，形状 (batch_size, T)。
        max_new_tokens (int): 要生成的新 token 数量。
        context_size (int, optional): 支持的最大上下文长度；
            若为 None，则使用 model.pos_emb.num_embeddings（即模型配置中的
            context_length）。
        use_cache (bool): 是否真正启用 KV 缓存；设为 False 时退化为
            "每步重新输入完整历史"的朴素方式（但仍复用这同一个函数接口，
            便于做 A/B 对比测试）。

    返回：
        idx (Tensor): 拼接了新生成 token 之后的完整序列，
            形状 (batch_size, T + max_new_tokens)。
    """
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings
    # 中文：如果调用者没有显式传入 context_size，就用模型配置里的
    # context_length（即位置嵌入表的行数）作为默认的最大上下文长度。

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            # 中文：开始一次新的生成任务前，必须先清空可能残留的旧缓存，
            # 否则新一轮生成会错误地"继承"上一次调用留下的历史 Key/Value。
            model.reset_kv_cache()
            # 中文：用完整的 prompt（裁剪到 ctx_len 长度以内）做一次前向传播，
            # 这一步会把 prompt 中每个 token 在每一层的 Key/Value 都计算并缓存下来。
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # 中文：取上一步 logits 最后一个位置（即"下一个 token"的预测分布）
                # 中概率最大的 token id，作为本轮新生成的 token（贪心解码）。
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # 中文：把新 token 拼接到完整序列 idx 上（这是最终要返回的结果）。
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
                # 中文：关键区别所在——这里只把"刚生成的这 1 个新 token"喂给模型，
                # 而不是像朴素版本那样把整个历史序列都重新输入一遍。模型内部
                # 的每一层注意力会自动把这个新 token 的 Key/Value 拼接到各自
                # 缓存的历史 Key/Value 后面，用完整的历史信息计算注意力，
                # 但只需要为这 1 个新 token 做投影计算，从而大幅减少重复计算量。
        else:
            # 中文：不使用缓存的分支，行为与 generate_text_simple 基本一致——
            # 每一步都把（裁剪后的）完整历史序列重新输入模型。保留这个分支
            # 主要是为了方便在同一个函数里对比"有无 KV 缓存"的生成速度差异。
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
####################################################


def main():
    # 中文：GPT-2 124M（"small"）模型的超参数配置。
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)  # 中文：固定随机种子，保证模型参数初始化可复现
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # disable dropout
    # 中文：切换到 eval 模式，关闭 dropout（推理阶段不需要随机丢弃神经元）。

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    # 中文：把文本编码为 token id 列表，再转成形状 (1, T) 的张量
    # （batch_size=1，因为只有一条输入序列）。

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 中文：确保之前的 GPU 操作全部完成，计时才准确
    start = time.time()

    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=encoded_tensor,
    #     max_new_tokens=200,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )

    ####################################################
    # NEW
    # 中文：使用带 KV 缓存的生成函数来生成 200 个新 token，
    # 相比上面被注释掉的朴素版本 generate_text_simple，速度应明显更快。
    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )
    ####################################################

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start
    # 中文：计算整个生成过程耗费的墙钟时间（wall-clock time）。

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    # 中文：把生成的 token id 序列解码回可读文本。

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    # 中文：打印生成速度（每秒生成的 token 数），用于直观感受 KV 缓存
    # 带来的推理加速效果。
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")
        # 中文：打印 GPU 峰值显存占用，KV 缓存虽然加速了计算，
        # 但也会额外占用显存（缓存了所有历史 token 的 Key/Value），
        # 这是用"空间换时间"的典型权衡。


if __name__ == "__main__":
    main()
