# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4.
# This file can be run as a standalone script.

"""
【模块级中文说明】
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书中
第 3 章(多头自注意力 MultiHeadAttention)与第 4 章(GPT 完整模型结构)代码的汇总版本。

在本章所在目录 `ch04/03_kv-cache` 中，本文件 `gpt_ch04.py` 扮演的角色是
**「未使用 KV 缓存(KV cache)的基线(baseline)实现」**：
    - 它的 `generate_text_simple` 每生成一个新 token，都会把「到目前为止的完整序列」
      重新丢进模型做一次完整的前向传播(forward pass)，也就是说前面已经算过的
      Key/Value 会被重复计算，存在大量冗余计算。
    - 该目录下通常还会配套一个「引入了 KV 缓存」的版本(例如 gpt_with_kv_cache.py)，
      通过缓存历史 token 的 Key/Value 张量，使得每步生成时只需要对「最新的 1 个 token」
      做前向计算，从而大幅提升自回归(autoregressive)生成的推理速度。
    - 本文件常被用作性能对比的基准(baseline)，配合 `main()` 函数中的计时逻辑
      (tokens/sec、显存占用)与 KV 缓存版本进行对比，直观展示 KV 缓存带来的加速效果。

阅读顺序建议：
    1. MultiHeadAttention  —— 第 3 章内容，多头自注意力机制的实现。
    2. LayerNorm / GELU / FeedForward / TransformerBlock —— 第 4 章内容，
       构成 Transformer Block 的各个子模块。
    3. GPTModel —— 把 Embedding、多个 TransformerBlock、输出头组装成完整的 GPT 模型。
    4. generate_text_simple —— 最朴素的自回归文本生成循环(不带 KV 缓存)。
    5. main —— 构造一个 124M 参数规模的 GPT-2 配置，实例化模型并做一次文本生成，
       同时统计耗时与吞吐量(tokens/sec)。
"""

import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    """
    多头自注意力(Multi-Head Self-Attention)模块。

    作用:
        对输入序列做「带因果掩码(causal mask)的缩放点积注意力」，并行地在多个
        「注意力头(head)」上分别计算注意力，再把各头的结果拼接、投影回原始维度。
        因果掩码保证第 t 个位置只能看到 <= t 的历史信息，这是 GPT 这类自回归
        语言模型能够做「预测下一个词」训练与推理的关键。

    构造参数:
        d_in (int): 输入特征维度(即每个 token 的 embedding 维度)。
        d_out (int): 输出特征维度，同时也是 Q/K/V 投影后的总维度
                     (会被均分给每个注意力头，需满足 d_out % num_heads == 0)。
        context_length (int): 支持的最大上下文长度，用于预先构造因果掩码矩阵。
        dropout (float): 注意力权重上的 dropout 比例。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool): Q/K/V 三个线性层是否使用偏置(bias)，默认为 False。

    forward 输入/输出形状:
        输入 x: (batch, num_tokens, d_in)
        输出 context_vec: (batch, num_tokens, d_out)
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
        # 每个头分到的维度 = 总输出维度 / 头数，这样多头拼接回去后维度仍然是 d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 三个独立的线性层，分别把输入 x 投影成 Query、Key、Value
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        # 多头拼接之后再做一次线性投影，让模型学习如何融合各头信息
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )
        # 预先构造一个上三角(不含对角线)的 0/1 掩码矩阵，形状 (context_length, context_length)
        # triu(diagonal=1) 表示严格上三角部分为 1，用来标记「未来位置」，后面会用它屏蔽掉
        # 当前位置看不到的未来 token，从而实现「因果注意力」(causal attention)。
        # persistent=False 表示这个 buffer 不会被保存进 state_dict，因为它只是常量掩码。

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # x 形状: (batch, num_tokens, d_in)

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values = self.W_value(x)
        queries = self.W_query(x)
        # 分别线性投影得到 Q/K/V，形状均为 (b, num_tokens, d_out)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 把最后一维 d_out 拆分成 (num_heads, head_dim)，
        # 这样就把「一个大的注意力」隐式地拆成了「num_heads 个并行的小注意力」

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # 交换 num_tokens 和 num_heads 维度，使得 num_heads 排在 batch 之后，
        # 方便后续按 head 维度做批量矩阵乘法(bmm)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        # queries: (b, num_heads, num_tokens, head_dim)
        # keys.transpose(2,3): (b, num_heads, head_dim, num_tokens)
        # 矩阵乘法结果 attn_scores 形状: (b, num_heads, num_tokens, num_tokens)
        # 即每个 head 内部，每个 query 位置对每个 key 位置的原始注意力分数

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 因为预构造的掩码是按最大 context_length 生成的，这里截取当前实际序列长度对应的子矩阵

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 把「未来位置」对应的注意力分数设为 -inf，
        # 这样经过 softmax 后这些位置的权重会变成 0，即无法看到未来信息(因果性)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 除以 sqrt(head_dim) 做缩放(scaled dot-product)，防止点积数值过大导致 softmax 梯度消失
        # 在最后一维(key 的位置维度)上做 softmax，得到归一化的注意力权重
        attn_weights = self.dropout(attn_weights)
        # 对注意力权重做 dropout，训练时随机丢弃部分连接，起正则化作用

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # attn_weights: (b, num_heads, num_tokens, num_tokens)
        # values: (b, num_heads, num_tokens, head_dim)
        # 加权求和后得到 (b, num_heads, num_tokens, head_dim)，
        # 再 transpose(1,2) 换回 (b, num_tokens, num_heads, head_dim)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 把多头拼接回单一维度: (b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
        # contiguous() 是因为前面的 transpose 会让张量在内存中不连续，view 需要连续内存
        context_vec = self.out_proj(context_vec)  # optional projection
        # 对拼接后的多头输出再做一次线性变换，融合各头信息

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    """
    层归一化(Layer Normalization)模块。

    作用:
        对每个 token 的特征向量(最后一维)做归一化，使其均值为 0、方差为 1，
        再通过可学习的缩放(scale)和平移(shift)参数还原表达能力。
        LayerNorm 能稳定深层网络的训练，缓解梯度爆炸/消失问题。

    构造参数:
        emb_dim (int): 特征(embedding)维度，即需要归一化的最后一维大小。

    forward 输入/输出形状:
        输入 x: (..., emb_dim)，通常是 (batch, num_tokens, emb_dim)
        输出: 形状与输入相同
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # 防止除以 0 的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        # 可学习的缩放和平移参数，分别初始化为全 1 和全 0(相当于初始时不改变归一化结果)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # 在最后一维(特征维)上求均值，keepdim=True 保持维度便于广播
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 在最后一维上求方差；unbiased=False 表示使用有偏估计(除以 N 而非 N-1)，
        # 与常见深度学习框架的 LayerNorm 实现保持一致
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 标准化: 减均值、除以标准差(加 eps 防止数值不稳定)
        return self.scale * norm_x + self.shift
        # 用可学习参数对归一化后的结果做仿射变换，恢复模型表达能力


class GELU(nn.Module):
    """
    GELU(Gaussian Error Linear Unit)激活函数模块，这里实现的是其 tanh 近似版本。

    作用:
        作为 FeedForward 中的非线性激活函数，相比 ReLU 更平滑，
        在 Transformer 类模型中被广泛使用(GPT-2/GPT-3 等)。

    forward 输入/输出形状:
        输入/输出形状相同，逐元素(element-wise)计算。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        # GELU 的 tanh 近似公式:
        # GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        # 相比精确的 GELU(基于误差函数 erf 计算)，这种近似计算更快，且在实践中效果相近


class FeedForward(nn.Module):
    """
    前馈网络(Feed-Forward Network, FFN)模块，即 Transformer Block 中的 MLP 子层。

    作用:
        对每个 token 的表示独立地做「升维 -> 非线性激活 -> 降维」的两层全连接变换，
        为模型增加非线性表达能力。按照 GPT-2 的设计，隐藏层维度是输入维度的 4 倍。

    构造参数:
        cfg (dict): 模型配置字典，这里用到 cfg["emb_dim"](embedding 维度)。

    forward 输入/输出形状:
        输入 x: (batch, num_tokens, emb_dim)
        输出: (batch, num_tokens, emb_dim)  (维度不变，只是中间升维到 4*emb_dim)
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # 升维: emb_dim -> 4*emb_dim，扩大特征空间以增强表达能力
            GELU(),
            # 非线性激活
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            # 降维: 4*emb_dim -> emb_dim，投影回原始维度以便与残差连接相加
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Transformer 块(Block)，是构成 GPT 模型的基本重复单元。

    作用:
        由「多头自注意力子层」和「前馈网络子层」组成，每个子层前面都先做
        LayerNorm(Pre-LayerNorm 结构)，子层输出再通过 dropout 后与残差
        (shortcut)相加，形成 GPT-2 风格的 Transformer Block。

    构造参数:
        cfg (dict): 模型配置字典，包含 emb_dim、context_length、n_heads、
                    drop_rate、qkv_bias 等键。

    forward 输入/输出形状:
        输入 x: (batch, num_tokens, emb_dim)
        输出: (batch, num_tokens, emb_dim)  (维度不变，便于堆叠多个 Block)
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
        # 自注意力子层：输入输出维度相同，均为 emb_dim
        self.ff = FeedForward(cfg)
        # 前馈网络子层
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 两个 LayerNorm，分别用在注意力子层和前馈子层之前(Pre-Norm 结构)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        # 用在残差分支上的 dropout

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        # 保存输入，用于后面的残差连接(residual connection)
        x = self.norm1(x)
        # Pre-LayerNorm: 先归一化，再送入注意力层(与原始 Transformer 的 Post-Norm 不同)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 残差相加：缓解深层网络训练时的梯度消失问题，让梯度可以直接回传

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        # 前馈子层同样使用 Pre-Norm + 残差连接的结构

        return x


class GPTModel(nn.Module):
    """
    完整的 GPT 语言模型。

    作用:
        将「token embedding + 位置 embedding」、多个堆叠的 TransformerBlock、
        最终的 LayerNorm 以及输出投影层(将隐藏状态映射回词表大小的 logits)
        组装成一个完整的自回归语言模型。

    构造参数:
        cfg (dict): 模型配置字典，包含:
            - vocab_size: 词表大小
            - context_length: 最大上下文长度(位置编码的最大长度)
            - emb_dim: embedding / 隐藏层维度
            - n_heads: 注意力头数
            - n_layers: TransformerBlock 堆叠层数
            - drop_rate: dropout 比例
            - qkv_bias: Q/K/V 投影是否使用 bias

    forward 输入/输出形状:
        输入 in_idx: (batch_size, seq_len)，为 token id 序列(整数张量)
        输出 logits: (batch_size, seq_len, vocab_size)，每个位置对下一个 token 的预测分布(未归一化)
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # token embedding 表：把每个 token id 映射为 emb_dim 维的向量
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 可学习的绝对位置编码表：为序列中的每个位置(0, 1, 2, ...)分配一个 emb_dim 维向量
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 对 token+位置 embedding 之和做 dropout

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # 堆叠 n_layers 个 TransformerBlock，构成模型的主体深度

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 最后再做一次 LayerNorm，稳定输出分布
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # 输出投影层(也叫 language modeling head)，把隐藏状态映射到词表维度，得到每个 token 的 logits

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # in_idx 形状: (batch_size, seq_len)，每个元素是词表中的 token id
        tok_embeds = self.tok_emb(in_idx)
        # 查表得到 token embedding，形状: (batch_size, seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 生成位置索引 [0, 1, ..., seq_len-1]，再查表得到位置 embedding，
        # 形状: (seq_len, emb_dim)，会自动广播到 batch 维
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # 词向量与位置向量相加，融合「词的语义信息」与「位置信息」
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        # 依次通过 n_layers 个 TransformerBlock 提取上下文表示，形状保持 (batch_size, seq_len, emb_dim)
        x = self.final_norm(x)
        logits = self.out_head(x)
        # 最终投影到词表维度: (batch_size, seq_len, vocab_size)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    最朴素(不带 KV 缓存)的自回归文本生成函数。

    作用:
        循环 max_new_tokens 次，每次都把「当前已生成的完整序列(裁剪到 context_size 以内)」
        重新输入模型做一次完整前向传播，取最后一个位置的 logits，贪心(argmax)选出
        概率最高的下一个 token，拼接到序列末尾，如此循环直到生成指定数量的新 token。

        注意：这种实现方式在每一步都会「重复计算」前面所有 token 的 Key/Value，
        计算量随生成长度增长呈近似平方级增长，是本文件作为 KV 缓存对比基线
        (baseline)的核心原因——KV 缓存版本会把历史 Key/Value 缓存下来，
        每步只需计算新 token 的 Key/Value，从而显著提速。

    参数:
        model: GPTModel 实例(或兼容接口的模型)。
        idx (Tensor): 初始 token id 序列，形状 (batch, num_tokens)。
        max_new_tokens (int): 需要生成的新 token 数量。
        context_size (int): 模型支持的最大上下文长度，用于裁剪过长的输入。

    返回:
        Tensor: 生成后的完整 token id 序列，形状 (batch, num_tokens + max_new_tokens)。
    """
    model.eval()
    # 切换到 eval 模式，关闭 dropout 等训练专用行为
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        # 只保留最近 context_size 个 token 作为模型输入，避免超出位置编码的最大长度

        # Get the predictions
        with torch.no_grad():
            # 推理阶段不需要计算梯度，节省显存和计算量
            logits = model(idx_cond)
            # 【无 KV 缓存的关键点】这里每次都是对 idx_cond 的「完整序列」重新做前向传播，
            # 也就是说前面已经算过的 token 的 Key/Value 会被重复计算，存在冗余

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]
        # 只关心「最后一个位置」对下一个 token 的预测(因为这才是要生成的新 token)

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)
        # 贪心解码(greedy decoding)：直接取 logits 最大值对应的 token id，不做采样

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
        # 把新生成的 token 拼接到序列末尾，作为下一轮迭代的输入

    return idx


def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }
    # 对应 GPT-2 small(124M 参数)的标准配置

    torch.manual_seed(123)
    # 固定随机种子，保证参数初始化可复现
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 优先使用 GPU，若不可用则退回 CPU
    model.to(device)
    model.eval()  # disable dropout
    # 推理模式，关闭 dropout，保证生成结果确定(在固定种子下)

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    # 使用 GPT-2 的 BPE 分词器
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    # unsqueeze(0) 增加 batch 维，形状变为 (1, num_tokens)，因为模型要求输入带 batch 维

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 生成前先同步 CUDA 流，确保之前的 GPU 操作(如模型搬运)全部完成，计时更准确
    start = time.time()

    token_ids = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    # 调用不带 KV 缓存的生成函数，生成 200 个新 token
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # 生成后再次同步，确保所有 GPU 计算真正完成后才停止计时
    total_time = time.time() - start
    # 记录整个生成过程耗时，用于和 KV 缓存版本做速度对比

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    # squeeze(0) 去掉 batch 维，转成 python list 后用分词器解码回文本

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    # 统计生成速度(每秒生成的 token 数)，是与 KV 缓存版本对比性能的核心指标
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")
        # 统计 GPU 峰值显存占用，同样可用于与 KV 缓存版本对比(KV 缓存通常会增加显存占用)


if __name__ == "__main__":
    main()
