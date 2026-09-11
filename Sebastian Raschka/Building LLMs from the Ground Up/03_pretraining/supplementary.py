# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# ============================================================================
# 本文件是前几章（数据处理、注意力、GPT 架构）组件的汇总，
# 供第 3 章「预训练 LLM」直接 import 使用。
# 前半部分（Dataset / MultiHeadAttention / LayerNorm / GELU / FeedForward /
# TransformerBlock / GPTModel）已在前面的架构章详解，这里只做简要注释；
# 后半部分（calc_loss_* / evaluate_model / generate_* / plot_losses 等训练相关
# 函数）是本章新增内容，会重点深入讲解。
# ============================================================================

import matplotlib.pyplot as plt        # 绘制训练/验证损失曲线
from matplotlib.ticker import MaxNLocator  # 让 x 轴只显示整数刻度（epoch 数）
import tiktoken                          # OpenAI 的 BPE 分词器（GPT-2 词表）
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader  # PyTorch 数据集/加载器基类


# GPTDatasetV1：把一整段文本切成「输入序列 / 目标序列」的滑动窗口样本。
# 核心思想：语言模型做的是「预测下一个 token」，所以目标序列就是输入序列整体右移一位。
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 先把整篇文本一次性编码成 token id 列表；allowed_special 允许出现 <|endoftext|> 特殊符
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 滑动窗口：步长 stride 决定相邻样本的重叠程度。
        #   stride < max_length -> 窗口重叠；stride == max_length -> 窗口首尾相接、互不重叠（本章训练用法）。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]        # 输入：位置 [i, i+max_length)
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标：整体右移一位 [i+1, i+max_length+1)
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)   # 样本总数（窗口个数）

    def __getitem__(self, idx):
        # 返回第 idx 个样本：(输入序列张量, 目标序列张量)，形状均为 (max_length,)
        return self.input_ids[idx], self.target_ids[idx]


# create_dataloader_v1：把文本封装成可迭代的 DataLoader，自动分批。
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")  # 固定使用 GPT-2 的 BPE 分词器（词表 50257）

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # drop_last=True：丢弃最后一个不满 batch_size 的批次，避免 batch 大小不一致；
    # shuffle：训练集打乱以提升泛化，验证集通常不打乱。
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


# MultiHeadAttention：因果（带掩码）多头自注意力。详解见架构章，这里简要标注张量形状。
# 关键点：用一个大的线性层同时算出所有头的 Q/K/V，再通过 view/transpose 拆成多个头并行计算。
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 输出维度必须能被头数整除

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        # 因果掩码：上三角（不含对角线）为 1，标记「未来位置」。注册为 buffer 随模型保存但不参与训练。
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # b=批大小, num_tokens=序列长度, d_in=输入维度

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Q·Kᵀ 得到注意力分数，形状 (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 截取到当前序列长度

        # Use the mask to fill attention scores
        # 把「未来位置」的分数填成 -inf，softmax 后其权重≈0，实现「只能看过去」的因果约束
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 除以 sqrt(head_dim) 做缩放，防止点积过大导致 softmax 梯度消失
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 训练时随机丢弃部分注意力权重

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 把多头结果拼回 (b, num_tokens, d_out)（d_out = num_heads * head_dim）
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection  # 输出线性投影，融合各头信息

        return context_vec


# LayerNorm：对每个 token 的特征向量做归一化（在最后一维 emb_dim 上），稳定训练。
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 可学习缩放 γ
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习偏移 β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                    # 沿特征维求均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)      # 有偏方差（除以 N 而非 N-1）
        norm_x = (x - mean) / torch.sqrt(var + self.eps)       # 标准化到均值0方差1
        return self.scale * norm_x + self.shift               # 再做仿射变换恢复表达能力


# GELU：GPT 使用的平滑激活函数，这里用 tanh 近似公式实现。
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU(x) ≈ 0.5·x·(1 + tanh( sqrt(2/π)·(x + 0.044715·x³) ))
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# FeedForward：Transformer 块内的逐位置前馈网络，先升维到 4×再降回，中间夹 GELU。
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 升维（扩展隐藏空间）
            GELU(),                                          # 非线性激活
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 降回原维度
        )

    def forward(self, x):
        return self.layers(x)


# TransformerBlock：一个 Transformer 层 = 多头注意力子层 + 前馈子层，
# 每个子层都用「Pre-LayerNorm + 残差连接」结构（先归一化再进子层，输出再加回输入）。
class TransformerBlock(nn.Module):
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
        # Shortcut connection for attention block
        # 注意力子层：残差 = 输入 + Dropout(Attention(LayerNorm(输入)))
        shortcut = x                      # 保存原始输入用于残差
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back  # 残差连接，缓解深层网络梯度消失

        # Shortcut connection for feed forward block
        # 前馈子层：同样的 Pre-Norm + 残差结构
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


# GPTModel：完整的 GPT 模型。词嵌入+位置嵌入 -> N 个 TransformerBlock -> 最终归一化 -> 输出头。
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])      # token 嵌入表 (vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 可学习位置嵌入 (context_length, emb_dim)
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠 n_layers 个 Transformer 块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 输出头：把每个位置的隐藏向量映射到词表 logits (emb_dim -> vocab_size)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape       # in_idx 形状 (batch, seq_len)，元素是 token id
        tok_embeds = self.tok_emb(in_idx)        # (batch, seq_len, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (seq_len, emb_dim)，按位置广播相加
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]  # 词义+位置信息融合
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)                # (batch, seq_len, vocab_size)：每个位置对下一个 token 的打分
        return logits


# ============================================================================
# 【本章重点】calc_loss_batch：计算单个 batch 的交叉熵损失
# ----------------------------------------------------------------------------
# 语言模型训练目标是「预测下一个 token」：模型对输入的每个位置都输出一个词表分布，
# 用它去预测目标序列（输入右移一位）对应位置的真实 token，越准损失越低。
# ============================================================================
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 数据搬到 GPU/CPU
    logits = model(input_batch)  # 前向：logits 形状 (batch, seq_len, vocab_size)
    # cross_entropy 只接受二维输入 (N, vocab_size) 和一维标签 (N,)，因此需要展平：
    #   logits.flatten(0, 1): 把前两维 (batch, seq_len) 合并 -> (batch*seq_len, vocab_size)
    #   target_batch.flatten(): (batch, seq_len) -> (batch*seq_len,)
    # 这样每个「位置」都被当作一个独立的分类样本，标签是该位置应预测的下一个 token id。
    # cross_entropy 内部会先做 log_softmax 再取负对数似然，并对所有位置求平均。
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


# ============================================================================
# 【本章重点】calc_loss_loader：对整个 DataLoader（或前 num_batches 个批次）求平均损失
# 用途：评估模型在训练集/验证集上的整体表现。num_batches 可限制评估的批次数以加速。
# ============================================================================
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")            # 空加载器，返回 NaN 避免除零
    elif num_batches is None:
        num_batches = len(data_loader)  # 默认遍历全部批次
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))  # 防止请求的批次数超过实际数量
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()   # .item() 取标量并脱离计算图，避免累积显存/梯度
        else:
            break
    return total_loss / num_batches     # 返回所选批次的平均损失


# ============================================================================
# 【本章重点】evaluate_model：在训练过程中周期性评估训练/验证损失
# ----------------------------------------------------------------------------
# model.eval()：切换到评估模式，关闭 Dropout、让 LayerNorm 等使用推理行为，保证评估稳定。
# torch.no_grad()：评估阶段不需要反向传播，关闭梯度追踪可省显存、加速。
# 评估完必须 model.train() 切回训练模式，否则后续训练 Dropout 不生效。
# eval_iter：只用前 eval_iter 个批次估算损失，加快评估速度。
# ============================================================================
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()                       # 进入评估模式（关闭 Dropout）
    with torch.no_grad():              # 禁用梯度计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()                      # 恢复训练模式
    return train_loss, val_loss


# ============================================================================
# 【本章重点】文本 <-> token id 的相互转换（注意 batch 维度的增删）
# 模型 forward 期望输入形状是 (batch, seq_len)，即使只推理一句话也要有 batch 维。
# ============================================================================
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})  # 文本 -> id 列表 (seq_len,)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension  # (seq_len,) -> (1, seq_len)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension  # (1, seq_len) -> (seq_len,)，去掉 batch 维
    return tokenizer.decode(flat.tolist())  # id 列表 -> 文本字符串


# ============================================================================
# 【本章重点】generate_and_print_sample：训练中生成一段样例文本，直观观察模型进步
# 每次调用都用固定 start_context 续写 50 个 token，随训练进行可看到输出越来越通顺。
# ============================================================================
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()                                        # 生成时切到评估模式
    context_size = model.pos_emb.weight.shape[0]        # 从位置嵌入表读取模型支持的最大上下文长度
    encoded = text_to_token_ids(start_context, tokenizer).to(device)  # 起始文本编码为 (1, seq_len)
    with torch.no_grad():                               # 生成不需要梯度
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size  # 贪心续写 50 个 token
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format  # 换行替空格，紧凑打印
    model.train()                                      # 恢复训练模式


# ============================================================================
# 【本章重点】plot_losses：绘制训练/验证损失曲线，并用双 x 轴同时显示 epoch 与「已见 token 数」
# 下轴 = epoch（训练轮数），上轴 = tokens seen（累计处理的 token 数），二者对同一条曲线两种解读。
# ============================================================================
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")               # 训练损失（实线）
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")  # 验证损失（点划线）
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis  # 下轴只显示整数 epoch

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis  # 与下轴共享 y 轴的第二 x 轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks  # 透明曲线，仅为对齐上轴刻度
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")  # 保存为 PDF
    plt.show()


# ============================================================================
# 【本章重点】generate_text_simple：最简单的自回归贪心解码
# 每一步：喂入当前序列 -> 取最后位置的 logits -> 选概率最高的 token -> 拼接 -> 重复。
# 这是「贪心（greedy）」策略，每步都选最可能的词，确定性输出（无采样随机性）。
# ============================================================================
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):  # 循环生成 max_new_tokens 个新 token

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 若序列超过模型上下文上限，只保留最后 context_size 个 token 作为条件
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():  # 推理无需梯度
            logits = model(idx_cond)  # (batch, cur_len, vocab_size)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        # 自回归只关心「下一个」token，因此只取最后一个位置的 logits
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)  # 转成概率分布
        # 注：贪心解码下 argmax(logits) 与 argmax(softmax) 结果相同，softmax 此处非必需，仅为清晰

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)  # 取概率最大的 token id

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)  # 把新 token 接到序列末尾

    return idx  # 返回「起始 + 新生成」的完整序列