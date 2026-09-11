# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# ============================================================================
# 本文件是第 2/3 章已实现组件的“汇总/复用”文件，供本章 notebook 直接 import。
# 内容与前两章基本一致，这里只做简要注释；本章重点在 gpt_download.py 与
# 04_part-1/04_part-2 两个 notebook。
# 包含：数据集与 DataLoader、多头注意力、LayerNorm、GELU、前馈网络、
#       Transformer 块、完整 GPTModel，以及训练/评估/文本生成等工具函数。
# ============================================================================

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    # 用滑动窗口把长文本切成 (输入, 目标) 序列对；目标是输入右移一位（预测下一个 token）。
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 先把整段文本编码成 token id 序列。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 以 stride 为步长滑动取长度 max_length 的窗口；target 相对 input 整体右移 1。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 下一个 token 作为标签
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)  # 样本数

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]  # 返回一对 (输入, 目标)


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 便捷工厂：构建 GPT-2 分词器 + 数据集 + DataLoader。
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")  # 使用与 GPT-2 一致的 BPE 分词

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


class MultiHeadAttention(nn.Module):
    # 因果多头自注意力：把 d_out 均分给 num_heads 个头并行计算，再拼回。
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 每个头维度必须整除

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim  # 单头维度

        # Q/K/V 三个线性投影；qkv_bias 控制是否带偏置（加载官方 GPT-2 权重时需为 True）。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs  # 多头拼接后的输出投影
        self.dropout = nn.Dropout(dropout)
        # 上三角因果掩码（对角线以上为 1），确保每个位置只能看到自身及左侧 token。
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 批大小、序列长度、输入维度

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        # 把最后一维拆成 (num_heads, head_dim)，实现“多头”。
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 把 head 维提到前面，便于对每个头做独立的注意力矩阵运算。
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head  # QK^T 得到注意力打分

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 按当前实际序列长度截取掩码

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # 未来位置置 -inf，softmax 后权重≈0

        # 缩放（除以 sqrt(head_dim)）后 softmax 得到注意力权重，再做 dropout。
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)  # 加权求和得到上下文向量并换回维度顺序

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # 拼接所有头
        context_vec = self.out_proj(context_vec)  # optional projection  # 输出线性投影

        return context_vec


class LayerNorm(nn.Module):
    # 层归一化：对每个 token 的特征维做归一化，再用可学习的 scale/shift 缩放平移。
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))   # 对应官方权重里的 g (gamma)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 对应官方权重里的 b (beta)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                    # 沿特征维求均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)      # 有偏方差（与 GPT-2 一致）
        norm_x = (x - mean) / torch.sqrt(var + self.eps)       # 标准化
        return self.scale * norm_x + self.shift                # 仿射变换


class GELU(nn.Module):
    # GELU 激活函数的 tanh 近似实现（GPT-2 使用的正是该近似）。
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    # 位置前馈网络：先升维到 4*emb_dim，经 GELU 非线性，再降回 emb_dim。
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 对应官方 mlp/c_fc（升维）
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 对应官方 mlp/c_proj（降维）
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    # 单个 Transformer 块：Pre-LN 结构 + 两处残差连接（注意力子层、前馈子层）。
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
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 注意力前的归一化（对应 ln_1）
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 前馈前的归一化（对应 ln_2）
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        # 注意力子层：先归一化 -> 注意力 -> dropout -> 残差相加。
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back  # 残差连接，缓解深层梯度消失

        # Shortcut connection for feed forward block
        # 前馈子层：同样是 归一化 -> 前馈 -> dropout -> 残差相加。
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    # 完整 GPT 模型：token 嵌入 + 位置嵌入 -> N 个 Transformer 块 -> 末层归一化 -> 输出头。
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])       # token 嵌入（对应 wte）
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])   # 位置嵌入（对应 wpe）
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 堆叠 n_layers 个 Transformer 块。
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最终层归一化（对应全局 g/b）
        # 输出头：把隐藏向量映射回词表 logits；bias=False，且其权重与 tok_emb 共享（weight tying）。
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)                                        # 查 token 向量
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 按位置索引查位置向量
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]  # 两者相加得到输入表示
        x = self.drop_emb(x)
        x = self.trf_blocks(x)   # 经过所有 Transformer 块
        x = self.final_norm(x)
        logits = self.out_head(x)  # 每个位置输出词表上的未归一化分数
        return logits


def calc_loss_batch(input_batch, target_batch, model, device):
    # 计算单个批次的交叉熵损失（把 (batch, seq, vocab) 展平后与目标对齐）。
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    # 对 DataLoader 中若干批次求平均损失（可用 num_batches 限制评估批数以省时）。
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")  # 空 loader 返回 nan
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))  # 防止越界
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches  # 返回平均损失


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 在训练/验证集上各评估 eval_iter 个批次；用 eval() + no_grad 关闭 dropout 与梯度。
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 评估后切回训练模式
    return train_loss, val_loss


def text_to_token_ids(text, tokenizer):
    # 文本 -> token id 张量，并在最前面加一个 batch 维度。
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    # token id 张量 -> 文本；先去掉 batch 维再解码。
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context):
    # 训练过程中的采样打印：用当前模型从 start_context 续写一小段并打印。
    model.eval()
    context_size = model.pos_emb.weight.shape[0]  # 位置嵌入行数即模型支持的最大上下文长度
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format  # 换行替空格，单行紧凑打印
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    # 绘制训练/验证损失曲线，并附加“已见 token 数”的第二 x 轴。
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis  # x 轴只显示整数 epoch

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis  # 共享 y 轴的第二 x 轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks  # 透明曲线仅用于对齐刻度
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 最简自回归贪心生成：每步取概率最大的 token 追加到序列末尾，循环 max_new_tokens 次。
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 只保留最后 context_size 个 token，避免超出模型支持的最大上下文。
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():  # 推理阶段无需梯度
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        # 生成只关心序列最后一个位置的预测分布。
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)  # 贪心取最大概率 token

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)  # 追加到序列，进入下一步

    return idx
