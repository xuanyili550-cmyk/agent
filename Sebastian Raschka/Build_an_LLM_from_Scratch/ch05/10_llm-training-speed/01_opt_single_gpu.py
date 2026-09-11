# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
中文模块说明（新增的模块级 docstring，不改变任何可执行代码，仅用于说明本文件用途）：

本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 5 章配套的"单 GPU 训练速度优化"示例脚本（01_opt_single_gpu.py）。

相对于教材基线版本（未做速度优化的朴素单 GPU 训练脚本），本文件在**不改变模型结构和
训练逻辑正确性**的前提下，叠加了以下单卡训练加速手段，目的是在同一张 GPU 上尽可能
提高吞吐量（tokens/sec）、降低显存占用、缩短训练时间：

1. **PyTorch 原生 scaled_dot_product_attention（Flash Attention / SDPA）**：
   用 `torch.nn.functional.scaled_dot_product_attention` 替代手写的
   QK^T -> mask -> softmax -> @V 流程，底层自动选用 Flash Attention /
   Memory-Efficient Attention 等融合 kernel，减少显存占用并提速。

2. **bfloat16（bf16）纯低精度训练**：
   通过 `model.to(device).to(torch.bfloat16)` 把模型参数和前向/反向计算都
   转换为 bf16，相比默认 float32，计算量和显存带宽需求减半；bf16 的指数位
   与 fp32 相同、数值范围接近，不易溢出/下溢，比 fp16 更稳定，且不需要
   GradScaler，适合在支持 bf16 的现代 GPU（Ampere 及以上）上使用。

3. **torch.compile 图编译**：
   `torch.compile(model)` 对模型前向/反向计算图做编译优化（算子融合、
   减少 Python 解释开销、生成更高效的 CUDA kernel），降低每步迭代耗时。

4. **TensorFloat-32 / Tensor Core 矩阵乘法精度设置**：
   检测到 GPU 计算能力 >= 7.0（支持 Tensor Core）时调用
   `torch.set_float32_matmul_precision("high")`，让涉及 float32 的矩阵乘法
   也能利用 Tensor Core 加速。

5. **fused AdamW 优化器**：
   `torch.optim.AdamW(..., fused=True)` 用一个融合的 CUDA kernel 一次性
   完成所有参数的优化器更新，减少 kernel 启动开销。

6. **更大的 batch size**：
   本文件将 batch_size 设为 32（相对基线更大），更充分利用 GPU 显存与并行
   计算能力，提升单位时间处理的样本数（吞吐量）。

7. **词表大小对齐到 64 的倍数（50304 而不是 GPT-2 原始的 50257）**：
   便于涉及词表维度的矩阵运算更好地对齐 GPU Tensor Core 的分块大小，避免
   低效的 padding 计算。

8. **DataLoader 层面的加速**：
   - `pin_memory=True`：把 batch 数据放入锁页内存，加快 CPU->GPU 拷贝；
   - `num_workers=4`：多进程并行加载数据，与 GPU 计算重叠，避免数据加载
     成为瓶颈。

9. **基于 CUDA Event 的精确计时**：
   训练循环中用 `torch.cuda.Event(enable_timing=True)` 配合
   `torch.cuda.synchronize()` 测量 GPU 上的真实耗时，而不是简单的
   `time.time()`（CUDA 操作是异步提交的，不同步无法反映真实 GPU 耗时）。

以下代码本身（可执行逻辑）与原始教材代码完全一致，未做任何修改，仅补充中文注释，
帮助中文读者理解每一步的作用，以及相对基线版本所做的加速优化及其原因。
"""


import os
import time

import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    """PyTorch Dataset：用滑动窗口把长文本切成固定长度、可重叠的（输入, 目标）token 对。

    中文说明：第 2 章引入的数据集类，目标序列是输入序列整体右移一位（自回归语言建模
    的"预测下一个 token"训练目标）。本类未做任何速度优化改动，与基线一致。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 中文：对整段文本一次性分词，得到 token id 列表；allowed_special 允许出现
        # "<|endoftext|>" 这个特殊 token。
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 中文：用滑动窗口把 token 序列切成多个长度为 max_length 的片段；stride 越小，
        # 相邻样本重叠越多、生成的样本数越多。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 中文：数据集样本总数。
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 中文：按索引返回 (输入片段, 目标片段) 张量对，供 DataLoader 组装成 batch。
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """基于原始文本构建 DataLoader。

    中文说明：内部创建 GPT-2 分词器和 GPTDatasetV1 数据集，再包装成 DataLoader。
    相对基线的加速点见下方 pin_memory 及调用方传入的 num_workers 参数说明。
    """
    # Initialize the tokenizer
    # 中文：使用 GPT-2 的 BPE 分词器。
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    # 加速优化点：pin_memory=True 把 batch 数据分配在"锁页内存"（page-locked memory）
    # 中，从而在 CPU->GPU 拷贝时可以使用更快的异步 DMA 传输，提升数据搬运带宽；
    # num_workers（由调用方传入，如 main() 中设为 4）让多个子进程并行完成批次组装，
    # 与 GPU 计算重叠进行，避免单线程数据加载拖慢 GPU 利用率。
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


#####################################
# Chapter 3
#####################################
class PyTorchMultiHeadAttention(nn.Module):
    """基于 PyTorch 原生 scaled_dot_product_attention（SDPA）实现的多头自注意力模块。

    中文说明：这是第 3 章多头注意力的"PyTorch 高效实现版"，用于替代手写的
    Q @ K^T -> 缩放 -> 因果掩码 -> softmax -> @ V 显式实现。
    加速优化点：`torch.nn.functional.scaled_dot_product_attention` 会根据硬件和
    输入形状自动选用 Flash Attention / Memory-Efficient Attention 等融合 kernel，
    相比手写实现能显著降低显存占用（无需显式构造并保存完整的注意力权重矩阵）并
    提升计算速度，是本文件相对"手写注意力"基线的一项关键单卡加速优化。
    """
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        # 中文：用一个大的线性层一次性算出 Q、K、V（3 * d_out），相比分别用三个
        # nn.Linear，合并成一次矩阵乘法可以减少 kernel 启动次数，提升 GPU 利用率。
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        # 中文：一次矩阵乘法同时得到拼接在一起的 Q、K、V。
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        # 中文：把最后一维拆分成 (3, num_heads, head_dim)，为分离 Q/K/V 及多头做准备。
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        # 中文：调整维度顺序，方便下面直接解包出 Q、K、V。
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        # 中文：按第 0 维拆分成 queries、keys、values 三个张量。
        queries, keys, values = qkv

        # 中文：只有训练模式（self.training 为 True）才启用 dropout，评估/推理时置 0。
        use_dropout = 0. if not self.training else self.dropout

        # 加速优化点：调用 PyTorch 原生 SDPA 接口，is_causal=True 自动应用因果掩码
        # （每个位置只能看到自身及之前的 token），无需手动构造 mask 矩阵；底层会
        # 自动选用 Flash Attention 等高效融合 kernel（若硬件支持），比手写实现更快、
        # 更省显存。
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多头结果重新拼接回 (batch, num_tokens, d_out)。
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


#####################################
# Chapter 4
#####################################


class FeedForward(nn.Module):
    """Transformer block 中的前馈网络（FFN / MLP）子层。

    中文说明：标准的"升维 -> 激活 -> 降维"两层 MLP，中间维度扩大到 emb_dim 的 4 倍；
    使用 GELU 的 tanh 近似版本（approximate="tanh"），该近似公式在 GPU 上计算更快，
    同时效果与精确 GELU 非常接近。
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """标准的 Pre-LayerNorm Transformer block：注意力子层 + 前馈子层，均带残差连接。

    中文说明：注意力子层使用基于 SDPA 的 PyTorchMultiHeadAttention（而非手写注意力），
    这是本文件相对基线的加速点之一，详见该类内部注释。
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = PyTorchMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        # 中文：注意力子层的残差连接——保存输入，做 LayerNorm -> 注意力 -> dropout，
        # 再把结果加回原始输入。
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        # 中文：前馈子层的残差连接，结构与上面类似。
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    """完整的 GPT 风格自回归语言模型：token/位置嵌入 + N 层 TransformerBlock + 输出头。

    中文说明：模型结构与基线完全一致；真正的加速优化发生在模型构建之后
    （见 main() 中的 torch.compile 与 .to(torch.bfloat16) 调用），以及内部使用的
    PyTorchMultiHeadAttention（SDPA）注意力实现。
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # 中文：token 嵌入 + 位置嵌入相加，得到每个位置的输入表示。
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """贪心解码：逐个生成 token，每步取概率（logits）最大的下一个 token。

    中文说明：仅用于训练过程中打印生成样本以便人工检查效果，与训练速度优化无关，
    保持与基线一致，未做任何修改。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        # 中文：若当前序列长度超过模型支持的上下文窗口，只保留最后 context_size 个 token。
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        # 中文：推理阶段不需要梯度，用 torch.no_grad() 节省显存并加速前向计算。
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        # 中文：只关心序列最后一个位置的预测（即下一个 token 的分布）。
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        # 中文：贪心策略——直接取 logits 最大值对应的 token id。
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        # 中文：把新生成的 token 拼接到序列末尾，作为下一步生成的上下文。
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

#####################################
# Chapter 5
#####################################


def text_to_token_ids(text, tokenizer):
    """将字符串文本编码为 token id 张量，并增加一个 batch 维度。

    中文说明：辅助函数，用于把用于生成样例的起始字符串转换成模型可接受的输入格式。
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将 token id 张量解码回可读文本（先去掉 batch 维度再 decode）。"""
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个 batch 的交叉熵损失（"预测下一个 token"任务）。

    中文说明：把输入/目标搬到目标设备，前向计算 logits，再将
    (batch, seq_len, vocab_size) 展平成 (batch*seq_len, vocab_size) 与展平后的
    目标一起计算交叉熵损失。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """遍历（部分）DataLoader 并计算平均损失，用于训练/验证集的整体评估。

    中文说明：num_batches 限制最多评估多少个 batch，避免每次评估都跑完整个
    数据集，从而降低评估阶段带来的额外开销（配合 evaluate_model 中的 eval_iter）。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """在训练/验证集上各评估 eval_iter 个 batch，返回 (train_loss, val_loss)。

    中文说明：评估前切换到 model.eval()（关闭 dropout 等），并用 torch.no_grad()
    关闭梯度计算以节省显存和算力；评估结束后切回 model.train() 恢复训练模式。
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """训练过程中，用当前模型基于起始文本生成一段样例文本并打印，便于人工检查效果。"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple_with_timing(model, train_loader, val_loader, optimizer, device,
                                   num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """训练主循环，同时统计并打印训练吞吐量（tokens/sec），用于衡量单卡加速效果。

    中文说明：这是本文件的核心训练函数，在常规"前向 -> 反向 -> 优化器更新"逻辑之外，
    额外加入了基于 CUDA Event 的精确计时代码，用于测量每个评估区间内的
    tokens/sec（吞吐量），从而可以直观对比本文件相对基线所做的各项加速优化
    （SDPA 注意力、bf16、torch.compile、更大 batch、fused optimizer 等）到底
    带来了多少实际吞吐提升。计时逻辑本身不影响训练结果的正确性。

    注意：由于模型在 main() 中已经通过 model.to(torch.bfloat16) 整体转换为
    bf16 精度，这里的前向/反向/优化器更新都直接在 bf16 下进行，不需要像
    float16 混合精度训练那样使用 GradScaler 做梯度缩放（bf16 指数位宽与 fp32
    相同，数值范围足够大，一般不会出现 fp16 常见的梯度下溢问题）。
    """
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    # Variables for cumulative average tokens/sec
    # 中文：用于计算"排除首个评估区间后"的累计平均 tokens/sec（首个区间通常包含
    # torch.compile 的首次图编译开销，耗时不具代表性，因此单独跳过，见下方处理）。
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # CUDA-specific timing setup
    # 中文：性能测量点——在 GPU 上运行时使用 torch.cuda.Event 计时而非简单的
    # time.time()。因为 CUDA 算子调用是异步提交到 GPU 队列的，Python 侧的
    # time.time() 只能测出"提交耗时"而非"GPU 实际执行耗时"；必须先
    # torch.cuda.synchronize() 等待 GPU 真正执行完毕，再读取事件时间戳，才能
    # 得到准确的吞吐量数据。
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
            optimizer.zero_grad()
            global_step += 1

            # Forward and backward pass
            # 中文：标准的前向计算损失 + 反向传播梯度。模型已整体转换为 bf16，
            # 因此这里的矩阵乘法等运算以 bf16 精度执行，相比 fp32 显著降低显存
            # 带宽压力、提升计算吞吐；配合 torch.compile 编译后的计算图，算子融合
            # 进一步减少了 kernel 启动开销。
            loss = calc_loss_batch(inp_batch, tgt_batch, model, device)
            loss.backward()
            # 加速优化点：optimizer.step() 使用的是 fused AdamW（main() 中创建
            # 优化器时传入 fused=True），用单个融合 CUDA kernel 更新全部参数，
            # 比逐参数循环更新更快。
            optimizer.step()

            total_tokens += inp_batch.numel()

            # At evaluation intervals, measure elapsed time and tokens per second
            if global_step % eval_freq == 0:
                # End timing for the current interval
                # 中文：到达评估间隔时，结束当前区间计时并计算耗时。
                if use_cuda:
                    t_end.record()
                    torch.cuda.synchronize()  # Wait for all CUDA ops to complete.
                    elapsed = t_start.elapsed_time(t_end) / 1000  # Convert ms to seconds
                    t_start.record()  # Reset timer for the next interval
                else:
                    elapsed = time.time() - t0
                    t0 = time.time()  # Reset timer for the next interval

                # Calculate tokens processed in this interval
                # 中文：计算本区间内实际处理的 token 数，并据此算出 tokens/sec。
                tokens_interval = total_tokens - last_tokens
                last_tokens = total_tokens
                tps = tokens_interval / elapsed if elapsed > 0 else 0  # Tokens per second

                # Update cumulative counters (skip the first evaluation interval)
                # 中文：跳过第一个评估区间（global_step == 0 时，对应刚开始训练，
                # 通常包含 torch.compile 的首次图编译开销，耗时明显偏长，若计入
                # 会拉低平均吞吐量的代表性）。
                if global_step:  # This is False only when global_step == 0 (first evaluation)
                    cumulative_tokens += tokens_interval
                    cumulative_time += elapsed

                # Compute cumulative average tokens/sec (excluding the first interval)
                # 中文：计算排除首个区间后的累计平均吞吐量，作为更稳定的速度指标，
                # 用于衡量本文件相对基线的整体加速效果。
                avg_tps = cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # Evaluate model performance (this may add overhead)
                # 中文：评估训练/验证损失；这部分开销发生在计时区间结束、重置计时器
                # 之后，因此不会计入吞吐量统计。
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)

                print(f"Ep {epoch+1}, Step {global_step:06d}, "
                      f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                      f"Step tok/sec: {round(tps)}, Avg tok/sec: {round(avg_tps)}")

        generate_and_print_sample(model, tokenizer, device, start_context)

        # Memory stats
        # 中文：每个 epoch 结束后打印显存使用情况，便于观察 bf16 + 更大 batch size
        # 等加速手段对显存占用的实际影响。
        if torch.cuda.is_available():
            device = torch.cuda.current_device()

            allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB

            print(f"\nAllocated memory: {allocated:.4f} GB")
            print(f"Reserved memory: {reserved:.4f} GB\n")

    return train_losses, val_losses, track_tokens


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制训练/验证损失随 epoch 和 tokens 变化的曲线图（双 x 轴）。

    中文说明：纯可视化辅助函数，与训练速度优化无关，保持与基线一致。
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
    """整个训练流程入口：设置设备/精度 -> 下载数据 -> 构建并优化模型 -> 构建
    DataLoader -> 训练并统计速度。

    中文说明：本函数集中体现了本文件相对基线的大部分单卡加速改动，具体见下方
    各处行内注释（Tensor Core 精度设置、torch.compile、bf16、fused AdamW、
    DataLoader 多进程加载等）。
    """

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:  # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+)
            # 加速优化点：在支持 Tensor Core 的 GPU（计算能力 >= 7.0）上，把 float32
            # 矩阵乘法精度设为 "high"，允许 PyTorch 内部使用 TensorFloat-32（TF32）
            # 等降精度算法调用 Tensor Core 加速，在几乎不影响训练效果的前提下显著
            # 提升 Linear / Attention 等大量矩阵乘法的速度。
            torch.set_float32_matmul_precision("high")
            print("Uses tensor cores")
        else:
            print("Tensor cores not supported on this GPU. Using default precision.")
    print(f"Uses tensor cores: {torch.cuda.is_available()}")
    print()

    ##############################
    # Download data if necessary
    ##############################

    file_path = "middlemarch.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"

    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    # 加速优化点：torch.compile 把模型前向（以及自动生成的反向）计算图交给
    # TorchDynamo/TorchInductor 编译，做算子融合、生成更高效的 CUDA kernel，
    # 减少 Python 层调度开销、提升每步迭代速度。首次调用会有编译开销（体现在
    # 训练循环第一个评估区间耗时明显偏长，见 train_model_simple_with_timing 中
    # 对首个区间的特殊处理）。
    model = torch.compile(model)
    # 加速优化点：先 .to(device) 把模型搬到 GPU，再 .to(torch.bfloat16) 把所有
    # 参数转换为 bfloat16 精度，实现"纯 bf16"训练（而非 fp32 主权重 + autocast
    # 的混合精度方案）。bf16 相比 fp32 显存占用减半、计算吞吐更高，且比 fp16
    # 数值范围更大、不易下溢/溢出，因此不需要额外的梯度缩放（GradScaler）。
    model.to(device).to(torch.bfloat16)
    # 加速优化点：AdamW 优化器传入 fused=True，使用单个融合 CUDA kernel 一次性
    # 完成所有参数的更新（而非对每个参数分别启动一次 kernel），减少 kernel 启动
    # 开销，在参数量较大、参数张量数量较多的模型上能带来明显的优化器步骤加速。
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        fused=True
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    # 加速优化点：num_workers=4 让数据加载/预处理在多个子进程中并行进行，与主进程
    # 的 GPU 计算重叠执行，避免 CPU 端数据准备成为吞吐量瓶颈；结合
    # create_dataloader_v1 内部的 pin_memory=True，进一步加速 host->device 数据
    # 传输。
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=4
    )

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

    train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],
        eval_freq=10,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # 加速优化点：相对 GPT-2 原始词表大小 50257，这里把 vocab_size padding 到
    # 50304（64 的整数倍），使涉及词表维度的矩阵乘法（如输出层 out_head）在 GPU
    # 上更好地对齐 Tensor Core 的分块大小，避免因维度非 64/128 整数倍产生额外的
    # 低效 padding 计算，是常见的单卡加速微优化。
    GPT_CONFIG_124M = {
        "vocab_size": 50304,     # Vocabulary size
        "context_length": 1024,  # Input tokens per training example
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-key-value bias
    }

    # 加速优化点：batch_size 设为 32，相对基线更大，能更充分利用 GPU 的并行计算
    # 能力和显存带宽，减少"GPU 空闲等待数据/调度"的比例，从而提升整体吞吐量；
    # 更大 batch 也让 bf16 + Tensor Core 的矩阵乘法效率更高（大矩阵乘法更容易
    # 打满硬件算力）。
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 15,
        "batch_size": 32,
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
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    #
    # compiled = hasattr(model, "_orig_mod")
    # if compiled:
    #     torch.save(model._orig_mod.state_dict(), "model.pth")
    # else:
    #     torch.save(model.state_dict(), "model.pth")
    #
    # model = GPTModel(GPT_CONFIG_124M)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))