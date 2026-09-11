# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本文件：多 GPU DDP（DistributedDataParallel，分布式数据并行）训练加速版本。

这是在单卡优化版本基础上，加入 PyTorch 分布式训练能力后的脚本，核心改动点（也是
DDP 多卡训练的关键概念）包括：
    1. 进程组初始化（Process Group）：每张 GPU 对应一个独立进程，多个进程通过
       `init_process_group` 建立通信组，才能在反向传播时互相同步梯度。
    2. rank / world_size：`rank` 是当前进程在通信组中的唯一编号（也用作该进程
       绑定的 GPU 编号），`world_size` 是参与训练的进程（GPU）总数。
    3. DistributedSampler：让每个进程（每张 GPU）只读取整个数据集中互不重叠的一
       个子集，避免多卡重复训练同一份数据。
    4. DDP 模型包装（`torch.nn.parallel.DistributedDataParallel`）：包装后，每次
       调用 `loss.backward()` 时，DDP 会自动在所有进程间对梯度做 all-reduce（平均），
       从而保证各进程上的模型参数始终保持一致。
    5. rank 0 专属逻辑：日志打印、样本生成、模型保存、数据下载等“只需要执行一次”
       的操作，通常只在 rank 0 进程上执行，避免多进程重复输出/写文件冲突。
    6. `torch.distributed.barrier()`：用于同步所有进程的执行进度，例如等 rank 0
       下载完数据文件后，其它进程才能继续往下读取该文件。
    7. `torch.distributed.all_reduce()`：用于跨进程聚合统计量（如本脚本中统计全
       局 tokens/sec 时，将各进程的局部 token 数量求和）。

以下代码在原英文注释基础上补充了详细中文注释，仅用于讲解，未修改任何可执行逻辑。
"""


import os
import time

import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken

# NEW imports (see Appendix A):
# 中文说明：以下是为支持多 GPU 分布式训练（DDP）新增的导入。
import platform
from torch.utils.data.distributed import DistributedSampler  # 中文：分布式采样器，保证每张 GPU 拿到不重叠的数据子集
from torch.nn.parallel import DistributedDataParallel as DDP  # 中文：DDP 模型包装器，负责梯度同步
from torch.distributed import init_process_group, destroy_process_group  # 中文：进程组的创建与销毁


# NEW: function to initialize a distributed process group (1 process / GPU)
# this allows communication among processes
# (see Appendix A):
def ddp_setup(rank, world_size):
    """
    中文说明：
        初始化分布式训练所需的“进程组”（process group）。
        分布式训练中，每张 GPU 对应一个独立的 Python 进程，这些进程之间需要
        通过底层通信后端（NCCL/Gloo）互相发送/接收数据（例如梯度同步），
        而 `init_process_group` 就是建立这种通信关系的入口。

    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    中文参数说明：
        rank: 当前进程在整个分布式训练中的唯一编号（从 0 开始），本脚本中
              同时也直接用作该进程所使用的 GPU 索引（单机多卡场景）。
        world_size: 参与本次分布式训练的进程（GPU）总数。
    """
    # Only set MASTER_ADDR and MASTER_PORT if not already defined by torchrun
    # 中文：MASTER_ADDR/MASTER_PORT 是所有进程用来“对齐集合”的地址和端口，
    # 若使用 torchrun 启动，这些环境变量通常已经被自动设置，这里做兜底。
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    # initialize process group
    # 中文：根据操作系统选择合适的分布式通信后端（backend）。
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        os.environ["USE_LIBUV"] = "0"
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        # 中文：Windows 下 NCCL 不可用，退回使用 CPU/跨平台通用的 Gloo 后端。
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        # 中文：Linux + NVIDIA GPU 场景下，NCCL 是性能最优的 GPU 间通信后端，
        # 这里正式创建进程组，之后该进程才能参与 all_reduce、barrier 等集合通信操作。
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 中文：将当前进程绑定到编号为 rank 的 GPU 上，确保每个进程只使用“自己的”那张卡。
    torch.cuda.set_device(rank)


#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    """
    中文说明：
        基础的 GPT 训练数据集类。使用滑动窗口把整段文本切分成若干个
        (输入序列, 目标序列) 样本对，目标序列相对输入序列整体右移一位，
        用于自回归语言模型的“预测下一个 token”训练目标。
        该类本身与分布式训练无关，多卡场景下会配合 DistributedSampler 使用，
        由 Sampler 负责在多个进程间切分样本索引，而不是在数据集内部处理。
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # 中文：先用分词器把整段文本一次性编码成 token id 序列
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        # 中文：用步长为 stride 的滑动窗口，从 token 序列中截取长度为 max_length 的样本
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """中文说明：返回数据集中样本（输入/目标序列对）的总数量。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """中文说明：根据索引取出对应的 (输入序列, 目标序列) 张量对。"""
        return self.input_ids[idx], self.target_ids[idx]


# NEW: Modify to set shuffle=False and use a sampler
# (See Appendix A):
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, drop_last=True, num_workers=0):
    """
    中文说明：
        构建训练/验证用的 DataLoader。
        与单卡版本相比，这里的关键改动（DDP 相关）是：
            - shuffle 固定为 False；
            - 显式传入 `sampler=DistributedSampler(dataset)`。
        原因：在分布式训练中，数据的“打乱与切分”职责被移交给了
        DistributedSampler ——它会根据当前进程的 rank 和总进程数 world_size，
        自动把数据集划分成互不重叠的若干份，每个进程只加载属于自己的那一份，
        因此 DataLoader 自身不能再做 shuffle（两者是互斥的）。
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # NEW: False because of DistributedSampler below
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        # 中文：DistributedSampler 会自动读取当前进程组的 rank/world_size，
        # 把 dataset 的索引均分给各个进程，从而实现“数据并行”中的数据切分。
        sampler=DistributedSampler(dataset)  # NEW
    )
    return dataloader


#####################################
# Chapter 3
#####################################
class PyTorchMultiHeadAttention(nn.Module):
    """
    中文说明：
        使用 PyTorch 内置的高效实现 `scaled_dot_product_attention`（可自动利用
        FlashAttention 等后端加速）构建的多头自注意力模块，与手写实现相比计算
        效率更高，但在数值/接口行为上等价。
    """
    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        """中文说明：前向传播，输入形状 (batch, num_tokens, embed_dim)，输出同形状的上下文向量。"""
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        # 中文：用一个线性层同时算出 Q、K、V 三部分，减少算子调用次数
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        # 中文：拆分出 3（qkv）、num_heads（头数）、head_dim（每个头的维度）
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        # 中文：调整维度顺序，便于下面按 qkv 拆分并进行多头并行计算
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        # 中文：调用 PyTorch 原生的缩放点积注意力实现，is_causal=True 表示使用因果掩码（只能看到当前及之前的 token）
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # 中文：把多个头的输出重新拼接回完整的 d_out 维度
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec


#####################################
# Chapter 4
#####################################


class FeedForward(nn.Module):
    """中文说明：Transformer 中的前馈网络（FFN），先升维再用 GELU 激活，最后降回原维度。"""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """中文说明：前向传播，依次经过升维线性层、GELU 激活、降维线性层。"""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    中文说明：
        单个 Transformer 块，包含“多头注意力 + 残差连接”与
        “前馈网络 + 残差连接”两个子层，每个子层前使用 LayerNorm（Pre-Norm 结构）。
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
        """中文说明：前向传播，依次经过注意力子层与前馈子层，均带残差连接。"""
        # Shortcut connection for attention block
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


class GPTModel(nn.Module):
    """
    中文说明：
        完整的 GPT 模型：词嵌入 + 位置嵌入 -> 多层 TransformerBlock -> 最终 LayerNorm
        -> 输出线性层（映射到词表大小，得到每个位置的下一个 token 的 logits）。
        该模型定义本身与是否使用 DDP 无关；是否分布式训练体现在“外部如何包装/调用”
        这个模型（见下方 main 函数中 `DDP(model, device_ids=[rank])`）。
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
        """中文说明：前向传播，输入 token id 张量 (batch, seq_len)，输出各位置在词表上的 logits。"""
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    中文说明：
        最简单的自回归文本生成函数：每一步都用当前上下文预测下一个 token，
        取概率最大的 token（贪心解码），并拼接到序列末尾，循环 max_new_tokens 次。
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

#####################################
# Chapter 5
#####################################


def text_to_token_ids(text, tokenizer):
    """中文说明：将文本字符串编码为 token id 张量，并在最前面加一个 batch 维度。"""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """中文说明：将 (1, seq_len) 形状的 token id 张量解码回文本字符串。"""
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    中文说明：
        计算单个 batch 的交叉熵损失。
        注意：当 model 是 DDP 包装后的模型时，这里的 `model(input_batch)` 调用
        实际上会触发 DDP 的前向 hook；真正的梯度同步发生在随后的 `loss.backward()`
        调用中（DDP 会自动对各进程的梯度做 all-reduce 平均）。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    中文说明：
        遍历给定的 data_loader，计算其前 num_batches 个 batch 的平均损失。
        注意：在分布式场景下，这里计算的是“当前进程本地看到的那部分数据”的损失，
        并不会自动跨进程聚合（本脚本中评估函数并未对该结果做 all_reduce）。
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
    """中文说明：切换到 eval 模式，分别在训练集/验证集上计算若干 batch 的平均损失，然后切回 train 模式。"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, device, start_context):
    """
    中文说明：
        用当前模型生成一小段示例文本并打印，便于训练过程中直观查看效果。
        DDP 相关改动：模型被 DDP 包装后，真正的模型实例被存放在 `model.module`
        属性中（DDP 本身只是一个转发调用/同步梯度的外壳），所以访问模型内部
        属性（如位置嵌入 pos_emb）时需要先判断是否是 DDP 实例，再决定是否要
        经过 `.module` 取到原始模型。
    """
    model.eval()

    # NEW: Modify for DDP
    # 中文：若 model 是 DDP 包装后的对象，需要通过 model.module 才能访问原始 GPTModel 的属性
    context_size = model.module.pos_emb.weight.shape[0] if isinstance(model, DDP) else model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tiktoken.get_encoding("gpt2")).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tiktoken.get_encoding("gpt2"))
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple_with_timing(model, train_loader, val_loader, optimizer, device,
                                   num_epochs, eval_freq, eval_iter, start_context):
    """
    中文说明：
        带耗时/吞吐量统计的训练主循环，同时也是本文件里 DDP 相关逻辑最集中的地方，
        主要包含以下几个 DDP 要点：
            1. 通过 `torch.distributed.get_rank()` 获取当前进程编号，用于控制“只在
               rank 0 打印日志/生成示例”，避免多进程重复输出刷屏。
            2. 每个 epoch 开始前调用 `train_loader.sampler.set_epoch(epoch)`：
               DistributedSampler 内部依据 epoch 值生成随机打乱的种子，
               若不设置，每个 epoch 的打乱顺序会完全一致，削弱了数据打乱的效果。
            3. 使用 `torch.distributed.all_reduce` 把各进程本地统计的 token 数量
               求和，得到“全局吞吐量”（tokens/sec），从而正确反映多卡叠加后的
               整体训练速度，而不是单卡的速度。
            4. `loss.backward()` 这一行虽然代码上和单卡版本完全一样，但由于 model
               已经是 DDP 包装的实例，这里的反向传播会自动触发梯度的 all-reduce
               （跨进程平均梯度），这是 DDP 能保持各进程模型参数一致的核心机制。
    """
    train_losses, val_losses, track_tokens = [], [], []
    total_tokens, global_step, last_tokens = 0, -1, 0

    # NEW: Determine the current rank (default to 0 if not distributed)
    # 中文：获取当前进程的 rank；若分布式环境未初始化（单卡运行），则默认 rank 为 0
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # Variables for cumulative average tokens/sec
    cumulative_tokens, cumulative_time = 0.0, 0.0

    # CUDA-specific timing setup
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
        # NEW: set epoch for DistributedSampler so each process gets a unique shuffle order
        # 中文：每个 epoch 开始前必须调用 set_epoch，DistributedSampler 会据此重新生成
        # 打乱顺序（种子与 epoch 相关），否则多个 epoch 之间的数据打乱顺序会完全相同。
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        for inp_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # Forward and backward pass
            # 中文：前向计算损失。model 是 DDP 包装后的对象，前向过程与普通模型一致。
            loss = calc_loss_batch(inp_batch, tgt_batch, model, device)
            # 中文：反向传播。关键点——DDP 会在这里自动对所有进程的梯度做 all-reduce
            # （默认取平均），使得每个进程反传结束后拿到的梯度是全局一致的，
            # 这样各进程的 optimizer.step() 才能让模型参数保持同步更新。
            loss.backward()
            optimizer.step()

            total_tokens += inp_batch.numel()

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

                # Calculate local tokens processed during this interval
                # 中文：计算“本进程”在这个统计区间内处理的 token 数量（局部值）
                local_interval = total_tokens - last_tokens
                last_tokens = total_tokens

                # Aggregate the tokens processed over all devices
                # 中文：借助 all_reduce 把所有进程（GPU）的局部 token 数量求和，
                # 得到这段时间内整个集群实际处理的 token 总数（全局值）。
                local_tensor = torch.tensor([local_interval], device=device, dtype=torch.float)
                global_tensor = local_tensor.clone()
                torch.distributed.all_reduce(global_tensor, op=torch.distributed.ReduceOp.SUM)
                global_interval = global_tensor.item()

                # Global tokens per second for this interval
                global_tps = global_interval / elapsed if elapsed > 0 else 0

                # Update cumulative tokens (local) and aggregate globally
                # 中文：同样地，把累计 token 数（本地）也做一次全局求和，用于计算全程平均吞吐量
                cumulative_tokens += local_interval
                local_cum_tensor = torch.tensor([cumulative_tokens], device=device, dtype=torch.float)
                global_cum_tensor = local_cum_tensor.clone()
                torch.distributed.all_reduce(global_cum_tensor, op=torch.distributed.ReduceOp.SUM)
                global_cumulative_tokens = global_cum_tensor.item()
                cumulative_time += elapsed
                global_avg_tps = global_cumulative_tokens / cumulative_time if cumulative_time > 0 else 0

                # Evaluate model performance (this may add overhead)
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(total_tokens)

                # NEW: Only print logs once per GPU (choosing the rank 0 GPU)
                # 中文：所有进程都会执行到这里、都算出了相同的全局指标，但只让 rank 0
                # 进程负责打印日志，避免 world_size 份重复的日志同时刷屏。
                if rank == 0:
                    print(f"Ep {epoch+1}, Step {global_step:06d}, "
                          f"Train: {train_loss:.3f}, Val: {val_loss:.3f}, "
                          f"Step tok/sec: {round(global_tps)}, Global avg tok/sec: {round(global_avg_tps)}")

        # NEW Only rank 0 prints the generated sample and memory usage stats
        # 中文：文本生成、显存统计等展示性操作同样只在 rank 0 上执行一次即可。
        if rank == 0 and epoch % 5 == 0:
            generate_and_print_sample(model, device, start_context)

            # Memory stats
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(current_device) / 1024**3  # Convert to GB
                reserved = torch.cuda.memory_reserved(current_device) / 1024**3    # Convert to GB

                print(f"\nAllocated memory: {allocated:.4f} GB")
                print(f"Reserved memory: {reserved:.4f} GB\n")

    return train_losses, val_losses, track_tokens


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """中文说明：绘制训练/验证损失随 epoch 和已见 token 数变化的曲线图（双 x 轴共用一个 y 轴）。"""
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

# NEW: Add rank and world_size
def main(gpt_config, settings, rank, world_size):
    """
    中文说明：
        单个分布式进程的完整训练流程入口。每个 GPU 对应的进程都会独立执行一次
        这个 main 函数（由 `__main__` 部分启动，rank/world_size 从环境变量中读取，
        通常配合 `torchrun --nproc_per_node=<GPU数量>` 一类命令启动多进程）。
        流程概览（含 DDP 关键步骤）：
            1. ddp_setup：初始化本进程的分布式通信组，并绑定到对应 GPU；
            2. rank 0 负责下载数据集文件，其它进程用 barrier 等待，避免重复下载/文件竞争；
            3. 构建模型后用 DDP 包装，之后每次 backward 都会自动同步梯度；
            4. 使用 DistributedSampler 驱动的 DataLoader 进行训练；
            5. 训练结束后调用 destroy_process_group 释放分布式资源。
    """

    ddp_setup(rank, world_size)  # NEW: initialize process groups
    # 中文：为当前进程创建对应的 CUDA 设备对象（rank 即该进程绑定的 GPU 索引）
    device = torch.device("cuda", rank)

    torch.manual_seed(123)

    # NEW: Print info only on 1 GPU
    # 中文：环境信息只需打印一次，因此限定只有 rank 0 进程执行打印
    if rank == 0:
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")

            capability = torch.cuda.get_device_capability()
            if capability[0] >= 7:  # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+)
                torch.set_float32_matmul_precision("high")
                print("Uses tensor cores")
            else:
                print("Tensor cores not supported on this GPU. Using default precision.")
        print()

    ##############################
    # Download data if necessary
    ##############################

    file_path = "middlemarch.txt"
    url = "https://www.gutenberg.org/cache/epub/145/pg145.txt"

    # NEW: Only download 1 time
    # 中文：只让 rank 0 进程负责下载数据文件，避免多个进程同时写同一个文件产生竞争/损坏
    if rank == 0:
        if not os.path.exists(file_path):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text_data = response.text
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
    # NEW: All processes wait until rank 0 is done, using the GPU index.
    # 中文：barrier 是一种“集合同步点”，所有进程都会在这里阻塞，直到全部进程都到达
    # 此处为止——用于保证 rank 0 下载完文件之后，其它进程才会继续往下读取该文件，
    # 避免出现其它进程读到“文件还没下载完”的情况。
    torch.distributed.barrier(device_ids=[device.index])

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model = torch.compile(model)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    # NEW: Wrap model with DDP
    # 中文：用 DDP 包装模型，这是启用多卡数据并行训练的核心一步。包装之后：
    #   - 每次前向调用行为与普通模型基本一致；
    #   - 每次 loss.backward() 时，DDP 会自动在 device_ids 对应的这些进程间对
    #     梯度执行 all-reduce（平均），保证各进程上的参数更新保持一致；
    #   - 之后若要访问模型原始属性，需要通过 `model.module` 访问（见前面
    #     generate_and_print_sample 函数中的用法）。
    model = DDP(model, device_ids=[rank])
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

    # 中文：内部使用 DistributedSampler，每个进程只会拿到训练集的一个互不重叠的子集
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        num_workers=4
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        num_workers=4
    )

    ##############################
    # Train model
    ##############################

    train_losses, val_losses, tokens_seen = train_model_simple_with_timing(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
    )

    # NEW: Clean up distributed processes
    # 中文：训练结束后销毁进程组，释放底层通信资源（如 NCCL 通信句柄），是与
    # init_process_group 配套的“收尾”操作。
    destroy_process_group()

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # NEW: Extract rank and world size from environment variables
    # 中文：这些环境变量通常由启动器（如 torchrun）在启动多进程时自动注入：
    #   WORLD_SIZE：总进程数（一般等于参与训练的 GPU 数量）；
    #   LOCAL_RANK/RANK：当前进程在本机/全局的编号。
    # 若未处于分布式环境中启动（直接用 python 运行），则回退为单进程（rank=0, world_size=1）。
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0

    GPT_CONFIG_124M = {
        "vocab_size": 50304,     # Vocabulary size
        "context_length": 1024,  # Input tokens per training example
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,  # * world_size,  # NEW: Increase learning rate to account for multiple GPUs
        "num_epochs": 50,
        "batch_size": 32,
        "weight_decay": 0.1
    }

    ###########################
    # Initiate training
    ###########################

    # 中文：把当前进程的 rank、总进程数 world_size 一起传入 main，
    # 每个 GPU 进程各自独立执行一遍 main 中的训练流程。
    train_losses, val_losses, tokens_seen, model = main(
        GPT_CONFIG_124M, OTHER_SETTINGS,
        rank, world_size  # NEW
    )

    ###########################
    # After training
    ###########################

    # NEW: Only create 1 plot
    # 中文：绘图、保存结果等收尾操作同样只需 rank 0 进程执行一次
    if rank == 0:
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
