# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块对应《从零构建大语言模型》第5章："在无标注数据上进行预训练"。

主要内容包括：
1. 采样解码函数 `generate`：在 ch04 简单贪心解码 `generate_text_simple` 的基础上，
   新增温度缩放（temperature scaling）与 top-k 采样，让生成文本更具多样性。
2. 训练循环 `train_model_simple`：标准的“前向传播 -> 计算损失 -> 反向传播 -> 参数更新”
   训练流程，并周期性地在训练集/验证集上评估损失、打印生成样例。
3. 损失计算相关函数：`calc_loss_batch`（单批次交叉熵损失）与
   `calc_loss_loader`（对整个 DataLoader 上若干批次求平均损失）。
4. 文本与 token id 互转的工具函数：`text_to_token_ids` / `token_ids_to_text`。
5. 训练曲线绘图函数 `plot_losses`。
6. 预训练权重加载相关函数：`assign`、`load_weights_into_gpt`，
   用于把 OpenAI 官方发布的 GPT-2 TensorFlow checkpoint 权重，
   映射拷贝进本仓库自己实现的 GPT 模型（对应 ch04 中定义的模型结构）。
7. GPT-2 官方权重的下载与解析：`download_and_load_gpt2`、`download_file`、
   `load_gpt2_params_from_tf_ckpt`，从 OpenAI 的 TensorFlow checkpoint 中
   把权重解析成嵌套字典结构，供 `load_weights_into_gpt` 使用。

本文件仅新增中文注释与文档字符串，不改动任何可执行代码。
"""

from .ch04 import generate_text_simple

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import requests
import torch
from tqdm import tqdm


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """带温度缩放与 top-k 采样的自回归文本生成函数。

    相比 ch04 中的 `generate_text_simple`（只用贪心解码，即每步都取概率最大的 token），
    本函数增加了两个可控生成多样性的机制：
        - top_k：只保留 logits 最大的 k 个候选 token，其余全部置为 -inf，
          防止采样到极低概率、语义不合理的 token。
        - temperature：对 logits 做除法缩放。温度越低（趋近于0）越接近贪心解码（更确定），
          温度越高，概率分布越平滑，生成结果越随机多样。

    参数:
        model: 已训练/加载权重的 GPT 模型，前向传播返回形状为
            (batch_size, seq_len, vocab_size) 的 logits。
        idx: 当前已有的 token id 序列，形状 (batch_size, num_tokens)。
        max_new_tokens: 最多生成多少个新 token。
        context_size: 模型支持的最大上下文长度，用于截断输入序列，
            避免超过位置编码支持的长度。
        temperature: 温度系数。当 <= 0 时退化为贪心解码（argmax）；
            当 > 0 时使用 softmax 采样。
        top_k: 若指定，则只在 logits 最大的 top_k 个 token 中做后续处理（采样或贪心）。
        eos_id: 结束符 token id，若采样到该 id 则提前停止生成。

    返回:
        idx: 扩展后的 token id 序列，形状 (batch_size, num_tokens + 实际新增的token数)。
    """

    # For-loop is the same as before: Get logits, and only focus on last time step
    # 逐步生成：每次循环只生成一个新 token，共循环 max_new_tokens 次（可能提前 break）
    for _ in range(max_new_tokens):
        # 只取最近 context_size 个 token 作为模型输入，避免超出模型支持的最大上下文长度
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            # 前向传播，logits 形状: (batch_size, seq_len, vocab_size)
            logits = model(idx_cond)
        # 只关心序列最后一个时间步的 logits（即“下一个 token”的预测分布）
        # 形状变为: (batch_size, vocab_size)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        # 新增：使用 top_k 过滤 logits，只保留概率最高的 k 个候选
        if top_k is not None:
            # Keep only top_k values
            # 取出每行(每个样本) logits 最大的 top_k 个值，形状: (batch_size, top_k)
            top_logits, _ = torch.topk(logits, top_k)
            # 每行 top_k 个值中的最小值，作为该行的阈值，形状: (batch_size,)
            min_val = top_logits[:, -1]
            # 所有小于阈值的 logits 都被置为 -inf，softmax 后概率趋近于 0，相当于被过滤掉
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        # 新增：应用温度缩放
        if temperature > 0.0:
            # 温度缩放：logits 除以温度值。温度<1 会拉大分布差异（更确定），
            # 温度>1 会拉平分布（更随机）
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            # 书中未提及的数值稳定性技巧：softmax 前减去每行最大值，
            # 防止指数运算溢出，同时保证在 mps 设备上结果与其他设备一致
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            # 对 logits 做 softmax，得到每个 token 的概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            # 按概率分布随机采样一个 token id（多项式采样，而非取最大概率）
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        # 否则和之前一样：直接取 logits 最大值对应的 token id（贪心解码）
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            # 若采样/预测到结束符 token，且指定了 eos_id，则提前终止生成循环
            break

        # Same as before: append sampled index to the running sequence
        # 和之前一样：把新生成的 token id 拼接到序列末尾，作为下一步生成的上下文
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """标准的 GPT 预训练循环（简化版），对无标注文本执行"下一个 token 预测"训练。

    参数:
        model: 待训练的 GPT 模型。
        train_loader / val_loader: 分别提供训练集 / 验证集批次数据的 DataLoader，
            每个批次产出 (input_batch, target_batch)，两者形状均为 (batch_size, seq_len)，
            target_batch 是 input_batch 整体右移一位后的结果。
        optimizer: 优化器（如 AdamW），用于根据梯度更新模型参数。
        device: 计算设备（"cpu" / "cuda" / "mps"）。
        num_epochs: 训练轮数。
        eval_freq: 每隔多少个训练 step 做一次评估（打印训练/验证损失）。
        eval_iter: 每次评估时，在训练集/验证集上各计算多少个批次的损失做平均。
        start_context: 每个 epoch 结束后，用于生成样例文本的起始提示词字符串。
        tokenizer: 分词器，用于将 start_context 编码为 token id，以及将生成结果解码为文本。

    返回:
        train_losses: 训练过程中记录的训练集损失列表（每次评估记录一个值）。
        val_losses: 训练过程中记录的验证集损失列表（每次评估记录一个值）。
        track_tokens_seen: 与上面两个列表一一对应，记录到该次评估为止，
            模型总共"看过"（训练用到）的 token 数量，用于横轴对比。
    """
    # Initialize lists to track losses and tokens seen
    # 初始化用于记录损失和已训练 token 数量的列表
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    # 主训练循环：按 epoch 迭代
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        # 将模型切换为训练模式（启用 dropout 等训练专属行为）

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            # 清空上一个 batch 遗留的梯度，避免梯度累积
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            # 反向传播，计算损失关于每个参数的梯度
            optimizer.step()  # Update model weights using loss gradients
            # 根据梯度和优化器规则（如 AdamW 的动量、学习率）更新模型参数
            tokens_seen += input_batch.numel()
            # 累计已训练过的 token 总数（batch_size * seq_len）
            global_step += 1

            # Optional evaluation step
            # 可选的评估步骤：每隔 eval_freq 个 step 评估一次
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        # 每个 epoch 结束后打印一段生成样例文本，直观查看模型学习进度
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """在训练集和验证集上分别评估当前模型的平均损失。

    参数:
        model: 当前正在训练的模型。
        train_loader / val_loader: 训练集 / 验证集 DataLoader。
        device: 计算设备。
        eval_iter: 各自最多评估多少个批次（用于加速评估，而非遍历整个数据集）。

    返回:
        train_loss: 训练集上 eval_iter 个批次的平均交叉熵损失（标量 float）。
        val_loss: 验证集上 eval_iter 个批次的平均交叉熵损失（标量 float）。
    """
    model.eval()
    # 切换为评估模式（关闭 dropout 等），并在 no_grad 上下文中前向传播以节省显存/加速
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    # 评估结束后恢复为训练模式，以便继续训练循环
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """给定起始提示词，用当前模型生成一段文本并打印出来，用于直观观察训练效果。

    参数:
        model: 当前模型，需具有 `pos_emb` 属性（位置编码 Embedding 层），
            用于推断模型支持的最大上下文长度。
        tokenizer: 分词器，用于编码/解码文本。
        device: 计算设备。
        start_context: 生成文本使用的起始提示词字符串。
    """
    model.eval()
    # 从位置编码 Embedding 层的权重形状中取出模型支持的最大上下文长度
    # gpt.pos_emb.weight 形状: (context_size, emb_dim)，取第0维即为 context_size
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        # 使用 ch04 中最简单的贪心解码函数生成后续 50 个 token
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
        # 将生成文本中的换行符替换为空格，让打印结果更紧凑（单行显示）
    model.train()
    # 生成完毕后恢复为训练模式，继续训练循环


def assign(left, right):
    """将 numpy 数组 `right` 的值拷贝赋给 `left`（原 PyTorch 参数张量），并做形状校验。

    主要用于加载预训练 GPT-2 权重时，把从 TensorFlow checkpoint 中读到的 numpy 权重
    赋值给本仓库 PyTorch 模型对应的参数。

    参数:
        left: 目标 PyTorch 参数（torch.nn.Parameter），仅用于读取其原始 shape 做校验。
        right: 来源权重，numpy 数组，将被转换为新的 torch.nn.Parameter。

    返回:
        新的 torch.nn.Parameter，其数值来自 right，形状与 left 相同。

    异常:
        若 left 与 right 的形状不一致，抛出 ValueError。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """将 OpenAI 官方 GPT-2 TensorFlow checkpoint 权重加载进本仓库自定义的 GPT 模型。

    `params` 是由 `load_gpt2_params_from_tf_ckpt` 解析 TF checkpoint 得到的嵌套字典，
    其键名沿用了 OpenAI 官方 GPT-2 权重命名习惯（如 "wte"、"wpe"、"c_attn"、"c_proj"、
    "ln_1"、"ln_2" 等），本函数负责把这些权重逐一映射、转置、拷贝到
    本仓库模型（ch04 中定义的 GPT 结构）对应的层与参数上。

    关键点：
        - OpenAI 官方权重中，Linear 层权重的存储方向与 PyTorch `nn.Linear.weight`
          的约定相反（TF: (in_features, out_features) vs PyTorch: (out_features, in_features)），
          因此几乎所有权重矩阵在赋值时都做了转置 `.T`。
        - GPT-2 的自注意力使用一个合并的 `c_attn` 权重同时产出 Q、K、V 三部分，
          需要用 `np.split(..., 3, axis=-1)` 按最后一维切成三份分别赋给
          Q/K/V 三个线性层。
        - GPT-2 采用权重共享（weight tying）：词嵌入层 `wte` 与输出层 `out_head`
          共用同一份权重矩阵，因此这里对 `gpt.tok_emb.weight` 和 `gpt.out_head.weight`
          都赋值为 `params["wte"]`。

    参数:
        gpt: 本仓库定义的 GPT 模型实例（需具有 tok_emb、pos_emb、trf_blocks、
            final_norm、out_head 等属性，结构见 ch04.py）。
        params: 嵌套字典形式的 GPT-2 权重参数（见 `load_gpt2_params_from_tf_ckpt`）。

    返回:
        无返回值，直接原地修改 `gpt` 的各参数。
    """
    # 加载位置编码权重（wpe: word position embedding），形状: (context_length, emb_dim)
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    # 加载词嵌入权重（wte: word token embedding），形状: (vocab_size, emb_dim)
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # GPT-2 官方权重把 Q、K、V 三个投影矩阵合并存储在 c_attn 的 "w" 中，
        # 沿最后一维（axis=-1）平均切成三份，分别对应 query/key/value 权重
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        # 注意：TF 权重矩阵形状为 (in_features, out_features)，
        # 而 PyTorch nn.Linear.weight 形状为 (out_features, in_features)，因此需要转置 .T
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 同理，将合并的偏置项按最后一维切成 Q/K/V 三份偏置向量
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 多头注意力输出投影层（c_proj），同样需要转置权重矩阵
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 前馈网络（MLP/FeedForward）第一层：升维投影 c_fc（emb_dim -> 4*emb_dim）
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        # 前馈网络第二层：降维投影 c_proj（4*emb_dim -> emb_dim）
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 第一个 LayerNorm（注意力前），ln_1 的 g/b 分别对应缩放(scale)和偏移(shift)参数
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        # 第二个 LayerNorm（前馈网络前）
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 最终输出前的 LayerNorm（Transformer 所有 block 之后的最后一层归一化）
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 权重共享（weight tying）：输出投影层直接复用词嵌入矩阵 wte，
    # 而不是训练一份独立的输出权重
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    """将字符串文本编码为模型输入所需的 token id 张量（附带 batch 维度）。

    参数:
        text: 原始输入文本字符串。
        tokenizer: 分词器（如 tiktoken 的 GPT-2 编码器）。

    返回:
        encoded_tensor: 形状 (1, num_tokens) 的 LongTensor，第0维为 batch 维度（大小为1）。
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    # unsqueeze(0) 在最前面插入一个 batch 维度，形状从 (num_tokens,) 变为 (1, num_tokens)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将模型输出/生成的 token id 张量解码回可读文本字符串。

    参数:
        token_ids: 形状 (1, num_tokens) 的张量（batch 维度大小须为1）。
        tokenizer: 分词器。

    返回:
        解码后的文本字符串。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    # squeeze(0) 去掉 batch 维度，形状从 (1, num_tokens) 变为 (num_tokens,)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个批次数据上的交叉熵损失（下一个 token 预测任务的标准损失）。

    参数:
        input_batch: 输入 token id，形状 (batch_size, seq_len)。
        target_batch: 目标 token id（即 input_batch 整体右移一位），
            形状同样为 (batch_size, seq_len)。
        model: GPT 模型。
        device: 计算设备。

    返回:
        loss: 标量张量，表示该批次上所有位置、所有样本的平均交叉熵损失。
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 前向传播，logits 形状: (batch_size, seq_len, vocab_size)
    logits = model(input_batch)
    # flatten(0, 1) 把 batch 维和序列维合并：
    #   logits: (batch_size, seq_len, vocab_size) -> (batch_size*seq_len, vocab_size)
    #   target_batch: (batch_size, seq_len) -> (batch_size*seq_len,)
    # 这样每个"位置"都被当作一个独立的多分类样本，计算标准的交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """在整个（或部分）DataLoader 上计算平均损失。

    参数:
        data_loader: 提供 (input_batch, target_batch) 批次的 DataLoader。
        model: GPT 模型。
        device: 计算设备。
        num_batches: 最多计算多少个批次的损失并取平均；若为 None，
            则遍历整个 data_loader。

    返回:
        所有参与计算的批次的平均损失（float）；若 data_loader 为空，返回 float("nan")。
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        # 若指定的 num_batches 超过了 data_loader 实际拥有的批次数，则取二者较小值，
        # 避免因请求过多批次而出错
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # .item() 将单元素张量转换为 Python float，避免不必要的计算图保留
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制训练/验证损失随 epoch（及已见 token 数）变化的曲线图，并保存为 PDF。

    图中使用双 x 轴：下方 x 轴为 epoch 数，上方 x 轴为已训练的 token 总数，
    便于同时从"训练轮数"和"数据吞吐量"两个角度观察损失变化趋势。

    参数:
        epochs_seen: 与 train_losses/val_losses 对应的 epoch（可为小数）序列。
        tokens_seen: 与 train_losses/val_losses 对应的累计已训练 token 数序列。
        train_losses: 训练集损失序列。
        val_losses: 验证集损失序列。

    返回:
        无返回值；会调用 plt.savefig 保存图像文件，并调用 plt.show 展示图像。
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    # 绘制训练损失与验证损失随 epoch 变化的曲线（主 x 轴：epoch）
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis
    # 强制 x 轴（epoch 轴）只显示整数刻度，避免出现如 "1.5 epoch" 这样的非整数刻度标签

    # Create a second x-axis for tokens seen
    # 创建共享同一 y 轴的第二个 x 轴，用于展示"已训练 token 数"这一维度
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    # alpha=0 表示这条曲线本身不可见，仅用于让第二个 x 轴的刻度与主曲线的数据点对齐
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # 自动调整子图布局，避免标签、图例等元素相互重叠或被裁剪
    plt.savefig("loss-plot.pdf")
    plt.show()


def download_and_load_gpt2(model_size, models_dir):
    """下载指定规模的 OpenAI 官方 GPT-2 权重文件，并解析为可用于加载的参数字典。

    参数:
        model_size: GPT-2 模型规模标识，取值须为 "124M"、"355M"、"774M"、"1558M" 之一。
        models_dir: 权重文件的本地存储根目录，实际文件会存放在
            `models_dir/model_size/` 子目录下。

    返回:
        settings: 从 hparams.json 中读取的模型超参数字典（如层数、隐藏维度等）。
        params: 由 `load_gpt2_params_from_tf_ckpt` 解析得到的嵌套权重字典，
            可直接传给 `load_weights_into_gpt` 使用。
    """
    import tensorflow as tf

    # Validate model size
    # 校验传入的模型规模是否合法
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    # 定义各类下载路径：本地保存目录、官方下载源、备用下载源
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    # 依次下载 TensorFlow checkpoint 相关的所有必需文件
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    # 加载模型超参数配置，并从 TF checkpoint 中解析出全部权重参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    """下载单个文件到指定路径，支持断点判断（若本地文件已存在且大小一致则跳过）
    以及主/备下载地址自动切换。

    参数:
        url: 主下载地址。
        destination: 本地保存的目标文件路径。
        backup_url: 备用下载地址；当主地址下载失败时会尝试该地址。

    返回:
        无显式返回值；下载失败时打印错误信息（不会抛出异常终止程序）。
    """
    def _attempt_download(download_url):
        # 内部辅助函数：尝试从指定 url 下载文件，返回是否成功（True）
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has same size
        # 若本地文件已存在且大小与远程文件一致，视为已是最新版本，跳过下载
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024  # 1 KB
        desc = os.path.basename(download_url)
        # 使用 tqdm 显示下载进度条，按 1KB 大小的块流式写入文件，避免大文件占用过多内存
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        # 先尝试主下载地址
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        # 主地址下载失败（网络异常等），若提供了备用地址则尝试备用地址
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        # 主备地址均下载失败，打印友好的错误提示信息（不抛出异常，避免中断整个流程）
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        # 捕获其他未预期的异常，打印错误信息但不中断程序
        print(f"An unexpected error occurred: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """从 OpenAI 官方 GPT-2 的 TensorFlow checkpoint 文件中解析出全部权重参数，
    并整理成嵌套字典结构，供 `load_weights_into_gpt` 使用。

    TensorFlow checkpoint 中每个变量的名称形如：
        "model/h3/attn/c_attn/w"  （表示第3个 Transformer block 的注意力层 c_attn 权重）
        "model/wte"              （词嵌入权重，不属于任何具体层，直接放在顶层）
    本函数按 "/" 分割变量名，依据名称中的层号（如 "h3"）将变量归入
    `params["blocks"][3]` 对应的子字典中，其余不含层号的变量（如 wte、wpe）
    则直接放在 `params` 顶层。

    参数:
        ckpt_path: TensorFlow checkpoint 文件路径前缀（如 "model.ckpt"）。
        settings: 模型超参数字典，需包含 "n_layer" 键，用于预先创建对应数量的
            block 子字典。

    返回:
        params: 嵌套字典，结构大致为：
            {
                "wte": np.ndarray,          # 词嵌入权重
                "wpe": np.ndarray,          # 位置编码权重
                "g": np.ndarray, "b": np.ndarray,  # 最终 LayerNorm 的 scale/shift
                "blocks": [
                    {
                        "attn": {"c_attn": {"w": ..., "b": ...}, "c_proj": {...}},
                        "mlp": {"c_fc": {...}, "c_proj": {...}},
                        "ln_1": {"g": ..., "b": ...},
                        "ln_2": {"g": ..., "b": ...},
                    },
                    ...  # 共 settings["n_layer"] 个元素
                ]
            }
    """
    import tensorflow as tf

    # Initialize parameters dictionary with empty blocks for each layer
    # 初始化参数字典，为每一层 Transformer block 预先创建一个空字典占位
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    # 遍历 checkpoint 中的每一个变量（tf.train.list_variables 返回 (name, shape) 列表）
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        # 加载该变量对应的数值，并用 np.squeeze 去除所有长度为1的维度（多余的单一维度）
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        # 处理变量名字符串，提取出除 "model/" 前缀之外的各级名称部分
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        # 确定该变量应存放到哪个子字典：若名称以 "h" 开头（如 "h3"），
        # 说明它属于某个具体的 Transformer block，需要定位到对应层号的子字典
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        # 根据名称中间的各级 key（如 "attn"、"c_attn"），逐层深入/创建嵌套字典
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        # 将变量的数值数组赋值给最内层的 key（如 "w" 或 "b"）
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
