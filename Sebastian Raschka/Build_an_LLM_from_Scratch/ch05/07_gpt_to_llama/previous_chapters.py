# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

"""
模块级中文说明：

本文件汇总了前面章节（第 2~4 章）中已经讲过的、在“GPT 转 Llama”这一章
（ch05/07_gpt_to_llama）里会被复用的工具函数。

内容包括：
1. 文本 <-> token id 张量 的相互转换函数（text_to_token_ids / token_ids_to_text）；
2. 自回归文本生成函数 generate，支持温度采样（temperature scaling）、
   top-k 采样以及遇到结束符（eos_id）提前停止生成。

该文件可以作为独立脚本运行，也可以被其他脚本 import 后复用其中的函数。
"""

import torch


#####################################
# Chapter 5
#####################################
def text_to_token_ids(text, tokenizer):
    """
    将原始文本编码为模型输入所需的 token id 张量。

    参数:
        text (str): 待编码的原始文本字符串。
        tokenizer: 分词器对象（例如 tiktoken 的编码器），需实现 encode 方法。

    返回:
        torch.Tensor: 形状为 (1, seq_len) 的二维张量，
            其中 1 是 batch 维度（batch_size=1），seq_len 是编码后的 token 数量。
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    # unsqueeze(0)：在第 0 维插入一个 batch 维度，
    # 张量形状从 (seq_len,) 变为 (1, seq_len)，以匹配模型期望的 (batch_size, seq_len) 输入格式
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    将模型输出/生成的 token id 张量解码为可读文本。

    参数:
        token_ids (torch.Tensor): 形状为 (1, seq_len) 的 token id 张量（batch_size=1）。
        tokenizer: 分词器对象，需实现 decode 方法。

    返回:
        str: 解码得到的文本字符串。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    # squeeze(0)：去掉 batch 维度，张量形状从 (1, seq_len) 变为 (seq_len,)
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    以自回归方式，基于给定的上下文逐个 token 生成新文本（适用于 GPT / Llama 等仅解码器模型）。

    参数:
        model: 语言模型（如 GPTModel 或 Llama 模型实例），前向传播接受
            形状为 (batch_size, num_tokens) 的 token id，输出形状为
            (batch_size, num_tokens, vocab_size) 的 logits。
        idx (torch.Tensor): 初始上下文 token id，形状为 (batch_size, num_tokens)。
        max_new_tokens (int): 最多生成的新 token 数量。
        context_size (int): 模型支持的最大上下文长度（即位置编码 / 因果注意力窗口大小），
            用于裁剪输入以避免超出模型支持的序列长度。
        temperature (float): 温度采样系数。大于 0 时对 logits 做温度缩放并做多项式采样，
            等于 0（默认）时退化为贪心解码（取概率最大的 token）。
        top_k (int, optional): 若指定，则只保留 logits 最大的 top_k 个候选 token 参与采样，
            其余全部置为 -inf，从而屏蔽低概率 token。
        eos_id (int, optional): 结束符 token id。若采样/贪心得到的下一个 token 等于该 id，
            则提前终止生成循环。

    返回:
        torch.Tensor: 形状为 (batch_size, num_tokens + 新生成的token数) 的 token id 张量，
            即原始上下文与新生成内容拼接后的完整序列。
    """

    # For-loop is the same as before: Get logits, and only focus on last time step
    # 逐步生成循环：每一步只关注最后一个时间步的 logits（即“下一个 token”的预测分布）
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        # 裁剪输入序列，只保留最近 context_size 个 token，
        # 防止序列长度超过模型支持的最大上下文窗口，形状仍为 (batch_size, <=context_size)
        with torch.no_grad():
            logits = model(idx_cond)
            # 前向传播（不计算梯度，节省显存/加速推理）
            # 输出 logits 形状：(batch_size, num_tokens, vocab_size)
        logits = logits[:, -1, :]
        # 只取最后一个时间步的 logits（下一个 token 的预测分布），
        # 形状从 (batch_size, num_tokens, vocab_size) 变为 (batch_size, vocab_size)

        # New: Filter logits with top_k sampling
        # 新增：使用 top_k 采样过滤 logits
        if top_k is not None:
            # Keep only top_k values
            # 只保留 logits 中数值最大的 top_k 个
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            # min_val 是 top_k 个最大值中最小的那个，作为阈值
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
            # 将小于该阈值的 logits 全部置为 -inf，这样后续 softmax 后这些位置的概率趋近于 0

        # New: Apply temperature scaling
        # 新增：应用温度缩放
        if temperature > 0.0:
            logits = logits / temperature
            # 温度缩放：temperature 越小，分布越尖锐（更接近贪心）；越大，分布越平滑（更随机）

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            # 新增（书中未提及）：为了在 mps 设备上获得与其他设备一致的数值结果，
            # 在做 softmax 之前先减去每行（每个样本）的最大值，提升数值稳定性
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            # 应用 softmax 得到概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        # 否则（temperature=0）：与之前一样，直接取 logits 最大的词表条目索引（贪心解码）
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            # 若生成的 token 等于结束符 eos_id，则提前终止生成循环
            break

        # Same as before: append sampled index to the running sequence
        # 与之前一样：把新采样/生成的 token 拼接到已有序列末尾
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
        # 拼接后序列长度 +1，下一轮循环会基于这个更长的序列继续生成

    return idx
