# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Additional utility and helper functions for text generation not covered
# in the main chapters

"""
本模块（generate.py）提供文本生成过程中会用到的可复用辅助函数。

背景说明：
- 在自回归（autoregressive）文本生成中，模型每次前向传播都需要把“当前已有的
  token 序列”作为输入，并预测下一个 token；随后把新预测的 token 拼接到序列
  末尾，再次输入模型，如此循环，直到生成到指定长度或遇到终止符。
- 但绝大多数 Transformer 模型都有一个固定的“上下文长度”（context length /
  context window）上限，即模型一次能够“看到”的最大 token 数量。如果输入序列
  长度加上还要生成的新 token 数量超过了这个上限，就会导致模型报错或者效果
  变差（因为超出位置编码/注意力窗口的范围）。
- 因此，在正式开始生成循环之前，通常需要对输入的 token 序列做“裁剪”
  （trim），只保留最靠后的一部分内容，为后续要生成的新 token 腾出位置。

本文件当前只包含一个工具函数：trim_input_tensor，用于实现上述“裁剪”逻辑。
"""


def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
    """裁剪输入的 token 序列，确保为后续生成的新 token 预留出足够空间。

    在自回归生成中，模型的“上下文长度”是固定的（context_len）。如果输入的
    prompt（提示词，已编码为 token id 序列）本身已经很长，加上还要生成的
    max_new_tokens 个新 token 之后可能会超过 context_len，导致越界。因此
    这里会在必要时对输入序列做“左侧截断”（只保留序列末尾的部分 token，
    丢弃前面的旧内容），只保留最近的 keep_len 个 token，从而保证
    “已有 token 数 + 待生成 token 数” 不会超过模型的上下文长度上限。

    参数：
        input_ids_tensor: 形状为 [batch_size, seq_len] 的张量（Tensor），
            表示当前已经编码好的输入 token id 序列。batch_size 通常为 1
            （单条 prompt），seq_len 是当前序列长度。
        context_len: 整数，模型支持的最大上下文长度（即模型一次能处理的
            最大 token 数量，例如 GPT 系列的 context window）。
        max_new_tokens: 整数，本次生成计划新增的 token 数量。

    返回：
        裁剪后的 input_ids_tensor，形状为 [batch_size, min(seq_len, keep_len)]。
        如果原始序列长度没有超过 keep_len，则原样返回，不做任何改动。
    """
    # 前置断言：要生成的新 token 数量必须小于模型的上下文长度，
    # 否则说明这次生成请求本身就不合理（新内容都放不下上下文窗口）。
    assert max_new_tokens < context_len

    # 计算“允许保留的最大已有 token 数”：
    # 用上下文总长度减去将要生成的新 token 数，剩下的就是留给已有输入的空间。
    # 用 max(1, ...) 兜底，保证至少保留 1 个 token（避免出现空序列）。
    keep_len = max(1, context_len - max_new_tokens)

    # If the prompt is too long, left-truncate to keep_len
    # 如果当前输入序列长度（input_ids_tensor.shape[1]，即 seq_len 维度）
    # 超过了允许保留的长度 keep_len，就只保留序列最后面的 keep_len 个 token
    # （即“左侧截断”：丢弃前面较早、较不重要的上下文，保留最近的内容）。
    # 切片 [:, -keep_len:] 表示：batch 维度全部保留，序列维度只取最后
    # keep_len 个元素，因此输出形状变为 [batch_size, keep_len]。
    if input_ids_tensor.shape[1] > keep_len:
        input_ids_tensor = input_ids_tensor[:, -keep_len:]

    # 返回裁剪（或未裁剪）后的 token 序列，供后续生成循环使用。
    return input_ids_tensor
