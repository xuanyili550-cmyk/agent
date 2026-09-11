# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明（新增，原文件无对应模块 docstring）：

本文件提供“带 KV 缓存（Key-Value Cache）”的自回归文本生成工具函数，
用于在推理阶段加速逐 token 解码。

背景 —— 为什么需要 KV 缓存：
在不使用缓存的朴素实现中，每生成一个新 token，都要把“到目前为止的完整
序列”重新喂给模型做一次前向传播；也就是说，对于已经计算过的历史 token，
它们的 Key/Value 向量会被一遍又一遍地重复计算，带来 O(n^2) 级别的浪费
（n 为已生成序列长度）。

KV 缓存的思路：
- 在自注意力（self-attention）中，每一层只需要用“当前新 token 的 Query”
  去和“所有历史 token 的 Key/Value”做注意力计算。
- 历史 token 的 Key/Value 一旦算出来，就不会再变化（模型权重固定、且是
  因果注意力，未来 token 不会影响过去 token 的表示）。
- 因此可以把每一层算出来的 Key/Value 张量“缓存”下来（对应 KVCache 类，
  定义在 .utils 模块中），下一步解码时只需要：
    1) 把新 token 输入模型，只计算这一个新 token 的 Query/Key/Value；
    2) 把新算出的 Key/Value 与缓存中的历史 Key/Value 在“序列长度”维度
       上拼接（见 gpt2.py 中 MultiHeadAttention.forward 里的
       torch.cat([cache[0], keys], dim=2) 等操作）；
    3) 用新 token 的 Query 与拼接后的完整 Key/Value 做注意力，得到输出；
    4) 把拼接后的 Key/Value 重新写回缓存，供下一步使用。

这样一来，每一步解码只需对“新增的 1 个（或若干个）token”做一次前向
计算，历史部分的 K/V 不再重复计算，把整体解码复杂度从 O(n^2) 降到
近似 O(n)。

张量形状变化（增量解码的核心直觉，以单层为例）：
- 首次“灌入”（prime）阶段：输入 idx 形状为 (batch, prompt_len)，
  该层缓存从 None 被填充为 keys/values 形状 (batch, num_heads, prompt_len, head_dim)。
- 之后每一步：输入只有新 token，形状为 (batch, 1)；
  该层新算出的 keys/values 形状是 (batch, num_heads, 1, head_dim)；
  与缓存拼接后变为 (batch, num_heads, prev_len + 1, head_dim)，
  即“历史长度”维度（dim=2）逐步增长，而每步真正需要新计算的部分
  始终只有长度为 1 的新增 token。

本文件本身不包含缓存的底层实现（拼接、存取逻辑在 KVCache 类以及各
模型的 attention/forward 中），而是提供两个上层“生成循环”函数：
- generate_text_simple：一次性生成 max_new_tokens 个 token 并返回完整序列。
- generate_text_simple_stream：以生成器（generator）方式逐 token 产出，
  支持遇到结束符提前停止。
"""

from ..generate import trim_input_tensor  # noqa: F401
from .utils import KVCache
import torch


def generate_text_simple(model, idx, max_new_tokens, context_size=None, use_cache=True):
    """
    使用（可选）KV 缓存的贪心解码（greedy decoding）生成函数。

    作用：
        在给定的起始 token 序列 idx 之后，连续生成 max_new_tokens 个新
        token（每步都取 logits 中概率最大的 token，即 argmax，不做采样），
        最终返回“原始输入 + 新生成部分”拼接后的完整 token 序列。

    参数：
        model (nn.Module): 语言模型，要求其 forward 支持 cache 参数
            （形如 model(idx, cache=cache) -> logits），并且有
            model.reset_kv_cache() 方法用于清空/重置内部位置计数器与缓存。
        idx (torch.LongTensor): 初始 token id 序列，形状 (batch_size, seq_len)。
        max_new_tokens (int): 要新生成的 token 数量。
        context_size (int, 可选): 模型的最大上下文长度限制；若为 None，
            则从 model.cfg["context_length"] 读取。仅在不使用缓存
            （use_cache=False）或首次灌入缓存时用于截断输入序列。
        use_cache (bool): 是否启用 KV 缓存加速解码，默认 True。

    返回：
        torch.LongTensor: 形状为 (batch_size, seq_len + max_new_tokens) 的
        完整 token 序列（原始输入 + 新生成的 token）。
    """
    model.eval()  # 切换到推理模式（关闭 dropout 等训练专用行为）
    # 确定截断用的上下文窗口长度：优先使用传入的 context_size，否则用模型配置里的默认值
    ctx_len = context_size or model.cfg["context_length"]

    with torch.no_grad():  # 推理阶段不需要梯度，节省显存和计算
        if use_cache:
            # ---------- 启用 KV 缓存的分支 ----------
            # 为模型的每一层创建一个空的 KV 缓存槽位（初始值为 None）
            cache = KVCache(n_layers=model.cfg["n_layers"])
            # 重置模型内部的位置计数器 current_pos（以及可能的历史状态），
            # 确保这是一次全新的、从位置 0 开始的解码
            model.reset_kv_cache()
            # “灌入”（prime）阶段：把（可能被截断到 ctx_len 长度的）完整初始
            # 序列一次性喂给模型，这一步会为每一层计算出 prompt 部分所有
            # token 的 Key/Value 并写入 cache，形状变为
            # (batch, num_heads, prompt_len, head_dim)；
            # 返回的 logits 形状为 (batch, prompt_len, vocab_size)
            logits = model(idx[:, -ctx_len:], cache=cache)

            for _ in range(max_new_tokens):
                # 只取最后一个位置（最新 token）的 logits 做贪心选择，
                # 得到形状 (batch, 1) 的下一个 token id
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # 把新 token 拼接到已生成序列末尾，序列长度 +1
                idx = torch.cat([idx, next_idx], dim=1)
                # 增量解码关键点：这里只把“新生成的这 1 个 token”
                # （形状 (batch, 1)）喂给模型，而不是整条历史序列；
                # 模型内部会用这 1 个新 token 计算出的 Key/Value 与
                # cache 中已有的历史 Key/Value 拼接（dim=2 上长度 +1），
                # 从而避免了对历史 token 重复做 Key/Value 投影计算
                logits = model(next_idx, cache=cache)
        else:
            # ---------- 不使用缓存的朴素分支（对照组） ----------
            for _ in range(max_new_tokens):
                # 每一步都把“最近 ctx_len 个 token”的完整子序列重新喂给模型，
                # 相当于每步都要重新计算历史 token 的 Key/Value，
                # 计算量随生成长度增长而显著增加（用于和上面的缓存分支做对比）
                logits = model(idx[:, -ctx_len:], cache=None)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def generate_text_simple_stream(model, token_ids, max_new_tokens, eos_token_id=None, context_size=None):
    """
    使用 KV 缓存的流式（streaming）贪心解码生成器。

    作用：
        与 generate_text_simple 类似，同样采用贪心解码 + KV 缓存来加速，
        区别在于本函数是一个生成器（generator）：每生成一个新 token 就
        立即通过 yield 产出，调用方可以边生成边处理/边打印，而不必等待
        全部 max_new_tokens 生成完毕；同时支持在遇到结束符（eos_token_id）
        时提前终止生成。

    参数：
        model (nn.Module): 语言模型，要求 forward 支持 cache 参数，并提供
            model.reset_kv_cache() 方法。
        token_ids (torch.LongTensor): 初始 token id 序列（prompt），
            形状 (batch_size, seq_len)。
        max_new_tokens (int): 最多生成的新 token 数量上限。
        eos_token_id (int, 可选): 结束符 token id；若某一步生成的所有
            batch 样本的下一个 token 都等于该值，则提前停止生成。
        context_size (int, 可选): 保留参数，当前函数体内未使用（未截断
            输入长度），仅出现在函数签名中，供调用方按接口习惯传入。

    产出（yield）：
        torch.LongTensor: 每次 yield 一个形状为 (batch_size, 1) 的新
        token id 张量，代表当前这一步贪心解码选出的下一个 token。
    """
    model.eval()  # 切换到推理模式

    with torch.no_grad():  # 生成过程不需要梯度
        # 为每一层创建空的 KV 缓存槽位
        cache = KVCache(n_layers=model.cfg["n_layers"])
        # 重置模型内部位置计数器，保证从位置 0 开始的干净状态
        model.reset_kv_cache()

        # Prime the cache with the initial context
        # 灌入阶段：把完整的初始 prompt（token_ids，形状 (batch, seq_len)）
        # 一次性前向传播，为每一层计算并缓存 prompt 部分所有位置的
        # Key/Value（形状变为 (batch, num_heads, seq_len, head_dim)）；
        # 得到的 logits 形状为 (batch, seq_len, vocab_size)
        logits = model(token_ids, cache=cache)

        for _ in range(max_new_tokens):
            # 取最后一个时间步的 logits 做贪心选择，得到形状 (batch, 1) 的下一个 token
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                # 所有样本都生成了结束符，提前终止生成循环
                break

            yield next_token  # 流式产出当前新生成的 token，调用方可立即消费

            # 把新 token 追加到已生成序列末尾（主要用于维护完整历史记录，
            # 例如供调用方还原完整文本；模型前向本身依赖的是 cache 而非这个变量）
            token_ids = torch.cat([token_ids, next_token], dim=1)

            # Feed only the new token to the model; cache handles history
            # 增量解码关键点：只把新生成的这 1 个 token（形状 (batch, 1)）
            # 喂给模型；模型内部会将其计算出的 Key/Value 与 cache 中
            # 已缓存的历史 Key/Value 沿序列长度维度（dim=2）拼接，
            # 使得每一步的计算量只与“新增 1 个 token”相关，
            # 而不必重新计算此前所有历史 token 的 Key/Value
            logits = model(next_token, cache=cache)
