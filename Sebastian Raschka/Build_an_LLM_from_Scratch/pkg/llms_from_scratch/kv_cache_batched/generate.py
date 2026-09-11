# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明：
本文件实现了支持“批量（batched）KV cache”的文本生成函数。

与非批量版本（单条样本一个 KV cache）不同，这里的关键区别在于：
1. `KVCache`（定义于同目录 `utils.py`）内部按 [层, 批次样本] 的二维结构存储缓存，
   即 `cache.cache[layer_idx][batch_idx]`，因此可以支持同一批次中不同样本
   拥有形状不同、增长速度不同的 K/V 缓存（例如不同长度的 prompt）。
2. `model.current_pos` 不再是一个标量，而是形状为 `(batch_size,)` 的张量，
   记录批次中“每个样本”各自已经生成/消费到的位置（position）。这样即使
   同批次内样本的有效长度不同，也能各自维护正确的旋转位置编码（RoPE）
   偏移量和因果注意力掩码（causal mask）范围。
3. 因为每个样本的位置可能不同，模型内部（见 `qwen3.py` 中 `Qwen3Model.forward`）
   会根据每个样本的 `start_pos` 动态构造出形状为
   `(batch_size, 1, num_tokens, max_len)` 的注意力掩码，以保证每个样本只能
   关注到自己已经生成的 token，不会关注到 padding 或未来的 token。

本文件只负责“贪心解码（greedy decoding，即每步取概率最大的 token）”的生成
循环控制逻辑，不涉及采样温度、top-k/top-p 等策略。

以下代码在原始英文注释基础上，新增了详细的中文注释，未改动任何可执行逻辑。
"""

from ..generate import trim_input_tensor  # noqa: F401
# 说明：这里从上一级目录（非批量版本）的 generate 模块导入 `trim_input_tensor` 函数，
# 但当前文件内部并未直接调用它；加上 `# noqa: F401` 是为了让代码检查工具（如 flake8）
# 不要因为“导入了但未使用”而报警——它是特意导出（re-export）给外部调用者使用的
# 工具函数，用于在生成前把过长的输入序列裁剪到合适长度。
from .utils import KVCache
# 从同目录的 utils.py 导入 KVCache 类：一个按 [层数, batch_size] 组织的
# 二维列表结构，用来分别保存每个 Transformer 层、每个批次样本各自的
# (key, value) 缓存张量。
import torch


def generate_text_simple(model, idx, max_new_tokens, context_size=None, use_cache=True):
    """
    使用贪心解码（greedy decoding）方式，基于批量（batched）KV cache 生成文本。

    参数：
        model (nn.Module):
            具备如下接口的语言模型：
              - `model.cfg`：配置字典，至少包含 "n_layers" 和 "context_length"；
              - `model.forward(input_ids, cache=None, start_pos=None)`：
                前向传播，返回 logits，形状为 (batch_size, seq_len, vocab_size)；
              - `model.reset_kv_cache(batch_size, device)`：初始化 / 重置
                `model.current_pos`，即每个样本当前已消费的位置计数器，
                形状为 (batch_size,)；
              - `model.current_pos`：形状为 (batch_size,) 的 long 型张量，
                记录批次内每个样本“下一个待写入 token”的位置索引。
        idx (torch.LongTensor):
            初始输入 token id 序列（即 prompt），形状为 (batch_size, prompt_len)。
            函数会在此基础上不断在序列维度（dim=1）上拼接新生成的 token。
        max_new_tokens (int):
            需要新生成的 token 数量。
        context_size (int, optional):
            模型支持的最大上下文长度；若为 None，则使用
            `model.cfg["context_length"]`。仅在首次做“全量上下文前向传播”时，
            用来把过长的 prompt 截断为最近的 `context_size` 个 token。
        use_cache (bool):
            是否启用 KV cache 加速：
              - True：只在第一次前向传播时处理整段 prompt，之后每步只输入
                新生成的单个 token，历史 key/value 从缓存中读取并追加，
                从而避免重复计算历史 token 的注意力，提升解码速度；
              - False：不使用缓存，每一步都把最近 `context_size` 个 token
                重新完整输入模型做前向传播（计算量更大，但实现最简单）。

    返回：
        torch.LongTensor：
            形状为 (batch_size, prompt_len + max_new_tokens) 的 token id 序列，
            即原始 prompt 与新生成的 `max_new_tokens` 个 token 在序列维度上
            拼接后的结果。
    """
    model.eval()
    # 确定上下文窗口长度：若未显式传入 context_size，则回退到模型配置中的
    # "context_length"（模型训练/支持的最大序列长度）。
    ctx_len = context_size or model.cfg["context_length"]
    # 批次大小 B，取自输入 idx 的第 0 维（batch 维度）。
    batch_size = idx.size(0)

    with torch.no_grad():
        # 推理阶段关闭梯度计算，节省显存并加速。
        if use_cache:
            # ------------------ 启用 KV cache 的分支 ------------------
            # initialize cache and positions
            # 初始化一个按 [n_layers, batch_size] 组织的空 KV 缓存容器：
            # 每个 (layer_idx, batch_idx) 位置初始为 None，表示该层、该样本
            # 尚未写入任何历史 key/value。
            cache = KVCache(n_layers=model.cfg["n_layers"], batch_size=batch_size)
            # 重置模型内部的位置计数器 `model.current_pos`，使其变为形状
            # (batch_size,) 的全零张量：批次内每个样本都从位置 0 开始计数。
            model.reset_kv_cache(batch_size=batch_size, device=idx.device)

            # initial full-context pass
            # 首次前向传播：把 prompt 截断为最近 ctx_len 个 token 作为输入，
            # input_ids 形状为 (batch_size, seq_len)，seq_len <= ctx_len。
            input_ids = idx[:, -ctx_len:]
            # 本次要处理的 token 数（即 prompt 的有效长度，裁剪后）。
            seq_len = input_ids.size(1)
            # 记录本次前向传播开始时，每个样本的起始位置（此时应全为 0），
            # 形状 (batch_size,)；克隆一份传给模型，避免后续原地修改
            # `model.current_pos` 影响到已经传入 forward 的这份张量。
            start_pos = model.current_pos.clone()
            # 一次性把整段 prompt 喂给模型：
            # - 模型内部会为这 seq_len 个 token 计算 Q/K/V，并把 K/V 写入 cache；
            # - 因为是首次调用，start_pos 全为 0，注意力掩码退化为标准的
            #   下三角因果掩码（causal mask），保证每个 token 只能看到它之前的 token。
            # logits 形状：(batch_size, seq_len, vocab_size)。
            logits = model(
                input_ids,
                cache=cache,
                start_pos=start_pos
            )
            # 前向传播完成后，把每个样本的位置计数器向前推进 seq_len，
            # 即 model.current_pos: (batch_size,) 中每个元素都加上 seq_len，
            # 表示“到目前为止已经消费/生成了这么多个 token”。
            model.current_pos += seq_len

            # iterative generation
            # 迭代生成剩余的 max_new_tokens 个 token，每一步只处理 1 个新 token。
            for _ in range(max_new_tokens):
                # 取最后一个时间步的 logits，对词表维度（最后一维）做 argmax，
                # 得到贪心解码的下一个 token id。
                # logits[:, -1] 形状 (batch_size, vocab_size)；
                # next_token 形状 (batch_size, 1)（keepdim=True 保留序列维）。
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1)
                # 只把新生成的这一个 token 喂给模型（而不是整段历史序列）：
                # - 模型会用 cache 中已保存的历史 K/V，与本步新计算出的
                #   K/V 拼接（在 utils/qwen3.py 的注意力实现里沿 seq 维度 cat），
                #   从而得到“完整历史 + 当前 token”的注意力上下文；
                # - start_pos=model.current_pos.clone() 告诉模型这个新 token
                #   在各自样本序列中的绝对位置，用于计算 RoPE 位置编码偏移
                #   以及构造正确范围的因果掩码。
                # 本次 logits 形状：(batch_size, 1, vocab_size)。
                logits = model(
                    next_token,
                    cache=cache,
                    start_pos=model.current_pos.clone()
                )
                # 每处理完一个新 token，批次内每个样本的位置计数器各自 +1。
                model.current_pos += 1
                # 将新生成的 token 拼接到输出序列末尾（沿序列维度 dim=1），
                # idx 形状从 (batch_size, L) 增长为 (batch_size, L+1)。
                idx = torch.cat([idx, next_token], dim=1)
        else:
            # ------------------ 不使用 KV cache 的分支 ------------------
            # 每一步都不依赖缓存，而是把“最近 ctx_len 个 token”整段重新
            # 输入模型，重新计算一遍完整的注意力，因此计算开销更大，
            # 但实现逻辑最简单，也不涉及跨步的位置/缓存管理问题。
            for _ in range(max_new_tokens):
                # 截取当前已生成序列的最后 ctx_len 个 token 作为本步输入，
                # 形状 (batch_size, min(当前长度, ctx_len))。
                input_ids = idx[:, -ctx_len:]
                # 不传入 cache 和 start_pos（均为 None），模型内部会按
                # 标准的从 0 开始的下三角因果掩码进行前向传播。
                logits = model(input_ids, cache=None, start_pos=None)
                # 同样取最后一个时间步 logits 做贪心 argmax，得到下一个 token。
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                # 拼接新 token 到序列末尾。
                idx = torch.cat([idx, next_token], dim=1)

    return idx
