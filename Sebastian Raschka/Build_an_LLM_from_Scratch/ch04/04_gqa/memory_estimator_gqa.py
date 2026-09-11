# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# KV-cache memory estimator for MHA vs GQA

"""
【中文模块说明】
本文件是一个命令行小工具，用于**估算并对比** Transformer 推理时
「多头注意力 (MHA, Multi-Head Attention)」与「分组查询注意力
(GQA, Grouped-Query Attention)」两种方案所需的 **KV 缓存 (KV cache)**
显存占用量。

背景知识（面向正在学习《从零构建大语言模型》的读者）：
- 在自回归推理（逐 token 生成）时，为了避免每生成一个新 token 就重新计算
  之前所有 token 的 Key/Value，模型会把每一层、每个注意力头算出的
  K（Key）和 V（Value）张量缓存下来，这就是 "KV cache"。
- 标准 MHA 中，Query、Key、Value 都有各自独立的、数量为 n_heads 的头，
  因此 KV cache 的大小与 n_heads 成正比，序列越长、层数越多、头数越多，
  显存开销越大，这是长上下文推理的主要显存瓶颈之一。
- GQA 的核心思想：让多个 Query 头共享同一组 Key/Value 头（把 n_heads
  个头分成 n_kv_groups 组，每组内的 Query 头共用一份 K/V），从而把
  需要缓存的 K/V 头数从 n_heads 降到 n_heads / n_kv_groups，
  显著降低 KV cache 的显存占用，同时通常只带来很小的模型效果损失。
  这是 Llama 2/3、Mistral 等现代大模型广泛采用的技术。

本脚本不涉及实际的模型前向计算，只是根据配置（层数、头数、嵌入维度、
上下文长度、数据类型等）用公式**估算**两种方案的 KV cache 总字节数，
并打印出对比结果（倍数、节省比例），帮助读者直观理解 GQA 节省显存的
数量级效果。
"""


import argparse
import math

# 不同数值精度（dtype）下，每个元素占用的字节数。
# 推理时常用 fp16/bf16（各占 2 字节）以节省显存，量化后甚至可用 int8/fp8（1 字节）。
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def convert_bytes(n):
    """
    将字节数转换为以 GB 为单位的可读字符串。

    参数:
        n (int/float): 字节数（Bytes）。

    返回:
        str: 形如 "1,234.56 GB" 的格式化字符串。

    说明:
        这里使用 1000**3（十进制 GB，即“千兆字节”），而不是 1024**3（二进制 GiB），
        属于工程上常见的近似换算，用于给出直观数量级参考，不追求精确对齐操作系统显示单位。
    """
    gb = n / (1000 ** 3)  # 字节 -> GB（十进制换算）
    return f"{gb:,.2f} GB"


def calc_kv_bytes_total(batch, context_length, emb_dim, n_heads,
                             n_kv_heads, n_layers, bytes_per_elem):
    """
    计算整个模型（所有层）KV cache 所需的总字节数。

    参数:
        batch (int): 批大小（batch size），即同时推理的序列条数。
        context_length (int): 上下文长度（序列长度），即需要缓存的 token 数量。
        emb_dim (int): 模型的嵌入维度（embedding dimension），即 d_model。
        n_heads (int): 注意力头总数（用于按 emb_dim / n_heads 推算每个头的维度 head_dim）。
        n_kv_heads (int): 实际需要缓存 Key/Value 的“头”的数量。
            - 对 MHA：n_kv_heads = n_heads（每个 Query 头都有自己独立的 K/V 头）。
            - 对 GQA：n_kv_heads = n_heads / n_kv_groups（多个 Query 头共享一组 K/V 头）。
        n_layers (int): Transformer 的层数（每一层都要单独维护一份 KV cache）。
        bytes_per_elem (int): 每个数值元素占用的字节数（由 dtype 决定，如 fp16=2 字节）。

    返回:
        int/float: 整个模型、所有层的 KV cache 总字节数。

    形状与推导说明:
        - 每个 token、每个 KV 头，需要缓存一个维度为 head_dim 的 Key 向量和一个
          维度为 head_dim 的 Value 向量，即两份 (head_dim,) 大小的数据（K 和 V），
          因此公式中乘以 2（代表 K 和 V 各一份）。
        - 单层 KV cache 的逻辑形状可以理解为：
              (batch, n_kv_heads, context_length, head_dim) 分别对应 K 和 V，
          所以单层总元素数 = batch * context_length * head_dim * n_kv_heads * 2。
        - 再乘以每个元素的字节数 bytes_per_elem，得到单层所需的字节数 per_layer。
        - 最后乘以层数 n_layers，得到整个模型的 KV cache 总字节数。
        - GQA 与 MHA 使用同一套公式，唯一区别在于传入的 n_kv_heads 不同：
          MHA 传入 n_heads，GQA 传入 n_heads // n_kv_groups，
          这正是 GQA 能省显存的数学根源——线性地缩小了 n_kv_heads。
    """
    head_dim = math.ceil(emb_dim / n_heads)  # 每个注意力头的维度：emb_dim 均分给 n_heads 个头（向上取整以防不能整除）
    # 单层 KV cache 字节数 = batch * 序列长度 * 每头维度 * KV头数 * 2（K和V各一份） * 每元素字节数
    per_layer = batch * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem
    return per_layer * n_layers  # 乘以层数，得到全模型的 KV cache 总字节数


def main():
    """
    命令行入口函数：解析参数、分别计算 MHA 与 GQA 的 KV cache 显存占用，
    并打印配置信息与对比结果（倍数、节省百分比）。

    命令行参数说明:
        --context_length: 上下文长度（默认 1024）。
        --emb_dim: 模型嵌入维度（必填）。
        --n_heads: 注意力头总数（必填）。
        --n_layers: Transformer 层数（必填）。
        --n_kv_groups: GQA 分组数，即每个 KV 头对应多少个 Query 头共享（必填）。
        --batch_size: 批大小（默认 1）。
        --dtype: 数值精度类型，决定每个元素占用字节数（默认 fp16）。

    无返回值，直接打印结果到标准输出。
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Estimate KV-cache memory for MHA vs GQA")
    p.add_argument("--context_length", default=1024, type=int)
    p.add_argument("--emb_dim", required=True, type=int)
    p.add_argument("--n_heads", required=True, type=int)
    p.add_argument("--n_layers", required=True, type=int)
    p.add_argument("--n_kv_groups", required=True, type=int)
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="fp16")
    args = p.parse_args()

    # 把与模型结构相关的配置汇总到一个字典里，方便后续统一打印
    cfg = {
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_kv_groups": args.n_kv_groups,
    }

    # GQA 要求把 n_heads 个 Query 头均匀分成 n_kv_groups 组，
    # 每组内部共享一份 K/V，因此 n_kv_groups 必须能整除 n_heads，否则分组不均匀，无法实现
    if cfg["n_heads"] % cfg["n_kv_groups"] != 0:
        raise ValueError("n_kv_groups must divide n_heads exactly.")

    bytes_per_elem = DTYPE_BYTES[args.dtype]  # 根据所选 dtype 查出每个元素的字节数
    head_dim = math.ceil(cfg["emb_dim"] / cfg["n_heads"])  # 仅用于打印展示的每头维度（与 calc_kv_bytes_total 内部计算一致）

    # MHA 下，每个 Query 头都有独立的 K/V 头，因此 KV 头数 = 总头数
    n_kv_heads_mha = cfg["n_heads"]
    # GQA 下，多个 Query 头共享一组 K/V，因此实际需要缓存的 KV 头数被压缩为 n_heads / n_kv_groups
    n_kv_heads_gqa = cfg["n_heads"] // cfg["n_kv_groups"]

    # 计算标准 MHA 方案下，全模型 KV cache 的总字节数（作为对比基准）
    total_mha = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_mha,
        cfg["n_layers"],
        bytes_per_elem,
    )

    # 计算 GQA 方案下，全模型 KV cache 的总字节数（唯一区别是传入更小的 n_kv_heads_gqa）
    total_gqa = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_gqa,
        cfg["n_layers"],
        bytes_per_elem,
    )

    ratio = total_mha / total_gqa  # MHA 显存占用是 GQA 的多少倍（直观衡量“压缩倍数”）
    savings = 1 - (total_gqa / total_mha)  # GQA 相对 MHA 节省的显存比例（0~1 之间，乘以100即百分比）

    # ---- 打印配置信息 ----
    print("==== Config ====")
    for k, v in cfg.items():
        print(f"{k:17}: {v}")
    print(f"batch_size       : {args.batch_size}")
    print(f"dtype            : {args.dtype} ({bytes_per_elem} Bytes/elem)")
    print(f"head_dim         : {head_dim}")
    print(f"GQA n_kv_heads   : {n_kv_heads_gqa}")
    print()

    # ---- 打印 KV cache 显存对比结果 ----
    print("==== KV-cache totals across all layers ====")
    print(f"MHA total KV cache  : {convert_bytes(total_mha)}")
    print(f"GQA total KV cache  : {convert_bytes(total_gqa)}")
    print(f"Ratio (MHA / GQA)   : {ratio:,.2f}x")
    print(f"Savings (GQA vs MHA): {savings*100:,.2f}%")


if __name__ == "__main__":
    main()
