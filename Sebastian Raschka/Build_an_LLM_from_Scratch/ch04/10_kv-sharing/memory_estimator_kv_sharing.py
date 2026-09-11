# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# KV-cache memory estimator for MHA, GQA, and cross-layer KV sharing.

"""
模块级中文说明
================
本文件是一个命令行小工具（脚本），用于**估算 Transformer 大语言模型在自回归推理时
KV 缓存（KV cache）所占用的显存/内存大小**，并对比以下几种注意力方案：

1. **MHA（Multi-Head Attention，多头注意力）**：每个 Query 头都有自己独立的 Key/Value 头，
   KV 头数 == 注意力头数，KV 缓存开销最大。
2. **GQA（Grouped-Query Attention，分组查询注意力）**：把多个 Query 头分成若干组，
   同一组内的 Query 头共享同一套 Key/Value 头，从而减少 KV 头数量，降低 KV 缓存开销。
3. **跨层 KV 共享（cross-layer KV sharing）**：更进一步，只有部分层（n_kv_producing_layers）
   真正计算并缓存自己的 K/V，其余层直接复用这些已经算好的 K/V，从而把 KV 缓存的层数也压缩，
   进一步节省内存（这类思想常见于 YOCO、CLA 等改进方案中）。

在《从零构建大语言模型》（Build a Large Language Model From Scratch）一书的知识体系里，
本脚本属于第 4 章「实现 GPT 模型」之后、围绕**推理效率与显存优化**的扩展内容
（对应目录 ch04/10_kv-sharing），用来帮助读者直观理解：
    - 为什么原始 MHA 在长上下文推理时 KV 缓存会爆显存；
    - GQA 通过减少 KV 头数能节省多少内存；
    - 跨层 KV 共享在 GQA 基础上还能再节省多少内存。

该脚本不涉及模型训练或前向推理的具体张量运算，而是纯粹基于公式对显存占用做**理论估算**，
方便读者在设计模型结构（emb_dim、n_heads、n_kv_groups、n_layers 等超参数）之前，
先用几行公式快速评估不同方案的显存代价。
"""

import argparse
import math

# 不同数值精度（dtype）下，每个元素占用的字节数。
# 这决定了 KV 缓存里每个数值（K 或 V 的一个标量）在显存中占多少字节：
#   - fp32（单精度浮点）：4 字节，精度最高但最占内存；
#   - bf16 / fp16（半精度浮点，含 Google 的 bfloat16 和 IEEE 的 float16）：2 字节，
#     目前大模型推理/训练中最常用的折中方案；
#   - fp8 / int8（8 位浮点 / 8 位整型量化）：1 字节，最省内存，但需要配合量化方案使用。
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def convert_bytes(n):
    """
    将字节数转换为易读的 GB（十进制 GB，1 GB = 1000^3 字节）字符串。

    参数:
        n (int/float): 字节数（bytes）。

    返回:
        str: 形如 "1.23 GB" 的格式化字符串，保留两位小数并加千分位分隔符。

    注意: 这里使用 1000^3 而不是 1024^3（1 GiB），是十进制 GB 的定义，
    和硬件厂商常用的标注方式一致；如果需要二进制 GiB，需要改成 1024 ** 3。
    """
    gb = n / (1000 ** 3)
    return f"{gb:,.2f} GB"


def calc_kv_bytes_total(batch, context_length, emb_dim, n_heads,
                        n_kv_heads, n_cached_layers, bytes_per_elem):
    """
    计算在给定配置下，KV 缓存（Key + Value 缓存）总共占用的字节数。

    这是本脚本的核心公式函数：无论是 MHA、GQA，还是加了跨层共享之后的版本，
    都是调用这同一个函数，只是传入不同的 n_kv_heads（KV 头数）和
    n_cached_layers（真正需要缓存 KV 的层数）。

    参数:
        batch (int): 批大小（batch size），即同时处理的序列条数。
        context_length (int): 上下文长度（序列长度，即 seq_len），
            决定了每个 K/V 头需要缓存多少个时间步的向量。
        emb_dim (int): 模型的嵌入维度（embedding dimension），
            即 Transformer 中每个 token 的向量总维度。
        n_heads (int): 注意力头（Query 头）的数量，用于结合 emb_dim 算出每个头的维度 head_dim。
        n_kv_heads (int): 实际参与 KV 缓存的“Key/Value 头”数量。
            - 对 MHA 而言 n_kv_heads == n_heads（每个 Query 头都有独立的 KV 头）；
            - 对 GQA 而言 n_kv_heads == n_heads // n_kv_groups（多个 Query 头共享一组 KV 头，
              所以 KV 头数比 Query 头数少）。
        n_cached_layers (int): 需要真正缓存 KV 的层数。
            - 普通模型中，每一层都有自己的 KV 缓存，此时等于 n_layers；
            - 引入跨层 KV 共享后，只有 n_kv_producing_layers 层会产生并缓存 K/V，
              其余层直接复用，因此这里传入的是产生 KV 的层数而不是总层数。
        bytes_per_elem (int): 每个数值元素占用的字节数（由 dtype 决定，见 DTYPE_BYTES）。

    返回:
        int/float: KV 缓存总字节数（所有需要缓存的层加起来的总量）。

    形状/公式说明:
        - head_dim = ceil(emb_dim / n_heads)：每个注意力头的维度，
          用 math.ceil 是为了兼容 emb_dim 不能被 n_heads 整除的情况（虽然实践中通常能整除）。
        - 单层 KV 缓存的逻辑形状可以理解为：
              K: (batch, n_kv_heads, context_length, head_dim)
              V: (batch, n_kv_heads, context_length, head_dim)
          即每个 KV 头在每个 batch 样本、每个时间步都要保存一个 head_dim 维的向量，
          K 和 V 各占一份，所以公式里要乘以 2（代表 Key 和 Value 两部分）。
        - per_layer = batch * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem
          表示单层的 K+V 缓存总字节数（把上面形状里的各维度连乘，再乘以 2 份 K/V 和每元素字节数）。
        - 最终乘以 n_cached_layers，得到所有「需要缓存 KV 的层」加起来的总字节数。
    """
    head_dim = math.ceil(emb_dim / n_heads)
    per_layer = batch * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem
    return per_layer * n_cached_layers


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Estimate KV-cache memory for MHA, GQA, and cross-layer KV sharing",
    )
    p.add_argument("--context_length", default=1024, type=int)
    p.add_argument("--emb_dim", required=True, type=int)
    p.add_argument("--n_heads", required=True, type=int)
    p.add_argument("--n_layers", required=True, type=int)
    p.add_argument("--n_kv_groups", required=True, type=int)
    p.add_argument("--n_kv_producing_layers", required=True, type=int)
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="fp16")
    args = p.parse_args()

    cfg = {
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_kv_groups": args.n_kv_groups,
        "n_kv_producing_layers": args.n_kv_producing_layers,
    }

    if cfg["n_heads"] % cfg["n_kv_groups"] != 0:
        raise ValueError("n_kv_groups must divide n_heads exactly.")
    if not 1 <= cfg["n_kv_producing_layers"] <= cfg["n_layers"]:
        raise ValueError("n_kv_producing_layers must be between 1 and n_layers.")

    bytes_per_elem = DTYPE_BYTES[args.dtype]
    head_dim = math.ceil(cfg["emb_dim"] / cfg["n_heads"])

    n_kv_heads_mha = cfg["n_heads"]
    n_kv_heads_gqa = cfg["n_heads"] // cfg["n_kv_groups"]

    total_mha = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_mha,
        cfg["n_layers"],
        bytes_per_elem,
    )
    total_gqa = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_gqa,
        cfg["n_layers"],
        bytes_per_elem,
    )
    total_mha_sharing = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_mha,
        cfg["n_kv_producing_layers"],
        bytes_per_elem,
    )
    total_gqa_sharing = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_gqa,
        cfg["n_kv_producing_layers"],
        bytes_per_elem,
    )

    print("==== Config ====")
    for k, v in cfg.items():
        print(f"{k:23}: {v}")
    print(f"batch_size             : {args.batch_size}")
    print(f"dtype                  : {args.dtype} ({bytes_per_elem} Bytes/elem)")
    print(f"head_dim               : {head_dim}")
    print(f"GQA n_kv_heads         : {n_kv_heads_gqa}")
    print()

    print("==== KV-cache totals across all layers ====")
    print(f"MHA total KV cache        : {convert_bytes(total_mha)}")
    print(f"GQA total KV cache        : {convert_bytes(total_gqa)}")
    print(f"MHA + KV sharing          : {convert_bytes(total_mha_sharing)}")
    print(f"GQA + KV sharing          : {convert_bytes(total_gqa_sharing)}")
    print(f"Ratio (MHA / GQA+sharing) : {total_mha / total_gqa_sharing:,.2f}x")
    print(f"Savings vs MHA            : {(1 - total_gqa_sharing / total_mha) * 100:,.2f}%")


if __name__ == "__main__":
    main()