# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# KV-cache memory estimator for MHA vs GQA with SWA.

"""
【模块中文说明】
本文件是一个命令行工具，用于**估算 Transformer 大语言模型在推理时 KV 缓存（KV-cache）
所占用的显存大小**，并对比以下几种情形：
    1. 标准多头注意力（MHA, Multi-Head Attention）在“全量上下文”（每层都缓存完整
       context_length 长度的 K/V）下的显存占用；
    2. 分组查询注意力（GQA, Grouped-Query Attention）在全量上下文下的显存占用
       （GQA 通过让多个 Query 头共享同一组 K/V 头，从而显著减少 KV 缓存）；
    3. 在引入**滑动窗口注意力（SWA, Sliding Window Attention）**后，MHA / GQA
       的 KV 缓存占用——SWA 层只需缓存最近 W 个 token 的 K/V，而不是全部
       context_length 个 token，因此可以进一步压缩显存。

这属于《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书中
第 4 章（构建 GPT 模型的注意力与 Transformer Block）里，关于 GQA 与 SWA 这类
“显存/计算优化技巧”的配套辅助脚本。它不参与模型的前向传播或训练，而是帮助读者
在选择模型结构超参数（头数、KV 分组数、滑动窗口大小、SWA:Full 层比例等）时，
提前用纸面计算的方式预估推理阶段 KV 缓存会占用多少显存，从而在训练/部署前
做出更合理的架构决策。
"""

import argparse
import math

# 不同数值精度（dtype）下，每个元素占用的字节数。
# 例如 bf16/fp16 每个数占 2 字节，fp8/int8 每个数占 1 字节。
# KV 缓存的显存大小 = 元素个数 * 每个元素的字节数，因此精度越低，显存占用越小。
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def convert_bytes(n):
    """
    将字节数（int/float）转换为以 GB 为单位、便于阅读的字符串。

    参数：
        n: 字节数（例如 KV 缓存总字节数）。

    返回：
        形如 "1.23 GB" 的字符串，保留两位小数。

    说明：这里使用 1000**3（十进制 GB，而非 1024**3 的 GiB）作为换算基准，
    是工程上估算显存占用时的常见近似做法，便于和厂商标称显存容量对齐。
    """
    gb = n / (1000 ** 3)
    return f"{gb:,.2f} GB"


def calc_kv_bytes_per_layer(batch, context_length, head_dim, n_kv_heads, bytes_per_elem):
    """
    计算**单层**注意力模块中 KV 缓存所占用的字节数。

    参数：
        batch: 批大小（batch size），即同时进行推理的序列条数。
        context_length: 该层需要缓存的 token 数量。对于全量注意力层，
            这是完整的上下文长度；对于滑动窗口注意力（SWA）层，
            这里传入的是有效窗口大小 W（见 estimate_totals 中的 eff_W）。
        head_dim: 每个注意力头的维度大小（emb_dim / n_heads）。
        n_kv_heads: 该层用于存储 Key/Value 的“头”的数量。
            - 对 MHA：n_kv_heads = n_heads（每个 Query 头都有自己独立的 K/V 头）。
            - 对 GQA：n_kv_heads = n_heads / n_kv_groups（多个 Query 头共享一组 K/V）。
        bytes_per_elem: 每个数值元素占用的字节数（由 dtype 决定，见 DTYPE_BYTES）。

    返回：
        单层 KV 缓存的总字节数（int/float）。

    张量形状说明：
        KV 缓存本质上是形状约为 (batch, n_kv_heads, context_length, head_dim) 的
        两份张量（K 和 V 各一份），元素总数为：
            batch * n_kv_heads * context_length * head_dim
        再乘以 2（分别对应 K 和 V 两个张量）以及每个元素的字节数，
        就得到了下面这一行的计算公式。
    """
    # KV = batch * tokens * head_dim * n_kv_heads * 2 (K,V) * bytes
    # 中文解读：batch（批大小）* context_length（需要缓存的 token 数）
    #          * head_dim（每个头的维度）* n_kv_heads（K/V 头的数量）
    #          * 2（K 和 V 各占一份）* bytes_per_elem（每个数的字节数）
    return batch * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem


def parse_ratio(ratio_str):
    """
    解析形如 "a:b" 的字符串参数，表示在一个重复的层结构“块”中，
    每 a 个滑动窗口注意力（SWA）层搭配 b 个全量注意力（Full）层。

    例如很多现代大模型（如 Gemma 2/3、GPT-OSS 等）会按照固定的比例，
    在网络中交替排列 SWA 层和 Full 层：SWA 层负责局部信息、节省显存，
    Full 层负责全局信息、保证长距离依赖能被捕捉到。

    参数：
        ratio_str: 形如 "5:1"（5 个 SWA 层配 1 个 Full 层）
                   或 "1:0"（全部是 SWA 层）的字符串。

    返回：
        (a, b) 二元组，均为非负整数，且 a + b > 0。

    异常：
        若格式不合法（无法按 ":" 分割为两个整数，或 a、b 均为 0 等），
        抛出 ValueError 并给出提示信息。
    """
    # "--swa_ratio a:b" means a SWA layers for every b full layers within a block
    # 中文：--swa_ratio a:b 表示在一个循环块里，有 a 个 SWA 层、b 个全量注意力层
    try:
        a_str, b_str = ratio_str.split(":")
        a, b = int(a_str), int(b_str)
        # 断言 a、b 均非负，且不能同时为 0（否则块长度为 0，无法分配层数）
        assert a >= 0 and b >= 0 and (a + b) > 0
        return a, b
    except Exception:
        raise ValueError("--swa_ratio must be in the form 'a:b' with nonnegative integers and a+b>0")


def distribute_layers(n_layers, a, b):
    """
    根据 SWA:Full 的比例 (a, b)，把总层数 n_layers 分配成
    “多少层是 SWA 层、多少层是全量注意力层”。

    分配逻辑：把网络看作由若干个长度为 (a+b) 的“块”重复堆叠而成，
    每个完整的块贡献 a 个 SWA 层和 b 个 Full 层；如果 n_layers 不能
    被 (a+b) 整除，剩余的 rem 层按照“先填满 a 个 SWA 名额，再填 Full”
    的顺序分配（这与很多真实模型中 SWA 层优先排列在块前部的做法一致）。

    参数：
        n_layers: Transformer 的总层数。
        a: 每个块中 SWA 层的数量。
        b: 每个块中 Full 层的数量。

    返回：
        (swa, full) 二元组：分别是总的 SWA 层数和总的 Full 层数，
        且 swa + full == n_layers。
    """
    block = a + b
    # 完整的块能重复多少次
    blocks = n_layers // block
    # 除不尽时，剩下不足一个块的层数
    rem = n_layers % block
    # 每个完整块贡献 blocks*a 个 SWA 层；剩余的 rem 层里，最多能再分配 a 个 SWA 层
    # （若 rem < a，则剩余层全部算作 SWA；若 rem >= a，则 SWA 名额用满，剩下的是 Full）
    swa = blocks * a + min(a, rem)
    # 同理计算 Full 层数：完整块贡献 blocks*b 个，剩余层中超出 a 的部分算作 Full
    full = blocks * b + max(0, rem - a)
    return swa, full


def estimate_totals(context_length, sliding_window_size, emb_dim, n_heads, n_layers,
                    n_kv_groups, batch_size, dtype, swa_ratio):
    """
    核心估算函数：给定模型的结构超参数与运行配置，计算并汇总
    MHA / GQA 在“全部使用全量注意力”以及“按 SWA:Full 比例混合”
    两种情形下，整个模型（所有层加总）的 KV 缓存显存占用。

    参数：
        context_length: 推理时的最大上下文长度（token 数）。
        sliding_window_size: 滑动窗口注意力的窗口大小 W
            （即 SWA 层最多回看多少个历史 token）。
        emb_dim: 模型的嵌入维度（embedding dimension）。
        n_heads: 注意力头的总数。
        n_layers: Transformer 层的总数。
        n_kv_groups: GQA 中的分组数；n_heads 必须能被其整除。
            n_kv_groups=1 时，GQA 退化为与 MHA 等价的头数配置的特例
            （注意：这里的 n_kv_heads_gqa = n_heads // n_kv_groups，
            所以 n_kv_groups=1 时 n_kv_heads_gqa = n_heads，等价于 MHA）。
        batch_size: 批大小。
        dtype: 数值精度，如 "fp16"、"bf16" 等，决定每个元素的字节数。
        swa_ratio: 形如 "a:b" 的字符串，表示 SWA 层与 Full 层的比例。

    返回：
        一个字典，包含：
            - bytes_per_elem: 每个元素的字节数
            - head_dim: 每个注意力头的维度
            - n_kv_heads_gqa: GQA 下 K/V 头的数量
            - eff_W: 实际生效的滑动窗口大小（不会超过 context_length）
            - n_swa_layers / n_full_layers: 分配得到的 SWA 层数与 Full 层数
            - total_mha_allfull: MHA 全量场景下的 KV 缓存总字节数
            - total_gqa_allfull: GQA 全量场景下的 KV 缓存总字节数
            - total_mixed_mha: MHA + SWA 混合场景下的 KV 缓存总字节数
            - total_mixed_gqa: GQA + SWA 混合场景下的 KV 缓存总字节数
    """
    # GQA 要求多个 Query 头必须能被“平均”分配到每一组 K/V 头上，
    # 否则无法实现“n_kv_groups 个 Query 头共享 1 个 K/V 头”这种整除关系。
    if n_heads % n_kv_groups != 0:
        raise ValueError("n_kv_groups must divide n_heads exactly.")

    bytes_per_elem = DTYPE_BYTES[dtype]
    # 每个注意力头的维度：用 ceil 是为了在 emb_dim 不能被 n_heads 整除时也能给出一个
    # 合理（偏保守，向上取整）的头维度估计，避免除法产生小数导致后续计算异常。
    head_dim = math.ceil(emb_dim / n_heads)
    # MHA 下，K/V 头数等于 Query 头数（每个头独立拥有自己的 K/V）
    n_kv_heads_mha = n_heads
    # GQA 下，K/V 头数是 Query 头数的 1/n_kv_groups（多组 Query 头共享同一组 K/V）
    n_kv_heads_gqa = n_heads // n_kv_groups

    # 解析 SWA:Full 比例，并据此把总层数分配为 SWA 层数与 Full 层数
    a_swa, b_full = parse_ratio(swa_ratio)
    n_swa_layers, n_full_layers = distribute_layers(n_layers, a_swa, b_full)

    # 有效滑动窗口大小：如果设置的窗口比上下文还长，那么 SWA 层实际上
    # 退化为全量注意力，缓存长度不会超过 context_length 本身。
    eff_W = min(context_length, sliding_window_size)
    L = context_length

    # Per-layer costs
    # 中文：以下分别计算“单层”在四种配置组合下的 KV 缓存字节数：
    #   MHA + 全量上下文 / GQA + 全量上下文 / MHA + SWA 窗口 / GQA + SWA 窗口
    per_mha_full = calc_kv_bytes_per_layer(batch_size, L, head_dim, n_kv_heads_mha, bytes_per_elem)
    per_gqa_full = calc_kv_bytes_per_layer(batch_size, L, head_dim, n_kv_heads_gqa, bytes_per_elem)
    per_mha_swa = calc_kv_bytes_per_layer(batch_size, eff_W, head_dim, n_kv_heads_mha, bytes_per_elem)
    per_gqa_swa = calc_kv_bytes_per_layer(batch_size, eff_W, head_dim, n_kv_heads_gqa, bytes_per_elem)

    # Totals
    # 中文：汇总整个模型（n_layers 层）的 KV 缓存总量：
    #   1) 若所有层都用全量注意力（不使用 SWA），MHA 与 GQA 各自的总显存占用；
    #   2) 若按 swa_ratio 混合排列 SWA 层与 Full 层，MHA 与 GQA 各自的总显存占用
    #      （SWA 层用较小的 eff_W 计算，Full 层仍用完整的 context_length 计算）。
    total_mha_allfull = per_mha_full * n_layers
    total_gqa_allfull = per_gqa_full * n_layers
    total_mixed_mha = n_swa_layers * per_mha_swa + n_full_layers * per_mha_full
    total_mixed_gqa = n_swa_layers * per_gqa_swa + n_full_layers * per_gqa_full

    return {
        "bytes_per_elem": bytes_per_elem,
        "head_dim": head_dim,
        "n_kv_heads_gqa": n_kv_heads_gqa,
        "eff_W": eff_W,
        "n_swa_layers": n_swa_layers,
        "n_full_layers": n_full_layers,
        "total_mha_allfull": total_mha_allfull,
        "total_gqa_allfull": total_gqa_allfull,
        "total_mixed_mha": total_mixed_mha,
        "total_mixed_gqa": total_mixed_gqa,
    }


def main():
    """
    命令行入口函数：解析用户传入的模型结构参数与运行配置，
    调用 estimate_totals 完成计算，并将配置信息和四种场景下的
    KV 缓存总显存占用（以 GB 为单位）格式化打印到终端，
    方便读者直观对比 MHA/GQA、全量注意力/SWA 混合注意力
    之间的显存开销差异。
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Estimate KV-cache memory for MHA/GQA with SWA layer ratio")
    # 上下文长度：推理时序列的最大长度，默认 1024
    p.add_argument("--context_length", default=1024, type=int)
    # 滑动窗口大小 W：SWA 层最多回看的历史 token 数，必填项
    p.add_argument("--sliding_window_size", required=True, type=int,
                   help="SWA window size W per SWA layer.")
    # 嵌入维度
    p.add_argument("--emb_dim", required=True, type=int)
    # 注意力头总数
    p.add_argument("--n_heads", required=True, type=int)
    # Transformer 总层数
    p.add_argument("--n_layers", required=True, type=int)
    # GQA 分组数：n_kv_groups=1 时，K/V 头数等于 n_heads，等价于 MHA
    p.add_argument("--n_kv_groups", required=True, type=int,
                   help="GQA groups; 1 means MHA-equivalent KV heads.")
    # 批大小，默认 1（单条序列推理）
    p.add_argument("--batch_size", default=1, type=int)
    # 数值精度，决定每个元素占用字节数，默认 fp16
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="fp16")
    # SWA:Full 层比例字符串，默认 "1:0" 表示全部层都是 SWA 层
    p.add_argument("--swa_ratio", default="1:0",
                   help="SWA:Full layer ratio. Example '5:1' -> 5 SWA for each 1 full. "
                        "'1:5' -> 1 SWA for 5 full. Default '1:0' = all SWA.")
    args = p.parse_args()

    # 把与模型结构相关的核心配置收集到一个字典里，方便后续统一打印
    cfg = {
        "context_length": args.context_length,
        "sliding_window_size": args.sliding_window_size,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_kv_groups": args.n_kv_groups,
    }

    # 调用核心估算函数，得到各类场景下的 KV 缓存字节数结果
    res = estimate_totals(
        context_length=cfg["context_length"],
        sliding_window_size=cfg["sliding_window_size"],
        emb_dim=cfg["emb_dim"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        n_kv_groups=cfg["n_kv_groups"],
        batch_size=args.batch_size,
        dtype=args.dtype,
        swa_ratio=args.swa_ratio,
    )

    # 打印配置信息，方便用户核对本次估算所依据的超参数
    print("==== Config ====")
    for k, v in cfg.items():
        print(f"{k:23}: {v}")
    print(f"batch_size             : {args.batch_size}")
    print(f"dtype                  : {args.dtype} ({res['bytes_per_elem']} Bytes/elem)")
    print(f"head_dim               : {res['head_dim']}")
    print(f"GQA n_kv_heads         : {res['n_kv_heads_gqa']}")
    print(f"Effective SWA window W : {res['eff_W']}")
    print(f"Layer ratio (SWA:Full) : {args.swa_ratio} -> "
          f"{res['n_swa_layers']} SWA, {res['n_full_layers']} Full")
    print()

    # 打印四种场景下的 KV 缓存总显存占用（换算为 GB），便于直接对比：
    #   - MHA 全量 vs GQA 全量：体现 GQA 相对 MHA 节省了多少显存
    #   - MHA+SWA vs GQA+SWA：体现在引入滑动窗口后，二者进一步节省的效果
    print("==== KV-cache totals across all layers ====")
    print(f"MHA KV total           : {convert_bytes(res['total_mha_allfull'])}")
    print(f"GQA KV total           : {convert_bytes(res['total_gqa_allfull'])}")
    print(f"MHA + SWA (ratio {args.swa_ratio})  : {convert_bytes(res['total_mixed_mha'])}")
    print(f"GQA + SWA (ratio {args.swa_ratio})  : {convert_bytes(res['total_mixed_gqa'])}")
    print()


if __name__ == "__main__":
    main()
