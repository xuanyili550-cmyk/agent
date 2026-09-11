# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# Sliding Window Attention (SWA) memory usage vs context length plot.
#
# This script mirrors the style and structure of plot_memory_estimates_mla.py.

"""
中文模块说明：
本脚本用于绘制并对比不同注意力机制在推理时 **KV 缓存（KV cache）显存占用**
随「上下文长度（context length）」增长的变化曲线，重点展示：

    1. MHA（Multi-Head Attention，多头注意力）——标准做法，每个注意力头都
       有独立的 K、V 投影，KV 缓存随序列长度线性增长。
    2. GQA（Grouped-Query Attention，分组查询注意力）——多个 Query 头共享
       同一组 K、V 头，从而把 KV 缓存按 kv_groups 的倍数缩小。
    3. SWA（Sliding Window Attention，滑动窗口注意力）——每一层只关注最近
       W 个 token（一个固定大小的窗口），而不是关注从头到尾的全部历史
       token。这样该层的 KV 缓存大小就被「封顶」在窗口大小 W，不会随着
       上下文长度无限增长。真实模型（如 Gemma、Mistral 等）通常按照某种
       比例（如 5:1）在「滑动窗口层」和「全局注意力层」之间交替排列，
       用少数全局层保留长距离信息，用大多数滑动窗口层节省显存。

    脚本还支持把 GQA 和 SWA 结合起来（GQA + SWA），因为二者是正交的优化：
    GQA 减少「每层每个 token 的 KV 大小」，SWA 减少「参与计算的 token 数量
    上限」，两者可以叠加使用。

在《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书中，
第 4 章讲解了 Transformer Block 与注意力机制的实现；本脚本属于该章节的
「配套工具脚本」，不涉及模型训练或推理本身，而是用简单的解析公式估算
不同架构选择（MHA/GQA、是否使用 SWA）在部署时对显存（KV 缓存）的影响，
帮助读者建立「架构选择 -> 显存开销」的直观认识，与
plot_memory_estimates_mla.py（对比 MLA，Multi-Head Latent Attention）
风格一致，属于同一系列的显存估算可视化工具。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Bytes per element
# 中文：不同数值精度（dtype）每个元素占用的字节数。
# KV 缓存的总显存 = 元素个数 * 每个元素的字节数，因此精度越低（如 fp8/int8），
# 显存占用越小，这也是很多推理框架用低精度存储 KV 缓存的原因。
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def convert_bytes_to_gb(n_bytes):
    """
    中文说明：
    将字节数转换为 GB（十进制 GB，即 1 GB = 1000^3 字节，而非 1024^3）。
    注意这里用的是「十进制」换算，和存储厂商标注容量的方式一致，
    并非操作系统里常见的二进制 GiB（1024^3）。

    参数：
        n_bytes: 字节数（int 或 float）。

    返回：
        float，对应的 GB 数值，用于绘图时的 y 轴数值。
    """
    return n_bytes / (1000.0 ** 3)


def parse_ratio(ratio_str):
    # "--swa_ratio a:b" means a SWA layers for every b full layers within a block
    # 中文：命令行参数 --swa_ratio 形如 "a:b"，表示在模型的层结构中，
    # 每 (a+b) 层为一个「循环单元」，其中 a 层使用滑动窗口注意力（SWA），
    # b 层使用全局（Full）注意力。例如 "5:1" 表示每 6 层里有 5 层是 SWA，
    # 1 层是全局注意力——这也是一些真实模型（如 Gemma 2）采用的比例。
    """
    中文说明：
    解析 "a:b" 格式的比例字符串，返回 (a, b) 两个非负整数，
    用于确定模型中 SWA 层与全局（Full）注意力层的层数比例。

    参数：
        ratio_str: 字符串，形如 "5:1"，a 表示 SWA 层的相对份数，
                   b 表示全局注意力层的相对份数。

    返回：
        (a, b): 两个 int，满足 a >= 0, b >= 0 且 a + b > 0。

    异常：
        若字符串格式不合法（无法按 ":" 拆分成两个整数，或 a+b == 0），
        会抛出 ValueError，提示正确的输入格式。
    """
    try:
        a_str, b_str = ratio_str.split(":")
        a, b = int(a_str), int(b_str)
        assert a >= 0 and b >= 0 and (a + b) > 0
        return a, b
    except Exception:
        raise ValueError("--swa_ratio must be in the form 'a:b' with nonnegative integers and a+b>0")


def calc_kv_bytes_total_mha(batch, context_length, emb_dim, n_layers, bytes_per_elem):
    # For MHA, n_kv_heads = n_heads, which cancels out:
    # total = B * L * E * 2 (K,V) * bytes * n_layers
    # 中文：标准多头注意力（MHA）中，K 和 V 的头数与 Query 的头数相同
    # （n_kv_heads = n_heads），因此在按 emb_dim 计算总参数量时，
    # 「每个头的维度 * 头数」正好约掉，直接等于 emb_dim。
    # 每个 token 在每一层需要缓存 K 和 V 两份，形状概念上相当于
    # (batch, context_length, emb_dim) 的 K 和 V 各一份，
    # 所以总字节数 = batch * context_length * emb_dim * 2(K和V) * 每元素字节数 * 层数。
    """
    中文说明：
    计算标准 MHA（多头注意力）在推理时，**全部 n_layers 层**的 KV 缓存
    总字节数（未做任何压缩，如 GQA 或 SWA）。

    参数：
        batch: int，推理时的 batch size（并发请求数）。
        context_length: int，当前已缓存的上下文长度（token 数），
                         对应 KV 缓存中「序列长度」这个维度。
        emb_dim: int，模型的隐藏层维度（embedding dimension），
                 对 MHA 来说它等于 n_heads * head_dim。
        n_layers: int，Transformer 的层数，每一层都有独立的 KV 缓存。
        bytes_per_elem: int，每个数值元素占用的字节数（由 dtype 决定）。

    返回：
        float/int，全部层的 K+V 缓存总字节数。
        概念上对应张量形状 (n_layers, 2, batch, context_length, emb_dim)
        展开后的元素总数，再乘以每元素字节数。
    """
    return batch * context_length * emb_dim * 2 * bytes_per_elem * n_layers


def calc_kv_bytes_total_gqa(
    batch, context_length, emb_dim, n_layers, bytes_per_elem, n_kv_groups
):
    # For GQA, n_kv_heads = n_heads / n_kv_groups
    # => scale the MHA total by 1 / n_kv_groups
    # 中文：分组查询注意力（GQA）让多个 Query 头共享同一组 K/V 头，
    # 即 n_kv_heads = n_heads / n_kv_groups（kv_groups 个 Query 头共用 1 组 KV）。
    # 因为 KV 缓存的大小只取决于 KV 头的数量（而不是 Query 头数量），
    # 所以只需把 MHA 的总显存直接除以 n_kv_groups，就得到 GQA 的 KV 缓存总量。
    """
    中文说明：
    计算 GQA（分组查询注意力）在推理时，**全部 n_layers 层**的 KV 缓存
    总字节数。GQA 通过让多个 Query 头共享一组 K/V，来压缩 KV 缓存大小。

    参数：
        batch: int，batch size。
        context_length: int，上下文长度（token 数）。
        emb_dim: int，隐藏层维度。
        n_layers: int，层数。
        bytes_per_elem: int，每元素字节数。
        n_kv_groups: int，KV 分组数（多少个 Query 头共享一组 K/V），
                     等价于「压缩倍数」，n_kv_groups 越大，KV 缓存越小。

    返回：
        float/int，全部层的 K+V 缓存总字节数，
        等于 calc_kv_bytes_total_mha 的结果除以 n_kv_groups。
    """
    base = calc_kv_bytes_total_mha(batch, context_length, emb_dim, n_layers, bytes_per_elem)
    return base / n_kv_groups


def calc_kv_bytes_total_mha_swa(
    batch, context_length, emb_dim, n_layers, bytes_per_elem, window, swa_ratio
):
    # Split layers into SWA vs Full
    # 中文：把 n_layers 层按 swa_ratio（如 "5:1"）拆分成两部分：
    # 一部分是滑动窗口注意力层（n_swa_layers），另一部分是全局注意力层
    # （n_full_layers）。滑动窗口层的「有效上下文长度」被限制在窗口大小
    # window 以内（不管真实上下文多长，KV 缓存都不会超过 window），
    # 而全局注意力层仍然按完整的 context_length 缓存 KV。
    """
    中文说明：
    计算「MHA + 滑动窗口注意力（SWA）」混合架构下，全部层的 KV 缓存
    总字节数。模型中一部分层使用 SWA（KV 缓存被窗口大小 window 封顶），
    另一部分层使用全局注意力（KV 缓存随 context_length 线性增长）。

    参数：
        batch: int，batch size。
        context_length: int，实际上下文长度（真实序列长度）。
        emb_dim: int，隐藏层维度。
        n_layers: int，模型总层数。
        bytes_per_elem: int，每元素字节数。
        window: int，滑动窗口大小 W，即 SWA 层最多能看到的历史 token 数，
                也就是 SWA 层 KV 缓存的「有效序列长度」上限。
        swa_ratio: str，形如 "a:b" 的比例字符串，决定 SWA 层与全局层的层数比例。

    返回：
        float/int，全部层（SWA 层 + 全局层）的 KV 缓存总字节数之和。
    """
    a, b = parse_ratio(swa_ratio)
    total_blocks = a + b
    # 按比例把总层数分配给 SWA 层，四舍五入取整（层数必须是整数）
    n_swa_layers = int(round(n_layers * (a / total_blocks)))
    # 剩下的层数分配给全局（Full）注意力层
    n_full_layers = n_layers - n_swa_layers

    # 全局层：KV 缓存按真实的 context_length 计算，随上下文长度线性增长
    total_full = calc_kv_bytes_total_mha(
        batch, context_length, emb_dim, n_full_layers, bytes_per_elem
    )
    # SWA 层：KV 缓存按窗口大小 window 计算（而不是 context_length），
    # 因此当 context_length 超过 window 之后，SWA 层的显存占用不再增长，
    # 这正是滑动窗口注意力能节省显存的核心原因。
    total_swa = calc_kv_bytes_total_mha(
        batch, window, emb_dim, n_swa_layers, bytes_per_elem
    )
    return total_full + total_swa


def calc_kv_bytes_total_gqa_swa(
    batch,
    context_length,
    emb_dim,
    n_layers,
    bytes_per_elem,
    n_kv_groups,
    window,
    swa_ratio,
):
    """
    中文说明：
    计算「GQA + 滑动窗口注意力（SWA）」组合架构下，全部层的 KV 缓存
    总字节数。这是 GQA（压缩每层 KV 大小）与 SWA（限制参与计算的
    token 数量上限）两种优化手段的叠加，也是很多现代高效模型
    （如部分 Gemma/Mistral 变体）实际采用的设计。

    参数：
        batch: int，batch size。
        context_length: int，真实上下文长度。
        emb_dim: int，隐藏层维度。
        n_layers: int，模型总层数。
        bytes_per_elem: int，每元素字节数。
        n_kv_groups: int，GQA 的 KV 分组数（压缩倍数）。
        window: int，滑动窗口大小 W。
        swa_ratio: str，"a:b" 格式，SWA 层与全局层的比例。

    返回：
        float/int，全部层（GQA-SWA 层 + GQA-全局层）的 KV 缓存总字节数之和。
    """
    a, b = parse_ratio(swa_ratio)
    total_blocks = a + b
    # 同样按比例拆分 SWA 层与全局层的层数
    n_swa_layers = int(round(n_layers * (a / total_blocks)))
    n_full_layers = n_layers - n_swa_layers

    # 全局层：在 GQA 压缩的基础上，按真实 context_length 计算 KV 缓存
    total_full = calc_kv_bytes_total_gqa(
        batch,
        context_length,
        emb_dim,
        n_full_layers,
        bytes_per_elem,
        n_kv_groups,
    )
    # SWA 层：在 GQA 压缩的基础上，再叠加窗口封顶（用 window 代替 context_length）
    total_swa = calc_kv_bytes_total_gqa(
        batch, window, emb_dim, n_swa_layers, bytes_per_elem, n_kv_groups
    )
    return total_full + total_swa


def main():
    """
    中文说明：
    脚本主入口，完成以下工作：
        1. 解析命令行参数（模型结构超参数、dtype、滑动窗口大小、SWA 比例、
           输出文件路径等）。
        2. 针对一组预设的上下文长度（从 256 到 131072，指数增长），
           分别计算 MHA、GQA（若 n_heads 能被 4 整除）及它们各自叠加
           SWA 后的 KV 缓存总量（转换为 GB）。
        3. 用 matplotlib 绘制「KV 缓存 (GB) vs 上下文长度（对数坐标）」的
           折线图，MHA 与 GQA 用不同颜色区分，普通版本用实线、
           SWA 版本用虚线，便于直观对比「是否使用 SWA」带来的显存节省。
        4. 保存图像到指定路径，并打印提示信息。

    参数：无（从命令行读取）。
    返回：无（函数以生成图片文件、打印日志为副作用，不返回值）。
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="KV-cache vs Context Length — MHA vs GQA with SWA overlays"
    )
    p.add_argument("--emb_dim", type=int, required=True)
    p.add_argument("--n_heads", type=int, required=True)
    p.add_argument("--n_layers", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bf16")
    p.add_argument(
        "--sliding_window_size", type=int, required=True, help="SWA window size W"
    )
    p.add_argument("--swa_ratio", type=str, default="5:1", help="SWA:Full ratio, e.g., 5:1")
    p.add_argument(
        "--output", type=Path, default=Path("kv_bytes_vs_context_length.pdf")
    )
    args = p.parse_args()

    batch_size = args.batch_size
    emb_dim = args.emb_dim
    n_heads = args.n_heads
    n_layers = args.n_layers
    bytes_per_elem = DTYPE_BYTES[args.dtype]

    # 中文：这里把 GQA 的分组数固定写死为 4（即每 4 个 Query 头共享 1 组 K/V），
    # 这是一个常见的经验设置（不是从命令行读取的可调参数）。
    kv_groups = 4
    # 只有当 n_heads 能被 kv_groups 整除时，GQA 的头分组才有意义
    # （否则无法把 n_heads 平均分成 kv_groups 组），所以先做合法性检查。
    valid_g4 = (n_heads % kv_groups == 0)

    # 中文：预设一组从小到大、按 2 的幂次增长的上下文长度，
    # 用作图像的 x 轴取样点，覆盖从很短（256）到很长（131072，约 128K）的场景，
    # 这样在对数坐标下能清晰展示 KV 缓存随上下文长度的增长趋势。
    context_lengths = [
        256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072
    ]

    # 中文：series 是一个字典，key 为图例标签，value 为对应曲线在各个
    # context_length 下的 y 值（单位 GB）列表。先固定加入 MHA 与
    # 「SWA on MHA」两条曲线；若 valid_g4 为真，再加入 GQA 与
    # 「SWA on GQA」两条曲线，一共最多 4 条曲线。
    series = {
        "MHA (KV total)": [],
        f"SWA on MHA (ratio {args.swa_ratio}, W={args.sliding_window_size})": [],
    }
    if valid_g4:
        series["GQA kv_groups=4 (full)"] = []
        series[
            f"SWA on GQA kv_groups=4 (ratio {args.swa_ratio}, W={args.sliding_window_size})"
        ] = []

    # 中文：逐个上下文长度 L，计算各架构对应的 KV 缓存总量，并追加到对应曲线的
    # 数据列表中，最终得到与 context_lengths 等长的 y 值序列。
    for L in context_lengths:
        # 标准 MHA，不做任何压缩，KV 缓存随 L 线性增长
        total_mha = calc_kv_bytes_total_mha(
            batch_size, L, emb_dim, n_layers, bytes_per_elem
        )
        # MHA + SWA：部分层的 KV 缓存被窗口大小封顶，整体增长趋势会趋于平缓
        total_mha_swa = calc_kv_bytes_total_mha_swa(
            batch_size,
            L,
            emb_dim,
            n_layers,
            bytes_per_elem,
            window=args.sliding_window_size,
            swa_ratio=args.swa_ratio,
        )
        # 字节数转换为 GB，方便在图上直接读数
        series["MHA (KV total)"].append(convert_bytes_to_gb(total_mha))
        series[
            f"SWA on MHA (ratio {args.swa_ratio}, W={args.sliding_window_size})"
        ].append(convert_bytes_to_gb(total_mha_swa))

        if valid_g4:
            # GQA（kv_groups=4），在 MHA 基础上按分组数压缩 KV 缓存
            total_gqa = calc_kv_bytes_total_gqa(
                batch_size, L, emb_dim, n_layers, bytes_per_elem, n_kv_groups=kv_groups
            )
            # GQA + SWA：两种压缩手段叠加，理论上显存占用最小
            total_gqa_swa = calc_kv_bytes_total_gqa_swa(
                batch_size,
                L,
                emb_dim,
                n_layers,
                bytes_per_elem,
                n_kv_groups=kv_groups,
                window=args.sliding_window_size,
                swa_ratio=args.swa_ratio,
            )
            series["GQA kv_groups=4 (full)"].append(convert_bytes_to_gb(total_gqa))
            series[
                f"SWA on GQA kv_groups=4 (ratio {args.swa_ratio}, W={args.sliding_window_size})"
            ].append(convert_bytes_to_gb(total_gqa_swa))

    # 中文：以下开始绘图部分。图像大小 10x5 英寸，x 轴使用对数坐标
    # （因为 context_lengths 是指数增长的），便于在同一张图里同时看清
    # 短序列和长序列（128K）区间的曲线形态。
    plt.figure(figsize=(10, 5))
    x = np.array(context_lengths, dtype=float)

    # 中文：为 MHA 系列和 GQA 系列各自指定一个固定颜色，
    # 这样「普通版」和「SWA 版」曲线即便线型不同（实线/虚线），
    # 只要属于同一注意力家族（MHA 或 GQA），颜色也保持一致，便于配对比较。
    colors = {
        "MHA": "#1f77b4",
        "GQA": "#ff7f0e",
    }

    for label, yvals in series.items():
        y = np.array(yvals, dtype=float)
        if np.all(np.isnan(y)):
            # 中文：若某条曲线的所有 y 值都是 NaN（当前逻辑下基本不会触发，
            # 属于防御性判断），则跳过绘制，避免在图上留下无意义的空线。
            continue

        # 中文：标签中含有 "SWA" 的曲线用虚线("--")表示「使用了滑动窗口注意力」，
        # 未使用 SWA 的曲线用实线("-")，一眼就能区分「压缩前/压缩后」。
        linestyle = "--" if "SWA" in label else "-"
        if "MHA" in label:
            color = colors["MHA"]
        elif "GQA" in label:
            color = colors["GQA"]
        else:
            color = None

        plt.plot(x, y, marker="o", label=label, linestyle=linestyle, color=color)

    # 中文：x 轴使用对数刻度，因为 context_lengths 是从 256 到 131072
    # 按倍数增长的，线性坐标会让前面的点挤在一起，对数坐标能均匀展示。
    plt.xscale("log")
    plt.xlabel("context_length (log scale)")
    plt.ylabel("Total KV cache (GB)")
    plt.title(
        "KV-cache vs Context Length — MHA vs GQA (SWA overlays)\n"
        f"(n_heads={n_heads}, emb_dim={emb_dim}, n_layers={n_layers}, "
        f"batch={batch_size}, dtype={args.dtype}; "
        f"SWA ratio={args.swa_ratio}, W={args.sliding_window_size})",
        fontsize=8,
    )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    # 中文：将图像保存为文件（默认 PDF 格式），保存后立即关闭画布释放资源
    plt.savefig(args.output)
    plt.close()

    if not valid_g4:
        # 中文：若 n_heads 不能被 4 整除，说明 kv_groups=4 的 GQA 配置不合法，
        # 前面已经跳过了 GQA 相关曲线的计算，这里给出提示，避免用户误以为脚本出错。
        print(
            f"Skipped GQA kv_groups=4 because n_heads={args.n_heads} "
            "is not divisible by 4."
        )
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
