# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Plot KV-cache vs context length for different n_kv_groups

"""
【模块说明 / 中文补充】
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
第 4 章「分组查询注意力 GQA (Grouped-Query Attention)」配套代码的一部分。

它的作用是：调用同目录下 memory_estimator_gqa.py 中的 KV 缓存(KV cache)
显存估算函数 calc_kv_bytes_total，分别计算标准多头注意力 MHA
(Multi-Head Attention) 与分组查询注意力 GQA 在不同上下文长度
(context_length) 下所需的 KV 缓存总显存，并把结果绘制成一张对比曲线图
(kv_bytes_vs_context_length.pdf)，直观展示：
    1) 随着上下文长度增大，KV 缓存显存是如何线性增长的；
    2) GQA 通过让多个 Query 头共享同一组 Key/Value 头，
       相比 MHA 能显著压缩 KV 缓存显存占用，且压缩比例
       (n_heads / n_kv_heads) 越大，节省的显存越多。

背景知识：在自回归推理(autoregressive generation)时，为了避免每生成一个
新 token 都要重新计算前面所有 token 的 Key/Value，通常会把已经算好的
K、V 张量缓存下来（即 KV cache）。序列越长、模型层数越多、KV 头数越多，
这个缓存就越大，因此 GQA/MQA 等减少 KV 头数的技术是长上下文推理省显存
的关键手段之一。这里的绘图脚本就是用来做该权衡的可视化演示。
"""

import matplotlib.pyplot as plt

# Import from ./memory_estimator.py
# 从同目录的 memory_estimator_gqa.py 中导入：
#   - calc_kv_bytes_total: 计算「所有层加总后」KV 缓存总字节数的核心函数
#   - DTYPE_BYTES: 不同数值精度(如 bf16/fp16/fp32)对应的每个元素占用的字节数
from memory_estimator_gqa import calc_kv_bytes_total, DTYPE_BYTES


def bytes_convert(n):
    """
    将字节数转换为 GB (十进制 GB，即 1 GB = 1000^3 字节) 的字符串表示。

    参数:
        n (int/float): 字节数(bytes)。

    返回:
        str: 保留两位小数的 GB 数值字符串，例如 "1.23"（注意不带单位后缀）。
    """
    gb = n / (1000 ** 3)  # 换算成十进制 GB（区别于 1024 进制的 GiB）
    return f"{gb:.2f}"


def savings_percent(total_mha, total_gqa):
    """
    计算 GQA 相对于 MHA 在 KV 缓存显存上的节省百分比。

    参数:
        total_mha (float): MHA 方式下的 KV 缓存总字节数(或任意等价单位)。
        total_gqa (float): GQA 方式下对应的 KV 缓存总字节数(需与 total_mha
                            使用相同单位)。

    返回:
        float: 节省的百分比，取值范围理论上在 (0, 100) 之间；
               例如返回 75.0 表示 GQA 比 MHA 节省了 75% 的显存。

    说明: 该函数在本文件的绘图流程中当前并未被直接调用，
          但保留下来可用于后续扩展（比如在图上标注具体节省的百分比数值）。
    """
    return (1.0 - (total_gqa / total_mha)) * 100.0


def plot_abs_kv_vs_context_multi_groups():
    """
    绘制「KV 缓存总显存(GB) 随上下文长度(context_length)变化」的对比曲线图，
    同时展示标准 MHA 与多种分组数(n_kv_groups)下的 GQA 曲线。

    整体思路:
        1) 固定一套模型超参数(头数、嵌入维度、层数、batch、数值精度)；
        2) 在一组从 256 到 131072（呈 2 倍递增）的上下文长度上，
           分别计算 MHA 与不同 n_kv_groups 的 GQA 所需 KV 缓存总字节数；
        3) 把结果换算成 GB 并画成折线图，x 轴用对数坐标展示，
           便于同时看清短序列和超长序列下的显存差异；
        4) 最终图像保存为 PDF 文件 "kv_bytes_vs_context_length.pdf"。

    参数: 无（所有超参数都在函数内部写死，属于本例的演示配置）。
    返回值: 无（副作用是在当前工作目录生成一个 PDF 图片文件）。
    """
    # ---- 模型/推理相关超参数（仅用于本次可视化演示，非真实训练配置）----
    n_heads = 24      # 注意力头总数（Query 头数）
    emb_dim = 2048     # 模型隐藏层维度 / 嵌入维度
    n_layers = 48      # Transformer 层数（KV 缓存需要在每一层都保存一份）
    batch_size = 1     # 推理时的 batch size
    dtype = "bf16"     # KV 缓存存储的数值精度，这里用 bfloat16（2 字节/元素）
    bytes_per_elem = DTYPE_BYTES[dtype]  # 根据 dtype 查表得到每个元素占用的字节数

    # x-axis (log scale)
    # x 轴：一组呈 2 的幂次递增的上下文长度，从 256 一直到 131072(=128K)，
    # 覆盖从短文本到超长上下文的典型区间。
    context_lengths = [
        256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072
    ]

    # ---- 先计算标准 MHA（Multi-Head Attention）的 KV 缓存曲线 ----
    mha_gb = []
    for L in context_lengths:
        # MHA 的特点是：每个 Query 头都拥有独立的一份 Key/Value 头，
        # 即 n_kv_heads == n_heads，不做任何 KV 头共享/压缩。
        total_mha = calc_kv_bytes_total(
            batch_size, L, emb_dim, n_heads,
            n_heads,  # MHA: n_kv_heads = n_heads
            n_layers, bytes_per_elem
        )
        # 把字节数换算为 GB 字符串再转回 float，方便后续用 matplotlib 画图
        mha_gb.append(float(bytes_convert(total_mha)))

    plt.figure()
    # 先画出 MHA 曲线，作为对比的「基线」（KV 缓存开销最大的情形）
    plt.plot(context_lengths, mha_gb, marker="o", label="MHA (KV total)")

    # GQA curves for selected n_kv_groups
    # 依次尝试几种不同的分组数(n_kv_groups)，对比 GQA 相对 MHA 的显存节省效果。
    # n_kv_groups 越大，意味着 KV 头被压缩得越狠（多个 Query 头共用同一组 K/V）。
    groups_list = [4, 8, 12, 24]
    for g in groups_list:
        # GQA 核心思想：把 n_heads 个 Query 头划分成 g 组，
        # 每组内的所有 Query 头共享同一对 Key/Value 头，
        # 因此实际需要缓存的 KV 头数从 n_heads 降为 n_heads // g。
        # 当 g == n_heads 时（如本例的 24），等价于 MQA（Multi-Query Attention，
        # 所有 Query 头共用同一份 K/V，压缩比最大）。
        n_kv_heads = n_heads // g
        gqa_gb = []
        for L in context_lengths:
            total_gqa = calc_kv_bytes_total(
                batch_size, L, emb_dim, n_heads,
                n_kv_heads, n_layers, bytes_per_elem
            )
            gqa_gb.append(float(bytes_convert(total_gqa)))

        # Compression rate relative to MHA
        # 压缩比 = n_heads / n_kv_heads，即 GQA 相对 MHA 缓存了多少倍「更少」的 KV 头。
        # 例如 n_heads=24, n_kv_heads=6 时，压缩比为 4x，意味着 KV 缓存理论上
        # 缩小到 MHA 的 1/4（因为 head_dim、context_length、层数等其他因子不变）。
        comp = (n_heads / n_kv_heads) if n_kv_heads > 0 else float("inf")
        plt.plot(context_lengths, gqa_gb, marker="o",
                 label=f"GQA (n_kv_groups={g}, {comp:,.1f}× compression)")

    # x 轴使用对数坐标：因为 context_lengths 跨越了 256~131072（约 512 倍），
    # 线性坐标会让短序列的差异挤在一起看不清，对数坐标能均匀展示每次翻倍的效果。
    plt.xscale("log")
    plt.xlabel("context_length (log scale)")
    plt.ylabel("Total KV cache (GB)")
    plt.title(
        "KV-cache vs Context Length — MHA vs GQA (multi-group)\n"
        "(n_heads=24, emb_dim=2048, n_layers=48, batch=1, dtype=bf16)",
        fontsize=8
    )
    plt.grid(True, which="both")  # 同时显示主/次网格线，便于在对数坐标下读数
    plt.legend()
    plt.tight_layout()
    # 将最终图像保存为 PDF 矢量图，便于放入书籍/文档中不失真地缩放展示
    plt.savefig("kv_bytes_vs_context_length.pdf")


if __name__ == "__main__":
    # 脚本入口：直接运行本文件时，执行绘图函数生成对比图 PDF。
    plot_abs_kv_vs_context_multi_groups()