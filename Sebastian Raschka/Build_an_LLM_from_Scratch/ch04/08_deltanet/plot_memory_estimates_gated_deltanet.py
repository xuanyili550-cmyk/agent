# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本文件用途（中文说明）
====================
这是一个「推理阶段内存占用估算与可视化」脚本，属于《从零构建大语言模型》
（Build a Large Language Model From Scratch）第 4 章 "08_deltanet" 示例代码的
配套工具，用于直观展示两种不同注意力/序列建模机制在自回归推理时所需的
KV 缓存（KV cache）显存开销随上下文长度（context length）增长的变化趋势：

1. 传统的全量多头注意力（Full Multi-Head Attention, MHA）
   —— 其 KV 缓存大小随上下文长度线性增长，是长上下文推理时显存爆炸的主因。
2. Gated DeltaNet（一种线性注意力/状态空间类的高效序列建模结构）
   —— 它用一个固定大小的「状态矩阵」（state matrix）来递推更新，
      因此其显存占用是**常数**，不随上下文长度增长而增长。
3. 二者按 3:1 的层数比例混合（3 层 DeltaNet : 1 层 MHA）的混合架构
   —— 这是许多现代长上下文大模型（如 Jamba、MiniMax、Gated DeltaNet 论文中提到的
      混合架构）采用的折中方案：既能保留部分全局注意力的建模能力，
      又能大幅降低 KV 缓存显存需求。

脚本会计算三种方案在不同上下文长度下的 KV 缓存显存占用（单位：GB），
并绘制成折线图保存为 PDF 文件（deltanet_memory_plot.pdf），
帮助读者直观理解「线性注意力 / 状态空间模型」相比「标准注意力」在
长序列推理内存效率上的巨大优势，这也是本书讨论高效 LLM 架构设计时的
一个重要教学示例。

注意：本文件只做「理论显存估算」，不涉及真实模型的前向/反向传播，
因此代码中不包含张量运算（PyTorch）本身，而是用 NumPy 做纯数值计算。
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# Bytes per element
# 中文：不同数值精度（dtype）下，每个元素占用的字节数。
# 这决定了同样形状的张量在不同精度下的显存占用倍数关系，
# 例如从 fp32 (4 字节) 转为 bf16/fp16 (2 字节) 可以直接把显存占用减半，
# 再转为 fp8/int8 (1 字节) 则可以减到四分之一。
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def calc_kv_bytes_total_mha(batch, context_length, emb_dim, n_layers, bytes_per_elem, n_heads):
    """计算「标准全量多头注意力（MHA）」在自回归推理时，全部层的 KV 缓存总字节数。

    背景知识：在自回归推理（一次生成一个 token）时，为了避免重复计算历史
    token 的 Key/Value 投影，模型会把每一层、每一个注意力头在每个时间步
    产生的 K 和 V 向量缓存下来，这就是所谓的 KV cache。它的大小随着
    已生成 token 数（即 context_length）线性增长，是长上下文推理显存
    占用的主要来源。

    参数：
        batch (int): 批大小（batch size），即同时推理的序列条数。
        context_length (int): 当前上下文长度（已缓存的 token 数量）。
        emb_dim (int): 模型的嵌入维度（embedding dimension），
            即所有注意力头拼接后的总维度。
        n_layers (int): Transformer 层数（这里对应“使用 MHA 的层数”）。
        bytes_per_elem (int): 每个张量元素占用的字节数，取决于 dtype
            （例如 bf16 为 2 字节）。
        n_heads (int): 多头注意力的头数。

    返回：
        float/int: 所有层的 K + V 缓存总字节数（未换算成 GB）。

    形状推导：
        每个注意力头的维度 d_head = emb_dim // n_heads。
        单层 KV 缓存的张量形状可理解为
            K: (batch, n_heads, context_length, d_head)
            V: (batch, n_heads, context_length, d_head)
        元素总数 = batch * n_heads * context_length * d_head，
        因为 K 和 V 各占一份，所以乘以系数 2，再乘以每元素字节数
        得到单层字节数，最后乘以层数 n_layers 得到总字节数。
    """
    # Full attention (MHA)
    # 中文：先算出每个注意力头的维度 d_head（整除，假设 emb_dim 能被 n_heads 整除）。
    d_head = emb_dim // n_heads
    # 中文：单层 KV 缓存占用 = batch * 序列长度 * 头数 * 每头维度 * 2（K和V两份）* 每元素字节数。
    # 关键点：这里的 context_length 是自变量，说明 KV 缓存与上下文长度成「线性关系」。
    per_layer = batch * context_length * n_heads * d_head * 2 * bytes_per_elem
    # 中文：乘以层数，得到整个模型（所有 MHA 层）的 KV 缓存总字节数。
    return per_layer * n_layers


def calc_kv_bytes_total_deltanet_no_conv(batch, emb_dim, n_layers, bytes_per_elem, n_heads):
    """计算「简化版 Gated DeltaNet（不含卷积混合）」在推理时，全部层的状态缓存总字节数。

    核心区别（教学要点）：DeltaNet 属于线性注意力 / 循环状态空间模型的一类，
    它不像标准注意力那样把每个历史 token 的 K/V 都单独存下来，而是把所有
    历史信息「压缩」进一个固定大小的状态矩阵（state matrix），每来一个新
    token 就用类似 RNN 的方式对这个状态矩阵做一次「增量更新（delta update）」，
    因此这个状态矩阵的大小**与上下文长度无关**——这正是它相比 MHA 在长序列
    推理时显存优势的根本原因。

    参数：
        batch (int): 批大小。
        emb_dim (int): 嵌入维度。
        n_layers (int): 使用 DeltaNet 结构的层数。
        bytes_per_elem (int): 每元素字节数，取决于 dtype。
        n_heads (int): 头数（DeltaNet 通常也是多头结构，每个头维护一个独立的状态矩阵）。

    返回：
        float/int: 所有 DeltaNet 层的状态矩阵缓存总字节数（未换算成 GB）。

    形状推导：
        每头维度 d_head = emb_dim // n_heads。
        每个头维护一个形状为 (d_head, d_head) 的状态矩阵（类似一个可递推更新的
        「联想记忆矩阵」），因此单层状态张量形状可理解为
            State: (batch, n_heads, d_head, d_head)
        元素总数 = batch * n_heads * d_head * d_head，
        注意这里**不包含 context_length**——这就是常数级显存占用的数学体现。
    """
    # Simple Gated DeltaNet (no convolutional mixing)
    # 中文：同样先算出每头维度。
    d_head = emb_dim // n_heads
    # 中文：单层状态矩阵占用 = batch * 头数 * d_head * d_head * 每元素字节数。
    # 关键点：这里没有 context_length 这个变量参与计算，
    # 说明无论上下文多长，DeltaNet 每层的状态缓存大小都是固定的（O(1)）。
    per_layer = batch * n_heads * d_head * d_head * bytes_per_elem
    # 中文：乘以层数，得到所有 DeltaNet 层的状态缓存总字节数。
    return per_layer * n_layers


def convert_to_gb(x):
    """将字节数（bytes）换算为吉字节（GB），便于绘图和阅读。

    参数：
        x (float 或 np.ndarray): 字节数，可以是标量或数组。

    返回：
        与输入同类型：换算后的 GB 数值（除以 1e9，采用十进制 GB 定义，
        而非 1024 进制的 GiB）。
    """
    return x / 1e9


def main():
    """脚本主入口：解析命令行参数，计算三种方案的显存占用曲线，并绘图保存为 PDF。

    整体流程：
        1. 解析命令行参数（批大小、嵌入维度、头数、层数、数值精度、
           上下文长度扫描范围等）。
        2. 构造一系列上下文长度取值 ctx（从 min_ctx 到 max_ctx，步长为 100）。
        3. 分别计算：
           a) 纯 MHA 架构的 KV 缓存显存（随 ctx 增长）；
           b) 纯 DeltaNet 架构的状态缓存显存（为常数，不随 ctx 变化）；
           c) 3:1 混合架构（3 层 DeltaNet : 1 层 MHA）的总显存（随 ctx 增长，
              但斜率只有纯 MHA 的 1/4，因为只有 1/4 的层使用了会随长度增长的 MHA）。
        4. 将三条曲线的显存单位从字节换算为 GB。
        5. 用 matplotlib 绘制折线图，并保存为 "deltanet_memory_plot.pdf"。

    参数：无（命令行参数通过 argparse 从 sys.argv 中读取）。

    返回：无（函数以副作用形式将图像写入磁盘文件）。
    """
    # 中文：构建命令行参数解析器，使用 ArgumentDefaultsHelpFormatter
    # 可以在 --help 中自动显示每个参数的默认值，方便用户查阅。
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Memory vs. Context Length: MHA vs. DeltaNet (3:1 mix)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--emb_dim", type=int, default=2048)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=48)
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bf16")
    p.add_argument("--min_ctx", type=int, default=128)
    p.add_argument("--max_ctx", type=int, default=131_072)
    args = p.parse_args()

    # 中文：横轴的步长，即每隔 100 个 token 采样一个上下文长度点，
    # 用于绘制平滑的曲线，同时避免逐 token 计算带来的性能浪费。
    step = 100
    # 中文：生成从 min_ctx 到 max_ctx（含）之间，步长为 step 的上下文长度数组，
    # 这将作为绘图的横轴（x 轴）取值。
    ctx = np.arange(args.min_ctx, args.max_ctx + 1, step, dtype=int)
    # 中文：根据用户选择的数值精度（dtype），查表得到每个元素占用的字节数，
    # 这个值会直接影响后续所有显存计算结果的数量级。
    bytes_per_elem = DTYPE_BYTES[args.dtype]

    # 1) Full attention only
    # 中文：方案一 —— 假设全部 n_layers 层都使用标准 MHA，
    # 对 ctx 数组中的每一个上下文长度分别计算对应的 KV 缓存总字节数，
    # 得到一条随上下文长度线性增长的曲线（数组形状为 (len(ctx),)）。
    mha_bytes = np.array([
        calc_kv_bytes_total_mha(args.batch, int(t), args.emb_dim, args.n_layers,
                                     bytes_per_elem, args.n_heads)
        for t in ctx
    ], dtype=float)

    # 2) DeltaNet only
    # 中文：方案二 —— 假设全部 n_layers 层都使用 DeltaNet。
    # 由于 DeltaNet 的状态缓存与上下文长度无关，这里只需算一次常数值，
    dnet_bytes_const = calc_kv_bytes_total_deltanet_no_conv(
        args.batch, args.emb_dim, args.n_layers,
        bytes_per_elem, args.n_heads
    )
    # 中文：然后用 np.full_like 把这个常数值广播（复制）到与 mha_bytes 相同长度的数组中，
    # 这样绘图时就能得到一条水平直线，直观体现「常数级显存占用」这一特性。
    dnet_bytes = np.full_like(mha_bytes, fill_value=dnet_bytes_const, dtype=float)

    # 3) 3:1 layer ratio (3 DeltaNet : 1 Full Attention)
    # 中文：方案三 —— 混合架构：每 4 层中有 1 层是 MHA，3 层是 DeltaNet
    # （这是很多长上下文混合模型采用的经验比例，兼顾全局建模能力与显存效率）。
    n_mha_layers = args.n_layers / 4
    n_dnet_layers = args.n_layers - n_mha_layers
    # 中文：对每个上下文长度，分别计算「n_mha_layers 层 MHA 的 KV 缓存」
    # 与「n_dnet_layers 层 DeltaNet 的状态缓存」之和，
    # 得到混合架构总显存曲线——它也会随上下文增长，但增长斜率只有纯 MHA 方案的 1/4，
    # 因为只有四分之一的层贡献了会随长度线性增长的部分。
    mix_bytes = np.array([
        calc_kv_bytes_total_mha(args.batch, int(t), args.emb_dim, n_mha_layers,
                                     bytes_per_elem, args.n_heads)
        + calc_kv_bytes_total_deltanet_no_conv(args.batch, args.emb_dim, n_dnet_layers,
                                                    bytes_per_elem, args.n_heads)
        for t in ctx
    ], dtype=float)

    # Convert to GB
    # 中文：把三条曲线的显存占用统一从字节换算为 GB，方便在图上直接读数（人类可读单位）。
    mha_gb = convert_to_gb(mha_bytes)
    dnet_gb = convert_to_gb(dnet_bytes)
    mix_gb = convert_to_gb(mix_bytes)

    # Plot
    # 中文：创建一个 7x4.5 英寸大小的图表画布。
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # 中文：分别绘制三条曲线：
    # 1) 纯 MHA —— 应呈现明显的线性增长趋势；
    ax.plot(ctx, mha_gb, label="Full Attention (MHA) KV cache")
    # 2) 纯 DeltaNet —— 应呈现一条水平直线（常数显存占用）；
    ax.plot(ctx, dnet_gb, label="All Gated DeltaNet (no conv)")
    # 3) 3:1 混合架构 —— 介于二者之间，斜率明显更平缓。
    ax.plot(ctx, mix_gb, label="3:1 layer ratio (3 DeltaNet : 1 Full Attention)")

    # 中文：设置坐标轴标签、网格线（便于读数）以及图例说明。
    ax.set_xlabel("Context length (number of tokens)")
    ax.set_ylabel("KV cache size (GB)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    # 中文：自动调整子图布局，避免标签被裁剪；
    fig.tight_layout()
    # 中文：将图像保存为 PDF 矢量图文件，dpi 参数在矢量格式下主要影响内嵌位图元素的分辨率；
    plt.savefig("deltanet_memory_plot.pdf", dpi=160)
    # 中文：关闭图形对象，释放内存资源（在脚本式批量绘图中是良好实践）。
    plt.close(fig)


if __name__ == "__main__":
    # 中文：仅当该脚本被直接运行（而非作为模块被 import）时，才执行 main() 函数，
    # 这是 Python 脚本的标准入口写法。
    main()
