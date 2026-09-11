# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
中文模块说明
============
本脚本用于绘制并对比两种注意力机制在推理时的 **KV 缓存（KV cache）显存占用**：

1. **MHA（Multi-Head Attention，多头注意力）**：标准的多头注意力，每一层、每个注意力头都要
   缓存各自独立的 Key 和 Value 张量，随着上下文长度（context_length）增长，显存占用线性增长，
   且与头数、每头维度直接相关。
2. **MLA（Multi-Head Latent Attention，多头潜在注意力）**：DeepSeek-V2/V3 等模型中使用的技术，
   核心思想是把每一层要缓存的 Key/Value 压缩进一个低维的“潜在向量”（latent vector，维度为
   latent_dim），而不是像 MHA 那样为每个头单独存一份完整的 K、V。这样可以大幅降低 KV 缓存的
   显存占用，尤其是在超长上下文（如 128K token）场景下效果显著。

在《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书中，本脚本属于
第 4 章（Transformer 架构 / 注意力机制）关于 MLA 的补充材料，用于直观展示：随着上下文长度
增加，MLA 相比传统 MHA 能节省多少倍的 KV 缓存显存，从而帮助读者理解为什么现代大模型
（如 DeepSeek 系列）会采用 MLA 这种注意力变体来降低推理成本。

脚本运行后会生成一张对数横轴（context_length 为对数刻度）的折线图，纵轴为 KV 缓存总大小
（单位 GB），并保存为 PDF 文件 kv_bytes_vs_context_length.pdf。
"""

import matplotlib.pyplot as plt

# Bytes per element
# 不同数值精度（dtype）下，每个元素占用的字节数。
# 例如 bf16/fp16 是半精度浮点数，每个数占 2 字节；fp32 是单精度浮点数，占 4 字节；
# fp8/int8 是 8 位表示，只占 1 字节。精度越低，显存占用越小，但数值表示范围/精度也越低。
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def convert_bytes_to_gb(n_bytes):
    """将字节数（bytes）换算为吉字节（GB）。

    参数：
        n_bytes: 原始字节数（int 或 float）。

    返回：
        float，换算后的 GB 数值。这里采用 1 GB = 1000^3 bytes 的十进制换算
        （而不是 1024^3 的二进制换算），是显存/存储行业常见的近似写法，
        用于图表展示时数值更直观。
    """
    return n_bytes / (1000. ** 3)


def calc_kv_bytes_total_mha(batch, context_length, emb_dim, n_heads,
                                 n_layers, bytes_per_elem):
    """计算标准 MHA（多头注意力）在推理时，整个模型所需的 KV 缓存总字节数。

    在自回归生成（autoregressive generation）时，为了避免每一步都重新计算
    历史 token 的 Key/Value，模型会把它们缓存下来（即 KV cache）。对于标准
    MHA，每一层、每个注意力头都要各自缓存一份完整的 K 和 V，因此显存占用
    会随着「层数 × 头数 × 每头维度 × 上下文长度」线性增长。

    参数：
        batch: 批次大小（batch size）。
        context_length: 需要缓存的上下文（token）长度，即缓存的序列维度 seq_len。
        emb_dim: 模型的嵌入维度（embedding dimension），也就是每层输入/输出的
            总隐藏维度。
        n_heads: 多头注意力的头数。
        n_layers: Transformer 的层数（每一层都各自维护一份 KV 缓存）。
        bytes_per_elem: 每个数值元素占用的字节数（由 dtype 决定，见 DTYPE_BYTES）。

    返回：
        int/float，整个模型（所有层）KV 缓存总共占用的字节数。

    形状/维度说明：
        单层的 K（或 V）张量形状约为 (batch, n_heads, context_length, head_dim)，
        其中 head_dim = emb_dim / n_heads。因为 K 和 V 各存一份，所以下面
        乘以 2；再乘以层数 n_layers 得到整个模型的总缓存量。
    """
    # 每个注意力头的维度 = 总嵌入维度 / 头数（MHA 中各头维度均分总维度）
    head_dim = emb_dim / n_heads
    # 单层 KV 缓存字节数：
    #   batch × context_length × head_dim × n_heads —— 这其实就是 batch × context_length × emb_dim，
    #   代表一份完整 K（或 V）张量的元素个数；
    #   × 2 —— 因为 K 和 V 各要存一份；
    #   × bytes_per_elem —— 换算成实际字节数。
    per_layer = batch * context_length * head_dim * n_heads * 2 * bytes_per_elem
    # 乘以层数，得到整个模型的 KV 缓存总字节数（每一层都要独立缓存自己的 K/V）
    return per_layer * n_layers


def calc_kv_bytes_total_mla(batch, context_length, n_layers, latent_dim, bytes_per_elem):
    """计算 MLA（Multi-Head Latent Attention，多头潜在注意力）的 KV 缓存总字节数。

    MLA 的核心思想：不再像 MHA 那样为每个头分别缓存完整的 K、V，而是把每个
    token 需要缓存的信息压缩进一个统一的低维「潜在向量」（latent vector），
    维度为 latent_dim（通常远小于 emb_dim，也不再需要乘以头数或再乘以 2）。
    推理时再由这个潜在向量动态还原/投影出各头需要的 K、V。这样一来，缓存量
    只与 latent_dim 成正比，与头数 n_heads、每头维度无关，从而大幅降低显存占用。

    参数：
        batch: 批次大小。
        context_length: 上下文（token）长度，对应缓存的序列维度 seq_len。
        n_layers: Transformer 层数（每层仍需各自维护一份潜在向量缓存）。
        latent_dim: 压缩后的潜在向量维度，是 MLA 相比 MHA 能省显存的关键参数，
            latent_dim 越小，压缩率越高，但可能损失一定的表达能力。
        bytes_per_elem: 每个数值元素占用的字节数（取决于 dtype）。

    返回：
        int/float，整个模型（所有层）MLA 潜在向量缓存总共占用的字节数。

    形状/维度说明：
        每层缓存的潜在向量形状约为 (batch, context_length, latent_dim)，
        注意这里不再有 “×2”（K、V 分开存）也不再有 “×n_heads”，
        这正是 MLA 能大幅压缩 KV 缓存的关键所在。
    """
    return batch * context_length * n_layers * latent_dim * bytes_per_elem


def plot_abs_kv_vs_context_multiple():
    """绘制「KV 缓存显存占用 随 上下文长度（context_length）变化」的对比折线图。

    该函数没有参数、没有返回值，其副作用是：
      1. 基于一组固定的模型超参数（头数、嵌入维度、层数、批大小、数值精度），
         计算标准 MHA 在一系列上下文长度下的 KV 缓存总量（单位 GB）；
      2. 针对多个不同的 latent_dim 取值，计算 MLA 方案下对应的 KV 缓存总量，
         并计算相对于 MHA 在最大上下文长度处的「压缩倍数」；
      3. 把 MHA 曲线和多条 MLA 曲线画在同一张图上（横轴取对数刻度），
         直观展示随着上下文变长，MLA 相比 MHA 能节省多少倍显存；
      4. 将图保存为 PDF 文件 "kv_bytes_vs_context_length.pdf"。

    这张图的教学意义：帮助读者直观理解——上下文越长，MHA 的 KV 缓存膨胀得
    越厉害，而 MLA 由于把每层缓存压缩到一个低维潜在向量，增长速度依然是
    线性的，但斜率（即每增加一个 token 所需的额外显存）远小于 MHA，
    latent_dim 越小，节省的显存越多（压缩倍数越大）。
    """
    # ------ 固定的模型超参数（模拟一个中等规模的 Transformer 语言模型）------
    n_heads = 24        # 注意力头数
    emb_dim = 2048       # 嵌入维度（隐藏层维度）
    n_layers = 48        # Transformer 层数
    batch_size = 1       # 批大小（推理场景常见取 1，即单条序列生成）
    dtype = "bf16"       # 数值精度，决定每个元素占用的字节数
    bytes_per_elem = DTYPE_BYTES[dtype]

    # 待评估的一系列上下文长度，从 256 到 131072（128K），呈指数增长，
    # 之后画图时会用对数横轴展示，这样才能在一张图里同时看清短上下文和
    # 超长上下文（如 128K）下的显存差异。
    context_lengths = [
        256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072
    ]

    # 依次计算每个上下文长度下，标准 MHA 所需的 KV 缓存大小（换算成 GB）
    mha_gb = []
    for L in context_lengths:
        total_mha = calc_kv_bytes_total_mha(
            batch_size, L, emb_dim, n_heads, n_layers, bytes_per_elem
        )
        mha_gb.append(convert_bytes_to_gb(total_mha))

    # 待对比的多种 MLA 潜在维度设置，从大到小（1024 -> 64），
    # latent_dim 越小，理论上压缩率越高（KV 缓存越小）。
    latent_dims = [1024, 512, 256, 64]
    plt.figure()
    # 先画出 MHA 这条基准曲线
    plt.plot(context_lengths, mha_gb, marker="o", label="MHA (KV total)")

    # 取最大的上下文长度作为参考点，用来计算「压缩倍数」（MHA 显存 / MLA 显存），
    # 通常在最长上下文处，压缩效果最直观、最具代表性。
    L_ref = context_lengths[-1]
    total_mha_ref = calc_kv_bytes_total_mha(batch_size, L_ref, emb_dim, n_heads, n_layers, bytes_per_elem)

    # 针对每一种 latent_dim，画出对应的 MLA 曲线，并在图例中标注压缩倍数
    for latent_dim in latent_dims:
        mla_gb = []
        for L in context_lengths:
            total_mla = calc_kv_bytes_total_mla(
                batch_size, L, n_layers, latent_dim, bytes_per_elem
            )
            mla_gb.append(convert_bytes_to_gb(total_mla))

        # 计算在最大上下文长度处，MHA 相对于当前 latent_dim 的 MLA 方案的压缩倍数
        # （即 MHA 需要多少倍于 MLA 的显存）；若 MLA 理论上为 0（极端情况）则记为无穷大，避免除零错误。
        total_mla_ref = calc_kv_bytes_total_mla(batch_size, L_ref, n_layers, latent_dim, bytes_per_elem)
        comp = total_mha_ref / total_mla_ref if total_mla_ref != 0 else float("inf")

        plt.plot(context_lengths, mla_gb, marker="o",
                 label=f"MLA (latent_dim={latent_dim}, {comp:,.1f}× compression)")

    # 横轴使用对数刻度，因为 context_lengths 是指数级增长的（256 到 131072），
    # 对数刻度能让每一段区间在视觉上均匀分布，方便观察不同量级下的趋势。
    plt.xscale("log")
    plt.xlabel("context_length (log scale)")
    plt.ylabel("Total KV cache (GB)")
    plt.title(
        "KV-cache vs Context Length — MHA vs MLA\n"
        f"(n_heads={n_heads}, emb_dim={emb_dim}, n_layers={n_layers}, "
        f"batch={batch_size}, dtype={dtype})",
        fontsize=8
    )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    # 将最终对比图保存为 PDF 文件，方便嵌入书中或分享
    plt.savefig("kv_bytes_vs_context_length.pdf")


if __name__ == "__main__":
    # 脚本入口：直接运行本文件时，执行绘图函数，生成并保存对比图
    plot_abs_kv_vs_context_multiple()