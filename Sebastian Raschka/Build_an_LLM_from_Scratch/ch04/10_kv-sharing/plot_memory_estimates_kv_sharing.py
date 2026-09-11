# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""Plot KV-cache memory for MHA, GQA, and cross-layer KV sharing.

中文说明：
本脚本是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 4 章 "10_kv-sharing" 示例的配套可视化工具。它本身不训练也不运行模型，而是
根据几个典型架构（标准多头注意力 MHA、分组查询注意力 GQA，以及在 GQA 基础上
进一步做“跨层 KV 共享”）的超参数，用**纯数学公式**估算推理时 KV 缓存（KV cache）
所占的显存大小，并绘制成一张对比曲线图，帮助读者直观理解：

1. MHA -> GQA：把 KV 头数从"和 Query 头数一样多"减少到"远小于 Query 头数"，
   可以大幅降低 KV 缓存显存占用；
2. GQA -> GQA + KV 共享：在 GQA 的基础上，让多层 Transformer 共享同一份 KV
   （即只有部分层真正"产生"新的 K/V，其余层直接复用），可以把显存占用进一步压低。

脚本使用 Google Gemma 3/4 系列模型的公开超参数作为预设（gemma4_e2b / gemma4_e4b），
计算不同上下文长度（context length，从 256 到 128k tokens）下，KV 缓存所需的
显存大小（单位 GB），并绘制出三条曲线（MHA / GQA / GQA + KV sharing）加以对比。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


# 不同浮点数据类型（dtype）每个元素占用的字节数。
# bf16（bfloat16）和 fp16（float16）都是 2 字节，fp32（float32）是 4 字节。
# 推理时的 KV 缓存通常使用半精度（bf16/fp16）以节省显存，这也是本脚本默认
# dtype 为 "bf16" 的原因（见 main() 中 argparse 的 default="bf16"）。
DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


# 预设的模型架构超参数，取自 Gemma 3/4 系列的两个变体（E2B 和 E4B，参数量分别
# 约为 2B 和 4B 有效参数）。每个预设包含：
#   n_query_heads          : 查询（Query）注意力头的数量
#   n_kv_heads             : 键值（Key/Value）注意力头的数量
#                            （GQA 中 n_kv_heads < n_query_heads，多个 Query 头
#                             共享同一组 K/V 头，这正是 GQA 节省显存的核心机制）
#   head_dim               : 每个注意力头的维度
#   n_layers               : Transformer 总层数
#   n_kv_producing_layers  : 真正需要计算并缓存自己 K/V 的层数（跨层 KV 共享时，
#                            其余的 n_layers - n_kv_producing_layers 层会直接复用
#                            某个"生产层"算出来的 K/V，而不需要各自存一份缓存）
PRESETS = {
    "gemma4_e2b": {
        "n_query_heads": 8,
        "n_kv_heads": 1,
        "head_dim": 256,
        "n_layers": 35,
        "n_kv_producing_layers": 15,
    },
    "gemma4_e4b": {
        "n_query_heads": 8,
        "n_kv_heads": 2,
        "head_dim": 320,
        "n_layers": 42,
        "n_kv_producing_layers": 24,
    },
}


def bytes_to_gb(n):
    """将字节数转换为十进制 GB（1 GB = 1000^3 字节，而非 1024^3）。

    参数：
        n: 字节数（int 或 float）。

    返回：
        对应的 GB 数值（float）。

    注：这里用的是十进制换算（1000^3），而不是二进制的 1024^3（那样应称为 GiB）。
    绘图/宣传材料中常用十进制 GB，这与硬件厂商标注显存容量的习惯一致。
    """
    return n / (1000 ** 3)


def context_tick_formatter(value, _pos):
    """Matplotlib 坐标轴刻度格式化函数：把具体的 token 数量转换成更易读的标签。

    参数：
        value: 当前刻度对应的数值（上下文长度，单位为 token 数）。
        _pos:  Matplotlib 要求的刻度位置参数（此处未使用，故以下划线开头）。

    返回：
        对应的字符串标签，例如 1024 -> "1k"，131072 -> "128k"；
        若该刻度值不在预定义的 labels 字典中，则返回空字符串（不显示标签）。

    这是配合下面 ax.xaxis.set_major_formatter(FuncFormatter(...)) 使用的回调函数，
    使得 x 轴（对数刻度）上只显示我们关心的几个"整齐"的 token 数量。
    """
    labels = {
        256: "256",
        1024: "1k",
        4096: "4k",
        16384: "16k",
        65536: "64k",
        131072: "128k",
    }
    return labels.get(int(round(value)), "")


def calc_kv_bytes_total(
    batch_size,
    context_length,
    head_dim,
    n_kv_heads,
    n_cached_layers,
    bytes_per_elem,
):
    """计算指定配置下，KV 缓存在所有相关层上总共占用的显存字节数。

    KV 缓存的显存占用公式为：
        总字节数 = batch_size（批大小）
                 × context_length（序列长度/上下文长度，即缓存了多少个 token 的 K/V）
                 × 2（K 和 V 各存一份，所以乘以 2）
                 × head_dim（每个注意力头的维度）
                 × n_kv_heads（KV 头的数量，GQA 中此值远小于 Query 头数）
                 × n_cached_layers（需要独立存储 K/V 的层数；
                                    对 MHA/GQA 通常等于总层数 n_layers，
                                    而对"GQA + KV 共享"则只等于真正产生 K/V 的层数
                                    n_kv_producing_layers，因为其余层复用已有缓存，
                                    不需要额外显存）
                 × bytes_per_elem（每个元素占用的字节数，取决于 dtype，如 bf16=2）

    参数：
        batch_size:      批处理大小（同时处理的序列数）。
        context_length:  上下文长度（token 数）。
        head_dim:        每个注意力头的维度。
        n_kv_heads:      KV 注意力头数量。
        n_cached_layers: 需要单独缓存 K/V 的层数。
        bytes_per_elem:  每个元素的字节数（由 dtype 决定）。

    返回：
        总的 KV 缓存字节数（int/float）。
    """
    return (
        batch_size
        * context_length
        * 2
        * head_dim
        * n_kv_heads
        * n_cached_layers
        * bytes_per_elem
    )


def compute_kv_curve(
    context_lengths,
    batch_size,
    head_dim,
    n_kv_heads,
    n_cached_layers,
    bytes_per_elem,
):
    """对一组上下文长度分别计算 KV 缓存显存占用，得到一条"上下文长度 -> 显存(GB)"曲线。

    参数：
        context_lengths: 上下文长度列表（如 [256, 512, ..., 131072]）。
        batch_size:      批大小，透传给 calc_kv_bytes_total。
        head_dim:        每个注意力头的维度。
        n_kv_heads:      KV 头数量（MHA/GQA/KV共享三种配置下该值不同）。
        n_cached_layers: 需要单独缓存 K/V 的层数。
        bytes_per_elem:  每个元素字节数（由 dtype 决定）。

    返回：
        一个 list[float]，与 context_lengths 一一对应，表示每个上下文长度下
        KV 缓存所需的显存大小（单位：GB）。这就是后面用来画折线图的一条曲线的
        纵坐标数据。
    """
    curve = []
    for context_length in context_lengths:
        # 对每个上下文长度，先算出总字节数，再换算成 GB，追加到曲线数据中。
        total_bytes = calc_kv_bytes_total(
            batch_size=batch_size,
            context_length=context_length,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            n_cached_layers=n_cached_layers,
            bytes_per_elem=bytes_per_elem,
        )
        curve.append(bytes_to_gb(total_bytes))
    return curve


def add_end_label(ax, x_value, y_value, text, color, y_offset=0.0):
    """在曲线的最右端（通常是最大上下文长度处）添加一个文字标签，标注该曲线的数值。

    参数：
        ax:       Matplotlib 的 Axes 对象，图形绘制在其上。
        x_value:  标签锚定的 x 坐标（通常传入曲线最后一个点的 x 值，即最大上下文长度）。
        y_value:  标签锚定的 y 坐标（通常传入曲线最后一个点的 y 值，即对应的显存 GB 数）。
        text:     要显示的文字内容（如 "MHA 12.3 GB"）。
        color:    文字颜色，通常与对应曲线的颜色保持一致，便于视觉关联。
        y_offset: y 方向的偏移量，用于在多条曲线终点数值接近、标签会重叠时
                  做手动错位（微调），默认不偏移。

    返回：
        无返回值（None），直接在传入的 ax 上添加文字。
    """
    ax.text(
        x_value * 1.16,  # 把标签放在曲线终点右侧一点的位置（x 方向放大 1.16 倍），避免与曲线重叠
        y_value + y_offset,
        text,
        color=color,
        fontsize=8,
        va="center",
        ha="left",
        clip_on=False,  # 允许标签绘制到坐标轴范围之外（因为图右侧特意留了空白区域用于放标签）
    )


def main():
    """脚本主入口：解析命令行参数、计算三条 KV 缓存曲线并绘制/保存对比图。

    主要流程：
        1. 用 argparse 解析命令行参数（选择预设模型、批大小、dtype、输出路径）。
        2. 取出所选预设的架构超参数（cfg）和 dtype 对应的字节数。
        3. 针对一组预定义的上下文长度（256 ~ 131072 tokens），分别计算：
           - MHA：n_kv_heads = n_query_heads（每个 Query 头都有自己独立的 K/V），
                  n_cached_layers = n_layers（每一层都要缓存）；
           - GQA：n_kv_heads = cfg 中较小的 n_kv_heads（多头共享 K/V），
                  n_cached_layers 仍为 n_layers；
           - GQA + KV sharing：在 GQA 基础上，n_cached_layers 进一步减小为
                  n_kv_producing_layers（只有部分层需要真正产生并缓存 K/V）。
        4. 用 Matplotlib 绘制三条曲线（对数横轴，避免小上下文长度的差异被压缩），
           添加标题、副标题（当前预设的超参数说明）、文字批注（128k 上下文时三者
           的显存对比数据），并在曲线右端标注具体数值。
        5. 将图保存为 PDF 文件（默认文件名包含预设名称），并打印保存路径。

    参数：
        无（参数通过 argparse 从命令行读取）。

    返回：
        无返回值（None）。函数的"输出"是保存到磁盘的图像文件以及打印到终端的提示信息。
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot KV-cache memory for MHA, GQA, and cross-layer KV sharing",
    )
    # --preset：选择要对比的模型架构预设（决定头数、层数等超参数）
    p.add_argument("--preset", choices=PRESETS.keys(), default="gemma4_e4b")
    # --batch_size：推理时的批大小，直接线性影响 KV 缓存显存占用
    p.add_argument("--batch_size", type=int, default=1)
    # --dtype：KV 缓存存储用的数据类型，决定每个元素占用的字节数
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bf16")
    # --output：输出图片文件路径，若不指定则根据预设名自动生成默认文件名
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    cfg = PRESETS[args.preset]
    bytes_per_elem = DTYPE_BYTES[args.dtype]
    # 横轴取值：一组从 256 到 131072（即 128k）呈 2 的幂次增长的上下文长度，
    # 用于覆盖从短上下文到长上下文（长文档/长对话）的典型场景。
    context_lengths = [
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    ]

    # 分别计算三种架构配置下的 KV 缓存显存曲线，存入字典，
    # 字典的 key 即图例（legend）中显示的名称。
    curves = {
        "MHA": compute_kv_curve(
            context_lengths,
            batch_size=args.batch_size,
            head_dim=cfg["head_dim"],
            n_kv_heads=cfg["n_query_heads"],  # MHA：KV 头数 = Query 头数（无共享）
            n_cached_layers=cfg["n_layers"],  # 每一层都需要独立缓存 K/V
            bytes_per_elem=bytes_per_elem,
        ),
        "GQA": compute_kv_curve(
            context_lengths,
            batch_size=args.batch_size,
            head_dim=cfg["head_dim"],
            n_kv_heads=cfg["n_kv_heads"],  # GQA：KV 头数远小于 Query 头数，多个 Query 头共享一组 K/V
            n_cached_layers=cfg["n_layers"],  # 仍然是每层都缓存，只是每层的 K/V 更小了
            bytes_per_elem=bytes_per_elem,
        ),
        "GQA + KV sharing": compute_kv_curve(
            context_lengths,
            batch_size=args.batch_size,
            head_dim=cfg["head_dim"],
            n_kv_heads=cfg["n_kv_heads"],  # 沿用 GQA 的 KV 头数设置
            n_cached_layers=cfg["n_kv_producing_layers"],  # 关键区别：只有部分层真正产生/缓存 K/V，
                                                            # 其余层跨层复用，进一步减少缓存层数
            bytes_per_elem=bytes_per_elem,
        ),
    }

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    # 为三条曲线分别指定颜色、线宽和绘制层级（zorder 越大越"盖在上层"），
    # 使得重点强调的 "GQA + KV sharing" 曲线（蓝色、zorder=4）显示在最上方。
    styles = {
        "MHA": {"color": "#2f2f2f", "linewidth": 2.0, "zorder": 3},
        "GQA": {"color": "#6f7f8f", "linewidth": 1.8, "zorder": 3},
        "GQA + KV sharing": {"color": "#2f7ae5", "linewidth": 2.2, "zorder": 4},
    }

    # 依次把三条曲线画到同一张图上；solid_capstyle="round" 让线条端点圆润一些（美观用途）。
    for label, values in curves.items():
        ax.plot(context_lengths, values, solid_capstyle="round", **styles[label])

    # y 轴上限取 MHA 曲线最大值的 1.08 倍，留出一点顶部空白，避免曲线贴着图框顶部。
    max_y = max(curves["MHA"]) * 1.08
    # x 轴使用以 2 为底的对数坐标：因为 context_lengths 是等比数列（每次翻倍），
    # 对数坐标能让各个数据点在横轴上均匀分布，便于观察趋势。
    ax.set_xscale("log", base=2)
    # x 轴右侧多留出一些空间（乘以 2.1），用于放置 add_end_label 添加的曲线终点标签。
    ax.set_xlim(context_lengths[0], context_lengths[-1] * 2.1)
    ax.set_ylim(0, max_y)
    # 只在这几个"整齐"的上下文长度处显示主刻度，避免对数轴上刻度过密显得杂乱。
    ax.xaxis.set_major_locator(FixedLocator([256, 1024, 4096, 16384, 65536, 131072]))
    # 用前面定义的 context_tick_formatter 把刻度值格式化成 "1k"、"128k" 这样的简洁标签。
    ax.xaxis.set_major_formatter(FuncFormatter(context_tick_formatter))

    ax.set_xlabel("Context length (tokens, log scale)")
    ax.set_ylabel("KV cache across all layers (GB)")
    ax.set_title("GQA and KV sharing compound the KV-cache savings", loc="left", pad=20)
    # 在标题下方（图的左上角、稍高于坐标轴区域）添加一行小字，说明当前使用的预设及其超参数，
    # 方便读者知道这张图具体对应哪个模型配置。
    ax.text(
        0.0,
        1.02,
        (
            f"{args.preset}: {cfg['n_query_heads']} query heads, "
            f"{cfg['n_kv_heads']} KV heads, {cfg['n_layers']} layers, "
            f"{cfg['n_kv_producing_layers']} K/V-producing layers, "
            f"batch {args.batch_size}, {args.dtype}"
        ),
        transform=ax.transAxes,  # 使用坐标轴的相对坐标系（0~1），而非数据坐标系，便于定位到图的角落
        fontsize=8,
        color="#666666",
        ha="left",
        va="bottom",
    )

    # 取出三条曲线在最大上下文长度（128k tokens）处的数值，用于后面的文字批注和终点标签。
    mha_128k = curves["MHA"][-1]
    gqa_128k = curves["GQA"][-1]
    sharing_128k = curves["GQA + KV sharing"][-1]

    # 在图的左上角内部添加一段文字批注，用具体数字总结 GQA 和 KV 共享分别带来的显存节省效果，
    # 帮助读者一眼看出"从 MHA 到 GQA 再到 KV 共享，显存降低了多少"。
    ax.text(
        0.02,
        0.96,
        (
            f"At 128k tokens, MHA would need {mha_128k:.1f} GB;\n"
            f"GQA cuts this to {gqa_128k:.1f} GB, and KV sharing to {sharing_128k:.1f} GB."
        ),
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none"},
    )

    # 去掉图形上方和右侧的边框线，只保留左侧和下方，视觉上更简洁（常见的"极简风"绘图美化）。
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#a8a8a8")
    ax.spines["bottom"].set_color("#a8a8a8")
    ax.tick_params(axis="both", which="major", length=3, color="#888888", labelsize=8)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.grid(False)  # 不显示网格线，保持图面干净

    # 在每条曲线的最右端（128k tokens 处）标注具体的显存数值（GB），
    # 三个标签用 y_offset 做了微调（0、0.25、-0.25），避免数值接近时文字互相重叠。
    add_end_label(ax, context_lengths[-1], mha_128k, f"MHA {mha_128k:.1f} GB", "#2f2f2f")
    add_end_label(ax, context_lengths[-1], gqa_128k, f"GQA {gqa_128k:.1f} GB", "#6f7f8f", 0.25)
    add_end_label(
        ax,
        context_lengths[-1],
        sharing_128k,
        f"GQA + KV sharing {sharing_128k:.1f} GB",
        "#2f7ae5",
        -0.25,
    )

    # 确定输出文件路径：若用户未通过 --output 指定，则根据当前预设名自动生成一个默认文件名。
    output = args.output
    if output is None:
        output = Path(f"kv_memory_mha_gqa_kvsharing_{args.preset}.pdf")
    fig.tight_layout()  # 自动调整子图间距，避免标签、标题被裁剪
    fig.savefig(output)
    plt.close(fig)  # 关闭图形对象，释放内存（尤其在批量生成多张图时很重要）
    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
