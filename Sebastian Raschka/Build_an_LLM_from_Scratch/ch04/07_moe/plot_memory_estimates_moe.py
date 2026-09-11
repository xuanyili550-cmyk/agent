# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
【模块中文说明】
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 4 章 "07_moe"（Mixture-of-Experts，混合专家模型）配套代码的一部分。

它的作用是：**用可视化的方式，直观展示"稠密（Dense）FFN"与"MoE FFN"在参数量上的核心区别**——
    - 稠密 FFN：每个 token 都会用到全部参数，"激活参数量 = 总参数量"；
    - MoE FFN：模型总参数量可以很大（很多个专家），但每个 token 只会激活其中
      top_k 个专家 + 路由器（router）的参数，所以"激活参数量"远小于"总参数量"。
      这正是 MoE 架构能够"参数量很大但推理/训练计算量却不成比例增长"的关键原理。

本文件依赖同目录下的 memory_estimator_moe.py 中的三个函数来做参数量的数学计算，
自身只负责：
    1）在不同专家数量（num_experts）下，重复调用参数量计算函数，得到
       "MoE 激活参数量" 和 "MoE 总参数量" 两条曲线；
    2）用 matplotlib 把这两条曲线以及稠密 FFN 的参数量（一条水平参考线）画在一张图上；
    3）提供一个命令行入口（argparse），方便直接在终端跑不同配置查看效果。

阅读建议：先看 memory_estimator_moe.py 里 estimate_params_and_hidden 的实现，
再回来看本文件如何调用它来批量绘图，会更容易理解 MoE "参数解耦"（总参数 vs 激活参数）的设计思想。
"""


import argparse
import matplotlib.pyplot as plt
from memory_estimator_moe import (
    estimate_params_and_hidden,
    calc_ffn_params,
    calc_router_params,
)


def calc_moe_active_and_total(
    emb_dim,
    hidden_dim,
    ffn_type,
    num_experts,
    top_k,
    match_dense=True,
):
    """
    计算给定配置下，MoE 层"每个 token 的激活参数量"与"MoE 层的总参数量"。

    这是本文件的核心计算函数，对 memory_estimator_moe.estimate_params_and_hidden
    的结果做了一次二次加工：把"每个专家的参数量"乘以 top_k（每个 token 实际
    路由到的专家数）再加上路由器参数，得到"激活参数量"这个 MoE 特有的重要指标。

    参数：
        emb_dim (int): 模型的嵌入维度（embedding dimension），例如 768、4096 等。
            对应 Transformer 中隐藏状态张量最后一维的大小，如 (batch, seq_len, emb_dim)。
        hidden_dim (int): 稠密（Dense）FFN 的中间层（隐藏层）维度。用作参照基准，
            也在 match_dense=False 时直接作为每个专家 FFN 的隐藏层维度。
        ffn_type (str): FFN 的类型，"gelu"（2 个参数矩阵）或 "swiglu"（3 个参数矩阵）。
            不同激活函数对应的前馈网络结构不同，参数矩阵数量也不同。
        num_experts (int): 专家（expert）的数量，即 MoE 层里并行的 FFN 子网络个数。
        top_k (int): 稀疏路由时，每个 token 实际会被送去计算的专家个数
            （Top-K 路由，MoE 的核心稀疏性来源）。
        match_dense (bool): 是否让 MoE 的总参数量尽量匹配（约等于）稠密 FFN 的总参数量。
            若为 True，会自动反推出每个专家应有的隐藏层维度（moe_hidden_dim），
            从而实现"总参数量对齐、但通过路由稀疏激活"的公平对比。

    返回：
        (active, total) 二元组：
            active (int/float): 每个 token 平均激活的参数量 = 路由器参数 + top_k × 单专家参数。
                这近似代表了 MoE 层在前向计算时"实际参与计算"的参数规模，
                直接影响推理/训练的 FLOPs（浮点运算量）。
            total (float): MoE 层的参数总量（所有专家参数 + 路由器参数），
                代表模型权重实际占用的显存/存储空间，与激活参数量可以相差很大。
    """
    if match_dense:
        # 先算出：如果不做 MoE，用同样 emb_dim/hidden_dim 的稠密 FFN 会有多少参数
        dense_params = calc_ffn_params(emb_dim, hidden_dim, ffn_type)
        # 以及路由器（一个从 emb_dim 映射到 num_experts 个 logit 的线性层）需要多少参数
        router = calc_router_params(emb_dim, num_experts)
        # 如果稠密参数量甚至还不够覆盖路由器开销，"匹配稠密总参数量"这个约束就没有意义了，
        # 这种极端情况下退化为直接使用传入的 hidden_dim
        if dense_params <= router:
            match_dense = False

    # 调用 memory_estimator_moe 里的核心函数，得到路由器参数量、
    # 每个专家的参数量、以及 MoE 总参数量等统计信息（内部会按需反推 moe_hidden_dim）
    stats = estimate_params_and_hidden(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        ffn_type=ffn_type,
        num_experts=num_experts,
        match_dense=match_dense,
    )

    # 激活参数量 = 路由器参数（每个 token 都要过一次路由器） + top_k 个专家的参数量
    # 这里之所以不是 num_experts × 单专家参数，正是 MoE "稀疏激活" 的体现：
    # 无论专家总数 num_experts 多大，每个 token 只真正用到其中 top_k 个专家的权重。
    active = stats["router"] + top_k * stats["per_expert_params"]
    return active, stats["moe_total"]


def plot_active_params_vs_experts(
    emb_dim,
    hidden_dim,
    ffn_type="swiglu",
    top_k=2,
    max_experts=512,
    y_log=True,
    save_path=None,
    match_dense=True,
):
    """
    绘制"专家数量（num_experts）"与"参数量"关系的折线图，用于直观对比：
        1）MoE 每个 token 的激活参数量（随专家数增多几乎不变或缓慢变化）；
        2）MoE 总参数量（随专家数增多而线性增长）；
        3）稠密 FFN 的参数量（激活量恒等于总量，画成一条水平参考虚线）。

    该图的教学意义在于：MoE 可以让"总参数量"远超稠密模型，
    但由于稀疏路由（每个 token 只激活 top_k 个专家），"激活参数量"（决定实际计算成本）
    却可以保持在与稠密模型相近的水平，这就是 MoE 用"更多参数换取更强表达能力，
    同时不显著增加推理计算量"的核心权衡（trade-off）。

    参数：
        emb_dim (int): 嵌入维度，同 calc_moe_active_and_total。
        hidden_dim (int): 稠密 FFN 的隐藏层维度，用作对比基准。
        ffn_type (str): "gelu" 或 "swiglu"，FFN 的类型，默认 "swiglu"（LLaMA 等现代模型常用）。
        top_k (int): 每个 token 路由到的专家数，默认 2（业界常见选择，如 Mixtral）。
        max_experts (int): x 轴（专家数量）绘制的上限，超过此值的候选专家数会被过滤掉。
        y_log (bool): 是否将 y 轴（参数量）设为对数坐标。由于总参数量随专家数
            线性增长、跨越多个数量级，用对数坐标能同时看清激活量和总量两条曲线。
        save_path (str or None): 若提供路径，则把图保存为 PNG 文件；否则直接用
            plt.show() 弹窗展示（适合本地交互式运行）。
        match_dense (bool): 是否让每个专家数配置下的 MoE 总参数量都尽量对齐稠密 FFN 的总参数量，
            便于"控制总参数量不变，只看路由稀疏化对激活参数量的影响"这种对比实验。

    返回：
        无返回值（None）。函数的副作用是弹出一张 matplotlib 图或把图保存到磁盘。
    """
    # 预设一批常见的专家数量取值（多为 2 的幂次，符合实践中常见的 MoE 配置习惯）
    experts = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512]
    # 根据 max_experts 过滤掉超出绘图范围的取值
    experts = [e for e in experts if e <= max_experts]

    # 稠密 FFN 的参数量：作为整张图里唯一"激活参数量 = 总参数量"的参照基准
    dense_active = calc_ffn_params(emb_dim, hidden_dim, ffn_type)
    moe_active = []  # 存放不同专家数下，MoE 每个 token 的激活参数量
    moe_total = []   # 存放不同专家数下，MoE 的总参数量
    for e in experts:
        # 对每一个候选专家数量 e，重新计算一次"激活参数量"和"总参数量"
        active, total = calc_moe_active_and_total(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            ffn_type=ffn_type,
            num_experts=e,
            top_k=top_k,
            match_dense=match_dense,
        )
        moe_active.append(active)
        moe_total.append(total)

    plt.figure(figsize=(7, 5))
    # 曲线一：MoE 激活参数量 —— 理想情况下应随专家数增多而基本持平或缓慢下降
    # （因为 match_dense=True 时，专家数越多，为了保持总量不变，单专家隐藏层维度会变小）
    plt.plot(experts, moe_active, marker="o", label="MoE active per token")
    # 曲线二：MoE 总参数量 —— 若 match_dense=True 则应接近一条水平线（对齐稠密模型）；
    # 若 match_dense=False 则会随专家数增多而线性增长
    plt.plot(experts, moe_total, marker="s", linestyle="--", label="MoE total parameters")
    # 参考线：稠密 FFN 的参数量，激活参数量恒等于总参数量，用点线画一条水平基准线
    plt.axhline(dense_active, linestyle=":", color="gray",
                label="FFN dense (active = total)")

    plt.xlabel(f"Number of experts (top_k = {top_k})")
    plt.ylabel("Parameters")
    if y_log:
        # 因为总参数量可能是激活参数量的几十甚至上百倍，用对数坐标才能在同一张图里看清两者
        plt.yscale("log")
    plt.title(
        f"Active vs Total Parameters per Token\n"
        f"(emb_dim={emb_dim}, hidden_dim={hidden_dim}, ffn={ffn_type}, top_k={top_k})"
    )
    plt.legend()
    plt.tight_layout()
    if save_path:
        # 若指定了保存路径，则输出高分辨率（dpi=200）PNG 图片，便于写入报告/文档
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    else:
        # 否则走交互式展示（本地运行、有图形界面时使用）
        plt.show()


def main():
    """
    命令行入口函数：解析用户传入的参数，并调用 plot_active_params_vs_experts 完成绘图。

    典型用法示例：
        python plot_memory_estimates_moe.py --emb_dim 4096 --hidden_dim 14336

    该函数本身不返回值，主要职责是把命令行字符串参数转换为
    plot_active_params_vs_experts 所需的关键字参数并调用它。
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Plot Dense vs MoE active parameters.")
    p.add_argument("--emb_dim", type=int, required=True, help="Embedding dimension")
    p.add_argument("--hidden_dim", type=int, required=True, help="Dense FFN hidden size")
    p.add_argument("--ffn_type", choices=["gelu", "swiglu"], default="swiglu")
    p.add_argument("--top_k", type=int, default=2, help="Active experts per token")
    p.add_argument("--max_experts", type=int, default=512, help="Max experts on x-axis")
    p.add_argument("--no_log", action="store_true", help="Disable log-scale y-axis")
    p.add_argument("--save", type=str, default=None, help="Optional path to save PNG")
    p.add_argument(
        "--no_match_dense",
        action="store_true",
        help=("Disable matching MoE parameters to dense FFN total; "
              "uses provided hidden_dim instead."),
    )
    args = p.parse_args()

    # 将解析出的命令行参数，一一对应地传给绘图函数；
    # 注意 --no_log / --no_match_dense 是"反义"开关，需要取反（not）才能得到
    # plot_active_params_vs_experts 期望的 y_log / match_dense 语义
    plot_active_params_vs_experts(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        ffn_type=args.ffn_type,
        top_k=args.top_k,
        max_experts=args.max_experts,
        y_log=not args.no_log,
        save_path=args.save,
        match_dense=not args.no_match_dense,
    )


if __name__ == "__main__":
    # 仅当直接运行本脚本（而非被其他模块 import）时才执行 main()，
    # 这是 Python 脚本常见的入口保护写法
    main()
