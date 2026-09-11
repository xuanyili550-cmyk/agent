# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# ==================== 中文说明（模块级 docstring） ====================
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
# 第 4 章「07_moe」（Mixture-of-Experts，专家混合模型）配套的辅助小工具脚本。
#
# 用途：
#   这是一个纯 CPU 上运行的「参数量 / 显存占用估算器」命令行脚本，不涉及任何真实的
#   模型前向传播或训练。它用来帮助读者直观理解：
#     1）一个标准的稠密（Dense）FFN（前馈网络）层有多少参数；
#     2）如果把这一层替换成 MoE（多个「专家」FFN + 一个路由器 Router）结构，
#        总参数量（存储/显存占用）会变成多少；
#     3）在推理时，MoE 每个 token 实际只会激活 top_k 个专家，因此「每 token 实际
#        参与计算的参数量」（active params）远小于「总参数量」（total params）——
#        这正是 MoE 用「稀疏激活」换取「大容量、低推理成本」的核心思想。
#
# 在书中的角色：
#   配合正文中关于 MoE 架构（如 Mixtral、DeepSeek 等模型采用的路由专家机制）的讲解，
#   让读者通过修改命令行参数（emb_dim、hidden_dim、num_experts、top_k、dtype 等），
#   直接算出不同配置下模型的参数量和显存占用（GB），从而建立起对 MoE
#   「参数量增大但计算量可控」这一权衡关系的量化理解。
# =======================================================================

import argparse

# 不同数值精度（dtype）下，每个参数元素占用的字节数。
# 这决定了「参数个数 -> 显存字节数」的换算比例。
# 例如 bf16/fp16 每个参数占 2 字节，fp8/int8 每个参数只占 1 字节（激进量化）。
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def convert_bytes(n):
    """将字节数（int）转换为人类可读的 GB 字符串。

    参数:
        n: 字节数（例如 参数个数 * 每个参数的字节数）。

    返回:
        形如 "1.23 GB" 的字符串。这里使用 1000**3（而非 1024**3）作为 1GB 的换算基数，
        即十进制 GB（GB, decimal），不是二进制 GiB，这与硬盘厂商/云厂商常见计量方式一致。
    """
    gb = n / (1000 ** 3)
    return f"{gb:,.2f} GB"


def get_num_param_matrices(ffn_type):
    """根据 FFN（前馈网络）的类型，返回该 FFN 内部包含的权重矩阵个数。

    参数:
        ffn_type: "gelu" 或 "swiglu"。
            - "gelu"：经典的两层 FFN（Linear -> GELU 激活 -> Linear），
              只有 2 个权重矩阵（一个升维矩阵 + 一个降维矩阵）。
            - "swiglu"：SwiGLU 结构（如 LLaMA 系列使用），需要 3 个权重矩阵
              （门控矩阵 gate_proj + 上投影矩阵 up_proj + 下投影矩阵 down_proj），
              因为 SwiGLU 是「门控 * 上投影」后再降维，比普通 GELU FFN 多了一个矩阵。

    返回:
        int，矩阵个数（2 或 3）。

    异常:
        ffn_type 既不是 "gelu" 也不是 "swiglu" 时抛出 ValueError。
    """
    if ffn_type == "gelu":
        return 2
    elif ffn_type == "swiglu":
        return 3
    else:
        raise ValueError("--ffn_type must be 'gelu' or 'swiglu'")


def calc_ffn_params(emb_dim, hidden_dim, ffn_type):
    """计算单个 FFN（前馈网络）模块的参数总量。

    参数:
        emb_dim: 模型的嵌入维度（embedding dimension），即 token 向量的宽度。
        hidden_dim: FFN 内部的隐藏层维度（中间升维后的宽度），通常远大于 emb_dim。
        ffn_type: "gelu" 或 "swiglu"，决定矩阵个数（见 get_num_param_matrices）。

    返回:
        int，该 FFN 模块的参数总数。

    要点:
        每个权重矩阵的形状近似为 (emb_dim, hidden_dim) 或 (hidden_dim, emb_dim)，
        参数个数都约为 emb_dim * hidden_dim（这里忽略了偏置项，做的是近似估算）。
        因此总参数量 = 矩阵个数 * emb_dim * hidden_dim。
        例如 SwiGLU 有 3 个这样大小的矩阵，所以是 3 * emb_dim * hidden_dim。
    """
    return get_num_param_matrices(ffn_type) * emb_dim * hidden_dim


def calc_router_params(emb_dim, num_experts):
    """计算 MoE 路由器（Router / Gate）的参数量。

    参数:
        emb_dim: 模型的嵌入维度。
        num_experts: 专家（expert）的总数量。

    返回:
        int，路由器参数总数。

    要点:
        路由器本质上是一个简单的线性层，把每个 token 的向量（维度 emb_dim）
        映射到 num_experts 个打分（用于决定该 token 应该分配给哪些专家）。
        因此其权重矩阵形状约为 (emb_dim, num_experts)，参数量为 emb_dim * num_experts。
        相比每个专家动辄百万级参数的 FFN，路由器的参数量通常非常小，可忽略不计，
        但完整估算时仍应把它算进总参数量里。
    """
    return emb_dim * num_experts


def estimate_params_and_hidden(
    emb_dim, hidden_dim, ffn_type, num_experts, match_dense=False
):
    """核心估算函数：对比「稠密 FFN」与「MoE（专家混合）」两种结构的参数量。

    参数:
        emb_dim: 模型嵌入维度。
        hidden_dim: 稠密（Dense）FFN 的隐藏层维度（作为对比基准，也是 MoE 不做
            match_dense 时每个专家默认使用的隐藏层维度）。
        ffn_type: "gelu" 或 "swiglu"，决定每个 FFN/专家内部矩阵个数。
        num_experts: MoE 中专家的总数量。
        match_dense: 布尔值，是否让 MoE 的「总参数量」尽量与稠密 FFN 的参数量对齐。
            - False（默认）：MoE 中每个专家的隐藏维度直接等于 hidden_dim，
              这样每个专家单独看和稠密 FFN 一样大，但因为有多个专家，
              MoE 总参数量会远大于稠密 FFN（这是最常见的「MoE 用更多总参数换取
              相同计算量」的做法）。
            - True：反过来，先固定「MoE 总参数量（含路由器）约等于稠密 FFN 参数量」
              这个目标，反推出每个专家应该用多小的隐藏维度 moe_hidden_dim。
              这在探索「同样存储预算下，MoE 相比稠密模型能否带来更强表达能力」时很有用。

    返回:
        dict，包含以下字段：
            - "dense_params": 稠密 FFN 的参数总数（int）。
            - "router": 路由器参数总数（int）。
            - "moe_hidden_dim": 实际使用的每专家隐藏维度（int），
              match_dense=True 时是反推出来的，否则等于传入的 hidden_dim。
            - "per_expert_params": 单个专家 FFN 的参数总数（int）。
            - "moe_total": MoE 结构的参数总数 = 所有专家参数之和 + 路由器参数（int）。
    """
    # 稠密（Dense）FFN 作为参照基准的参数量：矩阵个数 * emb_dim * hidden_dim。
    P_dense = calc_ffn_params(emb_dim, hidden_dim, ffn_type)
    # 路由器参数量：emb_dim * num_experts，用于把 token 分发给各个专家。
    R = calc_router_params(emb_dim, num_experts)

    if match_dense:
        # 目标：让 MoE 的总参数量（num_experts 个专家 + 路由器）约等于稠密 FFN 的参数量 P_dense。
        # 即：num_experts * num_param_matrices * emb_dim * moe_hidden_dim + R ≈ P_dense
        # 整理可得：moe_hidden_dim ≈ (P_dense - R) / (num_experts * num_param_matrices * emb_dim)
        num_param_matrices = get_num_param_matrices(ffn_type)
        num = P_dense - R  # 分子：稠密参数量减去路由器占用的那部分「预算」
        den = num_experts * num_param_matrices * emb_dim  # 分母：每个专家隐藏维度每增加 1 所消耗的参数量之和
        if num <= 0:
            # 如果稠密层参数量还不够覆盖路由器开销，说明模型太小、专家数太多，无法匹配，直接报错。
            raise ValueError("Dense layer too small for requested num_experts.")
        # 四舍五入取整，得到每个专家应使用的隐藏维度，以使 MoE 总参数量尽量贴近稠密 FFN。
        moe_hidden_dim = int(round(num / float(den)))
    else:
        # 不做参数量对齐时，每个专家直接沿用与稠密 FFN 相同的隐藏维度。
        moe_hidden_dim = hidden_dim

    # 单个专家的参数量（专家本质上就是一个独立的、更小或同等大小的 FFN）。
    per_expert_params = calc_ffn_params(emb_dim, moe_hidden_dim, ffn_type)
    # MoE 总参数量 = 所有专家参数之和 + 路由器参数。
    # 注意：这是「存储/显存中要保存的全部参数」，并不代表每个 token 推理时都会用到全部专家。
    moe_total = num_experts * per_expert_params + R

    return {
        "dense_params": P_dense,
        "router": R,
        "moe_hidden_dim": moe_hidden_dim,
        "per_expert_params": per_expert_params,
        "moe_total": moe_total,
    }


def main():
    """命令行入口函数：解析参数、计算并打印稠密 FFN 与 MoE 的参数量/显存对比结果。

    无参数、无返回值；直接向标准输出打印格式化后的估算报告。
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Estimate FFN vs MoE parameter memory"
    )
    # 模型嵌入维度，决定了每个权重矩阵的一个维度大小。
    p.add_argument("--emb_dim", type=int, required=True,
                   help="Model embedding dimension.")
    # 稠密 FFN 的隐藏层维度，作为对比基准。
    p.add_argument("--hidden_dim", type=int, required=True,
                   help="Dense FFN intermediate size (hidden dimension).")
    # FFN 类型：普通 GELU 两层结构，或 SwiGLU 三矩阵结构（LLaMA 系列常用）。
    p.add_argument("--ffn_type", choices=["gelu", "swiglu"], default="swiglu")
    # MoE 专家总数（存储上会保存这么多份专家 FFN）。
    p.add_argument("--num_experts", type=int, default=8)
    # 每个 token 实际路由到并激活计算的专家个数（top_k 稀疏激活，这是 MoE 节省算力的关键）。
    p.add_argument("--top_k", type=int, default=2)
    # 数值精度类型，决定每个参数占用的字节数（用于换算显存大小）。
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bf16")
    # 是否开启「参数量对齐」模式：让 MoE 总参数量与稠密 FFN 参数量尽量相等，
    # 从而反推每个专家应有多小的隐藏维度。
    p.add_argument(
        "--match_dense",
        action="store_true",
        help=("Auto-set per-expert hidden so MoE total params ~= dense FFN params "
              "(router included)."),
    )
    args = p.parse_args()

    # 根据所选 dtype 查表得到每个参数元素占用的字节数，用于后续「参数个数 -> 显存」换算。
    bytes_per_elem = DTYPE_BYTES[args.dtype]

    # 调用核心估算函数，得到稠密/路由器/每专家/MoE总参数量等结果字典。
    res = estimate_params_and_hidden(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        ffn_type=args.ffn_type,
        num_experts=args.num_experts,
        match_dense=args.match_dense,
    )

    # MoE 推理时，每个 token 实际参与计算的参数量：
    # 路由器本身总是要参与计算的（给每个 token 打分），
    # 再加上被选中的 top_k 个专家的参数量。
    # 这个数值远小于 moe_total，正是 MoE「稀疏激活」带来推理成本优势的量化体现。
    moe_active_params_per_token = (
        res["router"] + args.top_k * res["per_expert_params"]
    )

    print("==== Config ====")
    print(f"{'emb_dim':23}: {args.emb_dim}")
    print(f"{'hidden_dim':23}: {args.hidden_dim}")
    print(f"{'ffn_type':23}: {args.ffn_type}")
    print(f"{'num_experts':23}: {args.num_experts}")
    print(f"{'top_k':23}: {args.top_k}")
    print(f"{'dtype':23}: {args.dtype} ({bytes_per_elem} Bytes/elem)")
    print(f"{'match_dense':23}: {args.match_dense}")
    print()

    print("==== Model weights (parameters) ====")
    # 稠密 FFN 的参数量及其对应显存占用（作为基准，方便与下面的 MoE 结果对比）。
    print(f"{'Dense FFN params':23}: {res['dense_params']:,} "
          f"({convert_bytes(res['dense_params'] * bytes_per_elem)})")
    # 单个专家的参数量及显存占用（乘以 num_experts 即可估算全部专家占用的显存）。
    print(f"{'Per-expert params':23}: {res['per_expert_params']:,} "
          f"({convert_bytes(res['per_expert_params'] * bytes_per_elem)})")
    # 路由器参数量及显存占用（通常很小）。
    print(f"{'Router params':23}: {res['router']:,} "
          f"({convert_bytes(res['router'] * bytes_per_elem)})")
    # MoE 全部专家 + 路由器的总参数量及总显存占用（存储/显存意义上的「模型大小」）。
    print(f"{'MoE TOTAL params':23}: {res['moe_total']:,} "
          f"({convert_bytes(res['moe_total'] * bytes_per_elem)})")
    # MoE 每个 token 推理时实际激活参与计算的参数量及等效显存/带宽占用
    # （这是决定推理速度/FLOPs 的关键指标，远小于 MoE TOTAL）。
    print(f"{'MoE ACTIVE/Token':23}: {moe_active_params_per_token:,} "
          f"({convert_bytes(moe_active_params_per_token * bytes_per_elem)})")
    # 实际使用的每专家隐藏维度（若开启 match_dense，则是反推计算出来的值）。
    print(f"{'moe_hidden_dim':23}: {res['moe_hidden_dim']}")
    print()


if __name__ == "__main__":
    main()
