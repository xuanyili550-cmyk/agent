# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# KV-cache memory estimator for MHA vs GQA vs MLA

"""
模块中文说明
============
本文件是一个**命令行小工具**，用于估算在自回归推理（做文本生成）时，
Transformer 模型的 **KV 缓存（Key-Value Cache）** 所需的显存占用，
并对比三种不同的注意力机制在显存开销上的差异：

- **MHA（Multi-Head Attention，多头注意力）**：每个查询头都拥有独立的 K、V 头，
  缓存开销最大。
- **GQA（Grouped-Query Attention，分组查询注意力）**：多个查询头共享同一组 K、V 头，
  通过减少 K/V 头的数量来降低缓存开销（本书 ch04 中有专门实现）。
- **MLA（Multi-Head Latent Attention，多头潜在注意力，DeepSeek-V2/V3 中提出）**：
  不再为每个 token 存储完整的 K、V 张量，而是将其压缩成一个低维的“潜在向量”
  （latent vector），推理时再从该潜在向量重建出 K、V，从而大幅降低缓存显存占用。

在《从零构建大语言模型》（Build a Large Language Model From Scratch）一书中，
第 4 章（ch04）讲解了如何搭建 GPT 类模型的核心结构（注意力机制、Transformer Block
等）。本文件属于该章节的扩展/配套脚本（05_mla 目录），用来帮助读者**直观地量化**
不同注意力变体在“推理阶段显存占用”上的差距，从而理解为什么工业界大模型
（如 DeepSeek 系列）会采用 GQA / MLA 这类技术来降低部署成本。

本文件不涉及模型的前向计算或训练，只是根据模型的超参数（层数、头数、
嵌入维度、上下文长度、数据类型等），用纯数学公式估算 KV 缓存的字节数。
"""

import argparse
import math

# DTYPE_BYTES：不同数值精度（数据类型）对应的“每个元素占用的字节数”。
# 这个映射表用来把“元素个数”换算成“实际显存字节数”——
# 例如同样存 1 亿个数，用 fp32 要 4 亿字节，用 fp16/bf16 只要 2 亿字节，
# 用 fp8/int8 只要 1 亿字节。这也是为什么大模型推理时常用低精度（如 fp16/bf16/fp8）
# 来存 KV 缓存：可以直接把显存占用打对折甚至更多。
DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
}


def convert_bytes(n):
    """
    将字节数（int）转换成便于阅读的 GB 字符串。

    参数:
        n (int/float): 字节数（Bytes）。

    返回:
        str: 形如 "1,234.56 GB" 的格式化字符串。

    说明:
        这里使用 1000**3（十进制 GB，1 GB = 1,000,000,000 字节），
        而不是 1024**3（二进制 GiB）。两者在数值上有约 7% 的差异，
        阅读估算结果时需留意这一点，不要和操作系统显示的 GiB 混淆。
    """
    gb = n / (1000 ** 3)
    return f"{gb:,.2f} GB"


def calc_kv_bytes_total(batch, context_length, emb_dim, n_heads,
                             n_kv_heads, n_layers, bytes_per_elem):
    """
    计算“传统 KV 缓存”（适用于 MHA 和 GQA 两种情形）在所有 Transformer 层上
    总共需要占用的显存字节数。

    核心思路：
        - 在自回归推理时，每生成一个新 token，模型都要把该 token 在每一层、
          每一个 K/V 头上的 Key 向量和 Value 向量“缓存”下来，避免重复计算。
        - 缓存的总大小 = batch（同时处理几条序列）
                        × context_length（序列长度，即要缓存多少个 token 的 K/V）
                        × head_dim（每个注意力头的维度）
                        × n_kv_heads（K/V 头的数量——MHA 中等于 n_heads，
                          GQA 中小于 n_heads，因为若干个 Query 头共享一组 K/V 头）
                        × 2（Key 和 Value 各存一份，所以乘 2）
                        × bytes_per_elem（每个数值元素占用的字节数，取决于精度）
        - 再乘以 n_layers，因为每一层 Transformer Block 都有自己独立的 KV 缓存。

    参数:
        batch (int): 批大小，即同时处理的序列条数。
        context_length (int): 上下文长度（序列的 token 数），决定要缓存多少步的 K/V。
        emb_dim (int): 模型的嵌入维度（embedding dimension），用于换算出每个头的维度。
        n_heads (int): Query（查询）头的数量。
        n_kv_heads (int): Key/Value 头的数量——MHA 时等于 n_heads，
            GQA 时为 n_heads 除以分组数（多个 Query 头共享一组 K/V 头）。
        n_layers (int): Transformer 的层数（Block 数量）。
        bytes_per_elem (int): 每个元素占用的字节数（由 dtype 决定，见 DTYPE_BYTES）。

    返回:
        int/float: 全部层加起来的 KV 缓存总字节数。

    形状提示（便于理解维度含义）:
        单层 KV 缓存张量的逻辑形状约为
        (batch, n_kv_heads, context_length, head_dim) 且 K、V 各一份，
        故元素总数 = batch × n_kv_heads × context_length × head_dim × 2。
    """
    # Generic KV-cache: per-head dim is embed_dim / n_heads, times 2 for K and V
    # 中文：通用 KV 缓存公式——每个注意力头的维度 = 嵌入维度 / 头数（Query 头数），
    # 乘以 2 是因为要同时缓存 Key 和 Value 两份张量。
    head_dim = math.ceil(emb_dim / n_heads)  # 用 ceil 防止 emb_dim 不能被 n_heads 整除时向下取整丢精度
    # per_layer：单层 Transformer Block 的 KV 缓存字节数
    # = batch × 序列长度 × 每个头的维度 × K/V头数量 × 2(K和V) × 每元素字节数
    per_layer = batch * context_length * head_dim * n_kv_heads * 2 * bytes_per_elem
    # 所有层的缓存是相互独立、各自存储的，因此总量是逐层线性相加（等价于乘以层数）
    return per_layer * n_layers


def calc_mla_bytes_total(batch, context_length, n_layers, latent_dim, bytes_per_elem):
    """
    计算 MLA（Multi-Head Latent Attention，多头潜在注意力）方式下的 KV 缓存
    总字节数。

    核心思路：
        - MLA 的关键创新在于：不再对每个 token 分别存储“完整维度”的 Key 和 Value
          （即不再按 head_dim × n_kv_heads × 2 存储），而是先把 Key/Value 信息
          压缩（低秩投影）成一个维度小得多的“潜在向量”（latent vector，
          维度为 latent_dim），推理时需要用到 K、V 时再从这个潜在向量投影/重建出来。
        - 因为需要缓存的只是这个低维潜在向量本身，所以缓存大小与 latent_dim 成正比，
          而 latent_dim 通常远小于 head_dim × n_kv_heads × 2，从而显著降低显存占用
          （这正是 DeepSeek-V2/V3 采用 MLA 的核心原因）。

    参数:
        batch (int): 批大小。
        context_length (int): 上下文长度（需要缓存多少个 token 的潜在向量）。
        n_layers (int): Transformer 层数（每层都有独立的潜在向量缓存）。
        latent_dim (int): 每个 token 压缩后的潜在向量维度（这是 MLA 相比 MHA/GQA
            能节省显存的关键超参数，需在建模时预先设定）。
        bytes_per_elem (int): 每个元素占用的字节数（取决于数值精度）。

    返回:
        int/float: 全部层加起来的 MLA 潜在向量缓存总字节数。

    形状提示:
        单层潜在向量缓存的逻辑形状约为 (batch, context_length, latent_dim)，
        注意这里不再区分 K 和 V、也不再区分多个头——所有头共享同一个
        被压缩后的潜在表示，这也是显存能大幅下降的根本原因。
    """
    # Simple MLA (per-token compressed latent)
    # bytes ≈ batch × seqlen × n_layers × latent_dim × bytes_per_elem
    # 中文：简化版 MLA 公式——直接把“每个 token 的潜在向量维度”当作缓存的基本单位，
    # 不再乘以 head_dim、n_kv_heads 或 2（K/V 合并），因此显存开销天然更小。
    return batch * context_length * n_layers * latent_dim * bytes_per_elem


def main():
    """
    命令行入口函数：解析用户传入的模型超参数，分别计算并打印
    MHA、GQA、MLA 三种注意力机制在推理阶段的 KV 缓存显存占用，
    以及 GQA、MLA 相对于 MHA 的显存节省比例。

    该函数不接受也不返回 Python 对象参数/返回值，而是通过
    argparse 从命令行读取参数（如 --emb_dim、--n_heads 等），
    最终以 print 的形式将对比结果输出到终端，方便读者直观感受
    不同注意力变体在“推理显存开销”上的量级差异。
    """
    # 使用 argparse 构建命令行参数解析器；
    # ArgumentDefaultsHelpFormatter 会在 --help 输出中自动展示每个参数的默认值，方便用户查看
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Estimate KV-cache memory for MHA vs GQA vs MLA")
    p.add_argument("--context_length", default=1024, type=int)  # 上下文长度：要缓存多少个 token 的 K/V
    p.add_argument("--emb_dim", required=True, type=int)  # 模型嵌入维度（embedding dimension）
    p.add_argument("--n_heads", required=True, type=int)  # 注意力 Query 头的数量
    p.add_argument("--n_layers", required=True, type=int)  # Transformer 层数（Block 数量）
    p.add_argument("--n_kv_groups", required=True, type=int)  # GQA 中的分组数：n_heads 个 Query 头被分成几组共享一份 K/V
    p.add_argument("--latent_dim", required=True, type=int, help="MLA per-token latent dimension")  # MLA 每个 token 压缩后潜在向量的维度
    p.add_argument("--batch_size", default=1, type=int)  # 批大小
    p.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="fp16")  # 数值精度，决定每个元素占几个字节
    args = p.parse_args()

    # 把关键超参数收拢进一个字典，方便统一打印展示（见下方 "==== Config ====" 部分）
    cfg = {
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "n_kv_groups": args.n_kv_groups,
        "latent_dim": args.latent_dim,
    }

    # GQA 要求多个 Query 头能被“整除”地分到若干组去共享 K/V 头，
    # 否则无法均匀分组，所以这里做一次合法性校验
    if cfg["n_heads"] % cfg["n_kv_groups"] != 0:
        raise ValueError("n_kv_groups must divide n_heads exactly.")

    bytes_per_elem = DTYPE_BYTES[args.dtype]  # 查表得到当前 dtype 下每个元素占用的字节数
    head_dim = math.ceil(cfg["emb_dim"] / cfg["n_heads"])  # 计算每个注意力头的维度（emb_dim 均分给 n_heads 个头）

    n_kv_heads_mha = cfg["n_heads"]  # MHA（多头注意力）：K/V 头数量与 Query 头数量相同，即每个头都有自己独立的 K、V
    n_kv_heads_gqa = cfg["n_heads"] // cfg["n_kv_groups"]  # GQA（分组查询注意力）：K/V 头数量 = Query 头数量 / 分组数，头数变少即缓存变小

    # 计算 MHA 方式下的 KV 缓存总字节数（作为对比基准）
    total_mha = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_mha,
        cfg["n_layers"],
        bytes_per_elem,
    )

    # 计算 GQA 方式下的 KV 缓存总字节数——
    # 除了 n_kv_heads 换成了更小的 n_kv_heads_gqa，其余参数与 MHA 完全一致，
    # 这样才能公平地对比出“仅仅因为共享 K/V 头”带来的显存节省
    total_gqa = calc_kv_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["emb_dim"],
        cfg["n_heads"],
        n_kv_heads_gqa,
        cfg["n_layers"],
        bytes_per_elem,
    )

    # 计算 MLA 方式下的 KV 缓存总字节数——
    # 注意 MLA 的计算函数不需要 emb_dim / n_heads 这些参数，
    # 因为它不再按“每头维度 × 头数”存储，而是统一按 latent_dim 压缩存储
    total_mla = calc_mla_bytes_total(
        args.batch_size,
        cfg["context_length"],
        cfg["n_layers"],
        cfg["latent_dim"],
        bytes_per_elem,
    )

    # ratio：MHA 显存是 GQA 的多少倍（数值越大说明 GQA 节省效果越显著）
    ratio = total_mha / total_gqa if total_gqa != 0 else float("inf")
    # savings：GQA 相比 MHA 节省的显存百分比，1 - (GQA/MHA)
    savings = 1 - (total_gqa / total_mha) if total_mha != 0 else 0.0

    # 同理，计算 MLA 相对于 MHA 的倍数关系与节省比例
    ratio_mha_mla = total_mha / total_mla if total_mla != 0 else float("inf")
    savings_mla = 1 - (total_mla / total_mha) if total_mha != 0 else 0.0

    # 以下均为结果展示部分：先打印本次运行使用的配置，再打印三种方式的显存对比
    print("==== Config ====")
    for k, v in cfg.items():
        print(f"{k:17}: {v}")
    print(f"batch_size       : {args.batch_size}")
    print(f"dtype            : {args.dtype} ({bytes_per_elem} Bytes/elem)")
    print(f"head_dim         : {head_dim}")
    print(f"GQA n_kv_heads   : {n_kv_heads_gqa}")
    print()

    print("==== KV-cache totals across all layers ====")
    print(f"MHA total KV cache  : {convert_bytes(total_mha)}")
    print(f"GQA total KV cache  : {convert_bytes(total_gqa)}")
    print(f"MLA total KV cache  : {convert_bytes(total_mla)}")
    print(f"Ratio (MHA / GQA)   : {ratio:,.2f}x")
    print(f"Savings (GQA vs MHA): {savings*100:,.2f}%")
    print(f"Ratio (MHA / MLA)   : {ratio_mha_mla:,.2f}x")
    print(f"Savings (MLA vs MHA): {savings_mla*100:,.2f}%")


# 脚本入口：只有直接运行本文件（而不是被 import）时才会执行 main()
if __name__ == "__main__":
    main()