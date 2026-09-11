"""
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 4 章附加材料 "09_dsa" 目录下的测试文件。

它的作用是：针对 `gpt_with_kv_dsa.py` 中实现的 DeepSeek 稀疏注意力
(DeepSeek Sparse Attention, 简称 DSA) 相关代码进行单元测试，验证：
    1. 注意力模块 `MultiHeadAttentionWithDSA` 的输出形状是否正确；
    2. 因果掩码（causal mask）是否生效，即位置 p 的输出不应受到位置 p 之后
       （未来）token 的影响；
    3. "稀疏性" 是否生效，即每个 query 最多只能关注 topk 个 past token
       （这是 DSA 相对于标准全量注意力的核心区别，用于降低长序列的注意力计算量）；
    4. 自研的 `LightningIndexer`（轻量索引器，用于给每个 query 对所有历史 token
       打分并挑出 top-K 相关 token）是否与 Hugging Face `transformers` 库中
       GLM-MoE-DSA 参考实现的打分结果一致；
    5. 带 KV 缓存（KV cache）的自回归生成与不带缓存的逐 token 生成，
       在结果上是否完全一致（这是验证 KV cache 实现正确性的经典手段）；
    6. 当 topk 取值 >= 序列长度时，稀疏注意力的掩码退化为全 0（即不过滤任何
       token），此时 DSA 的输出应与普通稠密因果注意力完全相同。

这些测试是理解 "稀疏注意力如何在保持因果性的前提下，通过 top-K 选择降低
需要计算注意力权重的 token 数量" 这一思想的很好的教学示例。
"""

import os

import pytest
import torch
import torch.nn as nn
import tiktoken

from gpt_with_kv_dsa import (
    GPTModel,
    LightningIndexer,
    MultiHeadAttentionWithDSA,
    generate_text_simple_cached,
)


def import_transformers_dsa_model():
    """Import the reference DSA model from the installed Transformers package."""
    # 中文说明：尝试从已安装的 transformers 库中导入 GLM-MoE-DSA 的官方参考实现
    # （GlmMoeDsaConfig 配置类和 GlmMoeDsaModel 模型类），用作"标准答案"，
    # 与本仓库自己实现的 LightningIndexer 做数值对比测试。
    try:
        from transformers import GlmMoeDsaConfig, GlmMoeDsaModel
    except ImportError as err:
        # 如果当前环境没有安装支持 DSA 的 transformers 版本，则跳过依赖它的测试，
        # 而不是让测试直接失败——这样在没有该依赖的环境中也不会误报错误。
        pytest.skip(f"Transformers GLM-MoE-DSA reference unavailable: {err}")

    return GlmMoeDsaConfig, GlmMoeDsaModel


def test_output_shape():
    """Output shape must be (batch, seq_len, d_out).
    中文：验证 DSA 多头注意力模块的输出张量形状是否为 (batch, seq_len, d_out)，
    这是最基础的"接口契约"测试——保证模块可以正常嵌入到 Transformer block 中使用。
    """
    torch.manual_seed(0)  # 固定随机种子，保证测试结果可复现
    b, T, d = 2, 20, 128  # b=批大小, T=序列长度(token数), d=模型维度(emb_dim)
    attn = MultiHeadAttentionWithDSA(
        d_in=d, d_out=d, dropout=0.0, num_heads=4,
        index_n_heads=2, index_head_dim=16, topk=5,
    )
    x = torch.randn(b, T, d)  # 构造随机输入，形状 (b, T, d)
    out = attn(x)
    assert out.shape == (b, T, d), f"Wrong shape: {out.shape}"


def test_causal_property():
    """Tokens at position p must not be affected by tokens at positions > p.
    中文：验证"因果性"（causality）——自回归语言模型要求，在预测第 p 个位置时，
    模型只能看到第 0..p 位置的信息，绝不能看到 p 之后（未来）的 token。
    这是通过对比"修改未来 token 前后，前 6 个位置的输出是否不变"来验证的。
    """
    torch.manual_seed(1)
    b, T, d = 1, 20, 128
    attn = MultiHeadAttentionWithDSA(
        d_in=d, d_out=d, dropout=0.0, num_heads=4,
        index_n_heads=2, index_head_dim=16, topk=5,
    )
    x = torch.randn(b, T, d)
    out_full = attn(x)  # 用原始输入跑一遍，得到基准输出

    # Replace tokens at positions 6+ with random noise
    # 中文：把位置 6 及之后的 token 替换成完全随机的噪声，
    # 如果因果掩码正确生效，这个改动不应该影响位置 0~5 的输出结果。
    x_noisy = x.clone()
    x_noisy[:, 6:, :] = torch.randn(b, T - 6, d)
    out_noisy = attn(x_noisy)

    # 断言：替换未来 token 后，前 6 个位置的输出应与原始输出完全一致（在数值容差内）
    torch.testing.assert_close(out_noisy[:, :6, :], out_full[:, :6, :], rtol=0, atol=1e-5)


def test_sparsity():
    """Each query must attend to at most topk tokens.
    中文：验证 DSA 的核心"稀疏性"约束——在因果掩码的基础上叠加 top-K 选择后，
    每个 query 位置实际能关注到的历史 token 数量不应超过 topk。
    这是 DSA 区别于标准全量（dense）注意力、能降低计算量的关键特性。
    """
    torch.manual_seed(2)
    b, T, d = 1, 20, 128
    topk = 5
    attn = MultiHeadAttentionWithDSA(
        d_in=d, d_out=d, dropout=0.0, num_heads=4,
        index_n_heads=2, index_head_dim=16, topk=topk,
    )
    x = torch.randn(b, T, d)

    # Reconstruct the combined (causal + sparse) mask
    # 中文：手动重新构造"因果掩码 + 稀疏掩码"的组合掩码，用来独立核对
    # forward 内部逻辑是否真的把每行的可见 token 数限制在 topk 以内。
    q_pos = torch.arange(T)
    k_pos = torch.arange(T)
    # causal_bool[i, j] = True 表示 query 位置 i 不能看到 key 位置 j（j 在未来）
    causal_bool = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
    # 因果掩码：未来位置填 -inf（softmax 后趋于 0），其余位置为 0
    causal_float = torch.zeros(T, T).masked_fill_(causal_bool, float("-inf"))

    # 调用 LightningIndexer 得到每个 query 位置得分最高的 topk 个历史 token 下标
    # 形状: (b, T, topk)
    topk_idx = attn.indexer(x, x, topk)
    # 稀疏掩码初始化为全 -inf（默认不可见），再把 topk 选中的位置置 0（可见）
    sparse_mask = torch.full((b, T, T), float("-inf"))
    sparse_mask.scatter_(-1, topk_idx, 0.0)

    # 因果掩码 + 稀疏掩码相加：只要有一个是 -inf，结果就是 -inf（不可见）
    combined = causal_float.unsqueeze(0) + sparse_mask   # (1, T, T)
    # 统计每一行（每个 query 位置）中"可见"（值 > -inf）的 key 数量
    counts = (combined[0] > float("-inf")).sum(dim=-1).float()

    # 断言：所有 query 位置可见的 token 数都不超过 topk
    assert int(counts.max()) <= topk


@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Transformers reference test is too expensive for GitHub Actions",
)
def test_indexer_matches_transformers_reference():
    """The indexer must match the Transformers DSA scoring path.
    中文：将本仓库自己实现的 LightningIndexer 与 Hugging Face transformers 库中
    GLM-MoE-DSA 官方参考实现的 indexer 做严格数值对比——把两边的权重矩阵手动
    对齐（复制过去）之后，喂入相同输入，验证两者选出的 top-K 索引完全一致。
    这个测试计算开销较大，因此在 GitHub Actions CI 环境中会被跳过。
    """
    torch.manual_seed(4)
    b, T, d = 2, 6, 32
    topk = 3
    # 构造本仓库自己实现的轻量索引器
    indexer = LightningIndexer(d_model=d, index_n_heads=4, index_head_dim=8)
    GlmMoeDsaConfig, GlmMoeDsaModel = import_transformers_dsa_model()
    # 构造与本仓库参数尽量对齐的 transformers 参考模型配置
    reference_cfg = GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=d,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=8,
        index_n_heads=indexer.index_n_heads,
        index_head_dim=indexer.index_head_dim,
        index_topk=topk,
        q_lora_rank=d,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        n_routed_experts=4,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        mlp_layer_types=["dense"],
    )
    reference_model = GlmMoeDsaModel(reference_cfg)
    reference_model.eval()  # 关闭 dropout 等训练态行为，保证结果确定性
    # 取出参考模型第 0 层自注意力中的 indexer 子模块
    reference = reference_model.layers[0].self_attn.indexer
    # 参考实现里 k_norm 会对 key 做归一化，这里替换成恒等映射（不做归一化），
    # 使其与本仓库实现的计算路径保持一致，方便逐数值对比。
    reference.k_norm = nn.Identity()
    with torch.no_grad():
        # 把本仓库实现的三组线性层权重，逐一拷贝到参考实现对应的权重上，
        # 确保两边用的是完全相同的参数，这样输出差异只能来自算法逻辑本身。
        reference.wq_b.weight.copy_(indexer.W_q_index.weight)
        reference.wk.weight.copy_(indexer.W_k_index.weight)
        reference.weights_proj.weight.copy_(indexer.W_weights.weight)

    x = torch.randn(b, T, d)
    q_pos = torch.arange(T)
    k_pos = torch.arange(T)
    causal_bool = q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)
    causal_mask = torch.zeros(T, T).masked_fill_(causal_bool, float("-inf"))

    # 分别调用本仓库实现和 transformers 参考实现，得到各自选出的 top-K 索引
    actual = indexer(x, x, topk=topk, causal_mask=causal_mask)
    # 参考实现的接口需要传入 RoPE（旋转位置编码）相关的 (cos, sin) 元组，
    # 由于本教学实现的 indexer 不使用 RoPE，这里传入形状为 (b, T, 0) 的空张量占位。
    empty_rope = torch.empty(b, T, 0)
    expected = reference(x, x, (empty_rope, empty_rope), causal_mask)
    # 断言：两边选出的 top-K 索引必须完全相等（严格逐元素比较，无容差）
    assert torch.equal(actual, expected)


def test_cache_consistency():
    """Cached and non-cached generation must produce identical token sequences.
    中文：这是验证 KV 缓存（KV cache）正确性的关键测试——KV cache 是一种
    自回归生成时的推理加速技巧：把已经计算过的 key/value 张量缓存起来，
    避免在生成新 token 时重复计算历史 token 的 key/value。
    正确实现的 KV cache 不应改变模型的生成结果，因此这里对比
    "关闭缓存、每步重新计算全部历史" 与 "开启缓存、每步只算新 token"
    两种方式生成的 token 序列，要求二者完全一致。
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode("Hello, I am")
    cfg = {
        "vocab_size": 50257,
        "context_length": 30,
        "emb_dim": 256,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "index_n_heads": 2,
        "index_head_dim": 32,
        "topk": 200,   # large topk == full attention, so both modes match exactly
        # 中文：topk 设得远大于实际序列长度，相当于"top-K 选择永远选中全部历史
        # token"，即退化为普通全量因果注意力，这样才能保证有无 KV cache 的结果
        # 完全一致（否则 top-K 的选择结果可能因缓存拼接顺序等细节产生偏差）。
    }
    torch.manual_seed(42)
    model = GPTModel(cfg)
    model.eval()  # 关闭 dropout，保证两种生成方式的随机性来源一致（其实都为 0）
    idx = torch.tensor(encoded).unsqueeze(0)  # 形状 (1, prompt_len)
    # 不使用缓存：每一步都把"到目前为止的完整序列"重新喂给模型
    out_no_cache = generate_text_simple_cached(model, idx.clone(), max_new_tokens=5, use_cache=False)
    # 使用缓存：只在第一步喂入完整 prompt 建立缓存，之后每步只喂入新生成的 1 个 token
    out_with_cache = generate_text_simple_cached(model, idx.clone(), max_new_tokens=5, use_cache=True)
    # 断言：两种方式生成的 token 序列必须完全相同
    assert torch.equal(out_no_cache, out_with_cache)


def dense_attention_reference(attn, x):
    """Dense causal attention using the same projections as the DSA module.
    中文：这是一个"标准稠密因果注意力"的参考实现，复用传入的 DSA 注意力模块
    (attn) 中已经训练/初始化好的 W_query / W_key / W_value / out_proj 权重，
    但不做任何 top-K 稀疏选择，只应用普通的因果掩码。
    用于和 DSA 在 topk >= 序列长度时的输出做数值对比。

    Args:
        attn: MultiHeadAttentionWithDSA 实例，提供投影权重和头数/维度等超参数。
        x:    输入张量，形状 (b, T, d_in)。

    Returns:
        输出张量，形状 (b, T, d_out)。
    """
    b, T, _ = x.shape

    # 分别将输入投影为 Q/K/V，并 reshape 成多头形式，再转置成
    # (b, num_heads, T, head_dim) 以便做批量矩阵乘法。
    queries = attn.W_query(x).view(b, T, attn.num_heads, attn.head_dim).transpose(1, 2)
    keys = attn.W_key(x).view(b, T, attn.num_heads, attn.head_dim).transpose(1, 2)
    values = attn.W_value(x).view(b, T, attn.num_heads, attn.head_dim).transpose(1, 2)

    # 计算注意力得分：Q @ K^T，形状 (b, num_heads, T, T)
    attn_scores = queries @ keys.transpose(2, 3)
    # 标准因果掩码：上三角（不含对角线）位置为 True，代表"未来"，需要屏蔽
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    attn_scores = attn_scores.masked_fill(mask, float("-inf"))

    # 按 head_dim 的平方根缩放后做 softmax，得到注意力权重
    attn_weights = torch.softmax(attn_scores / attn.head_dim ** 0.5, dim=-1)
    # 加权求和得到上下文向量：(b, num_heads, T, head_dim)
    context_vec = attn_weights @ values
    # 把多头拼回单个维度：(b, T, num_heads, head_dim) -> (b, T, d_out)
    context_vec = context_vec.transpose(1, 2).contiguous().view(b, T, attn.d_out)
    return attn.out_proj(context_vec)


def test_topk_full_equals_dense():
    """With topk >= seq_len the sparse mask is all-zeros -> identical to dense attention.
    中文：当 topk 大于等于序列长度 T 时，LightningIndexer 对每个 query 选出的
    "top-K 个历史 token" 实际上就是全部历史 token（因为总共也没有那么多可选），
    此时稀疏掩码退化为全 0（不额外屏蔽任何位置），DSA 应当与普通稠密因果注意力
    在数值上完全等价。这个测试用于验证"稀疏机制在极限情况下能正确退化"。
    """
    torch.manual_seed(3)
    b, T, d = 1, 10, 64

    # topk=T：意味着每个 query 最多可以选中全部 T 个历史 token，等价于不做稀疏化
    attn_full = MultiHeadAttentionWithDSA(
        d_in=d, d_out=d, dropout=0.0, num_heads=4,
        index_n_heads=2, index_head_dim=16, topk=T,
    )
    x = torch.randn(b, T, d)
    out_dsa = attn_full(x)  # 走 DSA 的稀疏注意力路径（但实际不过滤任何 token）
    out_dense = dense_attention_reference(attn_full, x)  # 走普通稠密因果注意力路径
    # 断言：两种路径的输出在数值上应当一致（允许极小的浮点误差）
    torch.testing.assert_close(out_dsa, out_dense, rtol=0, atol=1e-5)
