# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本文件是 pytest 单元测试模块,用于验证《从零构建大语言模型》一书配套代码库中
Llama3 相关实现的正确性。

主要覆盖以下几个方面:
1. RoPE(旋转位置编码,Rotary Position Embedding)参数计算与应用是否与
   HuggingFace transformers 官方实现的 Llama RoPE 结果一致(test_rope)。
2. 分组查询注意力(Grouped Query Attention, GQA)的"慢速参考实现"
   GroupedQueryAttention 与"快速优化实现"GroupedQueryAttentionFast
   在数值上是否等价(test_grouped_query_attention_equivalence)。
3. 完整 Llama3Model 在有/无 KV 缓存(KV Cache)两种模式下,配合两种不同的
   文本生成函数(generate_text_simple / generate_text_simple_cached),
   生成结果是否与预期完全一致(test_model_variants)。
4. 自定义 RMSNorm(均方根层归一化)实现与 PyTorch 官方 torch.nn.RMSNorm
   以及 LitGPT 版本 RMSNorm 是否数值等价(test_rmsnorm_equivalence)。
5. 完整 Llama3Model 与 HuggingFace transformers 的 LlamaForCausalLM
   在加载相同权重后,前向传播输出的 logits 是否一致
   (test_llama3_base_equivalence_with_transformers)。

依赖说明:
- 部分测试依赖可选的 `transformers` 库,若未安装则通过
  `pytest.mark.skipif` 自动跳过。
- 部分测试在 GitHub Actions CI 环境中因算力/内存限制而跳过。
"""

from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.llama3 import (
    apply_rope,
    compute_rope_params,
    GroupedQueryAttention,
    GroupedQueryAttentionFast,
    load_weights_into_llama,
    LLAMA32_CONFIG_1B,
    Llama3Model,
)
from llms_from_scratch.kv_cache.llama3 import Llama3Model as Llama3ModelKV
from llms_from_scratch.kv_cache.generate import generate_text_simple as generate_text_simple_cached

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
import os
import pytest
import tiktoken
import torch


class LitGPTRMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    From https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
    Apache License 2.0-Clause License: https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """
    # 中文说明:这是从 LitGPT 项目移植过来的 RMSNorm(均方根层归一化)参考实现,
    # 在测试中作为"标准答案"之一,用来与本仓库/PyTorch 官方实现做数值对比。

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        # 可学习的缩放权重参数,初始化为全 1,形状为 (size,)
        self.weight = torch.nn.Parameter(torch.ones(size))
        # 防止除零的极小值 epsilon
        self.eps = eps
        # 计算均方值时所沿的维度,默认是最后一维(特征维)
        self.dim = dim
        # 是否在权重上加 1(即使用 "1 + weight" 而不是直接用 weight),
        # 某些实现(如 Gemma)采用这种"单位偏移"写法
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 记录输入原始 dtype,便于计算完成后转换回去(常用于混合精度场景)
        dtype = x.dtype
        # 先转为 float32 计算,避免半精度下的数值不稳定
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # 计算 x 在指定维度上的均方值(mean of squares),即 RMS 的平方
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        # 用 rsqrt(1/sqrt) 对 x 做归一化,eps 防止分母为 0
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        # 根据 add_unit_offset 决定是否给权重加 1
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        # 应用缩放权重,并转换回输入原始的 dtype
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        # 将权重重新初始化为全 1
        torch.nn.init.ones_(self.weight)


# 检测当前环境是否安装了 transformers 库,用于后续测试的条件跳过
transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_rope():
    """
    验证本仓库实现的 RoPE(旋转位置编码)——包括参数计算函数 compute_rope_params
    和应用函数 apply_rope——与 HuggingFace transformers 中 Llama 的官方
    RoPE 实现(LlamaRotaryEmbedding + apply_rotary_pos_emb)在数值上完全一致。

    该测试仅在环境中安装了 transformers 库时才会运行,否则会被跳过。
    """

    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

    # Settings
    # 以下为测试用的超参数设置
    batch_size = 1
    context_len = 8192  # 上下文长度(序列长度)
    num_heads = 4        # 注意力头数
    head_dim = 16         # 每个注意力头的维度
    rope_theta = 500_000   # RoPE 的基础频率参数 theta

    # Llama3 风格的 RoPE 频率缩放配置(用于长上下文扩展)
    rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }

    # Instantiate RoPE parameters
    # 调用本仓库实现,计算出 cos/sin 位置编码表
    cos, sin = compute_rope_params(
        head_dim=head_dim,
        theta_base=rope_theta,
        context_length=context_len,
        freq_config=rope_config,
    )

    # Dummy query and key tensors
    # 构造用于测试的随机 query/key 张量(固定随机种子以保证可复现)
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    # 对 query/key 应用本仓库实现的旋转位置编码
    queries_rot = apply_rope(queries, cos, sin)
    keys_rot = apply_rope(keys, cos, sin)

    # Generate reference RoPE via HF
    # 构造与本仓库配置对应的 HuggingFace RoPE 参数,作为参照标准
    hf_rope_params = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    }

    class RoPEConfig:
        # 中文说明:模拟 HuggingFace 的配置对象(config),用于实例化
        # LlamaRotaryEmbedding。字段命名需与 transformers 内部期望的属性一致。
        rope_type = "llama3"
        rope_scaling = hf_rope_params
        factor = 1.0
        dim: int = head_dim
        rope_theta = 500_000
        max_position_embeddings: int = 8192
        hidden_size = head_dim * num_heads
        num_attention_heads = num_heads

        def __init__(self):
            # Transformers >=5.0.0 expects `rope_parameters` on the instance.
            # 中文说明:较新版本的 transformers 要求配置对象上存在
            # rope_parameters 属性,这里手动补充,兼容新旧版本 API。
            self.rope_parameters = {**hf_rope_params, "rope_theta": rope_theta}

        def standardize_rope_params(self):
            # 中文说明:将 rope_parameters 标准化为 transformers 内部期望的格式,
            # 补齐 rope_type / rope_theta 等必要字段,并处理旧字段名的兼容映射。
            params = dict(getattr(self, "rope_parameters", {}) or {})
            if "rope_type" not in params:
                params["rope_type"] = getattr(self, "rope_type", "default")
            if "rope_theta" not in params:
                params["rope_theta"] = getattr(self, "rope_theta")
            # Handle older key name used in this repo.
            # 处理本仓库中使用的旧字段名 original_context_length,
            # 将其映射为 transformers 期望的 original_max_position_embeddings
            if (
                "original_max_position_embeddings" not in params
                and "original_context_length" in params
            ):
                params["original_max_position_embeddings"] = params["original_context_length"]
            self.rope_parameters = params
            return params

    config = RoPEConfig()

    # 实例化 HuggingFace 官方的旋转位置编码模块
    rot_emb = LlamaRotaryEmbedding(config=config)
    # 构造位置索引张量 (0, 1, 2, ..., context_len-1)
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    # 计算 HuggingFace 版本的参照 cos/sin
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    # 使用 HuggingFace 官方函数对 query/key 应用旋转位置编码,作为参照结果
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    # 以下四个断言分别核对 sin 表、cos 表、key 旋转结果、query 旋转结果
    # 是否与 HuggingFace 参照实现完全吻合(在浮点误差容忍范围内)
    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)


# 用于其他测试(如 GPT 相关)的基础配置字典,此处保留但当前文件内未直接使用其全部字段
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}


def test_grouped_query_attention_equivalence():
    """
    验证分组查询注意力(GQA)的两种实现——朴素/慢速版本 GroupedQueryAttention
    与优化/快速版本 GroupedQueryAttentionFast——在加载相同权重、相同输入的
    情况下,前向传播输出是否数值等价(误差在 atol=1e-4 范围内)。
    """
    torch.manual_seed(42)
    # b:批大小, t:序列长度, d_in:输入维度, d_out:输出维度,
    # num_heads:注意力头数, num_kv_groups:key/value 分组数(GQA 核心参数)
    b, t, d_in, d_out, num_heads, num_kv_groups = 2, 8, 32, 64, 4, 2

    x = torch.randn(b, t, d_in)
    # 计算 RoPE 所需的 cos/sin 表,head_dim = d_out // num_heads
    cos, sin = compute_rope_params(
        head_dim=d_out // num_heads,
        theta_base=50_000,
        context_length=t,
        freq_config={
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": t,
        }
    )

    # Causal mask for the slow version
    # 构造因果掩码(上三角为 True,表示需要被屏蔽的未来位置),仅慢速版本需要显式传入
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)

    # 分别实例化慢速参考实现与快速优化实现
    attn1 = GroupedQueryAttention(d_in, d_out, num_heads, num_kv_groups)
    attn2 = GroupedQueryAttentionFast(d_in, d_out, num_heads, num_kv_groups)

    # Copy weights to make both models identical
    # 将 attn1 的权重复制给 attn2,确保两者参数完全一致,才能公平比较输出差异
    attn2.load_state_dict(attn1.state_dict())

    # Run both
    # 分别对同一输入 x 执行前向传播
    y1 = attn1(x, mask, cos, sin)
    y2 = attn2(x, cos, sin)

    # Compare outputs
    # 计算两者输出的最大绝对差值,并打印出来便于调试
    max_diff = (y1 - y2).abs().max().item()
    print(f"Max difference between slow and fast outputs: {max_diff:.4e}")
    # 核心断言:两种实现的输出在给定绝对误差容限内应当一致
    assert torch.allclose(y1, y2, atol=1e-4)


@pytest.fixture(scope="session")
def llama3_weights_path(tmp_path_factory):
    """Creates and saves a deterministic Llama3 model for testing."""
    # 中文说明:这是一个会话级(session scope)pytest 夹具(fixture),
    # 在整个测试会话中只创建一次确定性的 Llama3 模型权重文件,
    # 供多个测试用例复用,避免重复初始化模型带来的开销。
    path = tmp_path_factory.mktemp("models") / "llama3_test_weights.pt"

    if not path.exists():
        # 固定随机种子,保证每次生成的模型权重是确定且可复现的
        torch.manual_seed(123)
        model = Llama3Model(LLAMA32_CONFIG_1B)
        # 将模型的 state_dict(权重字典)保存到临时文件中
        torch.save(model.state_dict(), path)

    return path


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skipping in GitHub Actions due to compute or memory constraints"
)
@pytest.mark.parametrize("ModelClass", [Llama3Model, Llama3ModelKV])
@pytest.mark.parametrize("generate_fn", [generate_text_simple, generate_text_simple_cached])
def test_model_variants(ModelClass, generate_fn, llama3_weights_path):
    """
    验证不同的 Llama3 模型实现(普通版 Llama3Model / 带 KV 缓存版 Llama3ModelKV)
    在配合各自适配的文本生成函数(普通版 generate_text_simple / 支持 KV 缓存的
    generate_text_simple_cached)时,加载相同权重后生成的 token 序列是否与
    预先记录的期望结果(expect)完全一致。

    通过 pytest.mark.parametrize 对 ModelClass 与 generate_fn 做笛卡尔积组合测试,
    其中不匹配的组合(如普通模型配合缓存版生成函数)会在函数体内被主动跳过。

    在 GitHub Actions CI 环境中该测试会被跳过,原因是算力/内存受限。
    """

    # Skip incompatible combinations
    # 跳过不兼容的组合:普通生成函数不能用于支持 KV 缓存(reset_kv_cache)的模型
    if generate_fn is generate_text_simple and getattr(ModelClass, "reset_kv_cache", False):
        return
    # 跳过不兼容的组合:缓存版生成函数不能用于不支持 KV 缓存的普通模型
    if generate_fn is generate_text_simple_cached and not getattr(ModelClass, "reset_kv_cache", False):
        return

    # 固定随机种子后实例化模型(种子主要影响未被权重覆盖的部分,如 dropout 等)
    torch.manual_seed(123)
    model = ModelClass(LLAMA32_CONFIG_1B)
    # 加载之前由 llama3_weights_path 夹具生成的确定性权重
    model.load_state_dict(torch.load(llama3_weights_path, weights_only=True))
    # 切换为评估模式,关闭 dropout 等训练专用行为
    model.eval()

    start_context = "Llamas eat"

    # 使用 GPT-2 的 tokenizer 对输入文本进行编码
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # 调用对应的生成函数(可能是普通自回归生成,也可能是带 KV 缓存加速的生成),
    # 从输入 token 序列继续生成 5 个新 token
    out = generate_fn(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=5,
        context_size=LLAMA32_CONFIG_1B["context_length"]
    )
    print("Encoded output text:", out)
    # 预先记录的期望输出 token 序列(基于固定随机种子和确定性权重得到)
    expect = torch.tensor([
        [43, 2543, 292, 4483, 100383, 8113, 76873, 42175, 72641]
    ])
    # 核心断言:实际生成结果必须与预期结果完全相等
    assert torch.equal(expect, out)


def test_rmsnorm_equivalence():
    """
    验证 PyTorch 官方实现 torch.nn.RMSNorm 与本文件中移植自 LitGPT 的
    LitGPTRMSNorm 在相同输入、相同(默认)权重下,输出是否数值等价。
    """
    torch.manual_seed(42)

    hidden_size = 64
    batch_size = 8
    seq_len = 16

    # PyTorch 官方 RMSNorm 实现
    rms_norm = torch.nn.RMSNorm(hidden_size, eps=1e-6)
    # LitGPT 移植版 RMSNorm 实现
    lit_norm = LitGPTRMSNorm(hidden_size)

    # Sync weights
    # 中文说明:此处将 lit_norm 的权重复制给自身(自我拷贝),
    # 由于两者权重均默认初始化为全 1,因此实际上无需额外同步即可保证一致;
    # 这行代码更多是保留了"同步权重"这一操作步骤的意图占位。
    with torch.no_grad():
        lit_norm.weight.copy_(lit_norm.weight)

    x = torch.randn(batch_size, seq_len, hidden_size)

    # 分别用两种实现对相同输入做前向传播
    out1 = rms_norm(x)
    out2 = lit_norm(x)

    # 核心断言:两种 RMSNorm 实现的输出应在给定误差范围内一致
    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_llama3_base_equivalence_with_transformers():
    """
    端到端验证:将 HuggingFace transformers 的 LlamaForCausalLM 随机初始化后的
    权重,通过 load_weights_into_llama 转换并加载进本仓库实现的 Llama3Model 中,
    然后用同一批随机输入 token 分别做前向传播,核对两者输出的 logits 是否一致。

    该测试仅在环境中安装了 transformers 库时才会运行,否则会被跳过;
    整个前向传播过程处于 torch.inference_mode() 下以节省显存/内存并禁用梯度计算。
    """
    from transformers.models.llama import LlamaConfig, LlamaForCausalLM
    # 定义一个小规模的 Llama3 配置,用于快速测试(不追求真实规模,只追求结构一致)
    cfg = {
        "vocab_size": 257,
        "context_length": 8192,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "n_kv_groups": 2,
        "rope_base": 500_000.0,
        "rope_freq": {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        },
        "dtype": torch.float32,
    }

    # 实例化本仓库的 Llama3Model(权重此时是随机初始化的,稍后会被覆盖)
    ours = Llama3Model(cfg)

    # 构造与本仓库配置字段一一对应的 HuggingFace LlamaConfig
    hf_cfg = LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg["n_kv_groups"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        max_position_embeddings=cfg["context_length"],
        rms_norm_eps=1e-5,
        attention_bias=False,
        rope_theta=cfg["rope_base"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        rope_scaling={
            "type": "llama3",
            "factor": cfg["rope_freq"]["factor"],
            "low_freq_factor": cfg["rope_freq"]["low_freq_factor"],
            "high_freq_factor": cfg["rope_freq"]["high_freq_factor"],
            "original_max_position_embeddings": cfg["rope_freq"]["original_context_length"],
        },
    )
    # 实例化 HuggingFace 官方的 LlamaForCausalLM(带随机初始化权重),作为"金标准"
    theirs = LlamaForCausalLM(hf_cfg)

    # 取出 HuggingFace 模型的权重字典
    hf_state = theirs.state_dict()
    # 调用本仓库提供的权重转换/加载函数,把 HuggingFace 格式的权重
    # 映射并加载到本仓库实现的 Llama3Model(ours)中,使两者权重完全一致
    load_weights_into_llama(ours, {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}, hf_state)

    # 构造相同的随机输入 token 序列(批大小 2,序列长度 8)
    x = torch.randint(0, cfg["vocab_size"], (2, 8), dtype=torch.long)
    # 分别用本仓库实现与 HuggingFace 实现做前向传播
    ours_logits = ours(x)
    theirs_logits = theirs(x).logits.to(ours_logits.dtype)

    # 核心断言:在权重完全一致的前提下,两者输出的 logits 应当高度接近
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
