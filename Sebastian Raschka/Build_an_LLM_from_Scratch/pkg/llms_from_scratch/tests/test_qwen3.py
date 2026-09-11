# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块为 Qwen3 实现的 pytest 单元测试集合。

测试覆盖范围包括：
1. 基础 Qwen3Model 前向传播（含普通稠密前馈网络与 MoE 混合专家前馈网络）的形状与逻辑正确性；
2. MoE（MoEFeedForward）门控/专家路由的手工参考实现对照测试；
3. 带 KV 缓存（KVCache）的模型与不带缓存的普通模型在数值上的等价性测试（含单条与批量两种场景）；
4. RoPE（旋转位置编码）自实现与 HuggingFace transformers 官方实现的数值一致性测试；
5. RMSNorm 自实现与 HuggingFace Qwen3RMSNorm 参考实现的数值一致性测试；
6. Qwen3Tokenizer 分词器在特殊 token、聊天模板包装、多轮对话、前缀稳定性等方面
   与 HuggingFace AutoTokenizer 的等价性测试；
7. 使用真实（但极小规模）配置的 Qwen3Model 与 HuggingFace Qwen3ForCausalLM
   之间的端到端数值等价性测试（用于验证权重加载与整体架构实现的正确性）。

其中依赖 HuggingFace `transformers` 库或需要联网下载模型/分词器文件的测试，
均通过 `pytest.mark.skipif` 在未安装/不可用时自动跳过。
"""

from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import (
    apply_rope,
    compute_rope_params,
    load_weights_into_qwen,
    QWEN_CONFIG_06_B,
    Qwen3Model,
    Qwen3Tokenizer,
    MoEFeedForward,
    RMSNorm,
)
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model as Qwen3ModelKV
from llms_from_scratch.kv_cache.utils import KVCache
from llms_from_scratch.kv_cache.generate import generate_text_simple as generate_text_simple_cached

from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model as Qwen3ModelKVBatched
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple as generate_text_simple_batched

from llms_from_scratch.utils import download_file

# 以下为标准库与第三方库导入：
# importlib   -- 用于动态检测可选依赖（如 transformers）是否已安装
# os / shutil / tempfile -- 用于处理分词器文件的下载、重命名与临时文件操作
# platform    -- 用于判断操作系统类型，从而跳过某些平台上不稳定的测试（如 Linux 上的 MoE 路由）
# collections.abc.Mapping -- 用于兼容判断 HuggingFace 返回对象是否为映射类型
# pytest      -- 测试框架，提供 fixture、参数化、跳过标记等能力
# torch / torch.nn -- PyTorch 深度学习框架，构建与运行模型
import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
import os
import shutil
import tempfile
import platform
from collections.abc import Mapping
import pytest
import torch
import torch.nn as nn


class Qwen3RMSNorm(nn.Module):
    # Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
    # License: Apache License, Version 2.0 (see file above)
    # 中文说明：这是从 HuggingFace transformers 源码中直接搬运的 Qwen3RMSNorm 参考实现，
    # 用作“标准答案”，在测试中与本项目自实现的 RMSNorm 做数值对比，验证自实现的正确性。
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        # 中文说明：weight 为可学习的缩放参数，初始化为全 1；variance_epsilon 为防止除零的极小值。
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 中文说明：先记录输入的原始 dtype，再将输入转换为 float32 以提升数值稳定性（RMSNorm 对精度敏感）。
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # 计算最后一维上各元素平方的均值，即“均方”（mean square），用于替代方差（RMSNorm 不减均值）。
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 用均方的平方根倒数对输入做归一化（rsqrt = 1/sqrt）。
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        print(input_dtype)
        # 归一化后转换回原始 dtype，再乘以可学习权重，得到最终输出。
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# 中文说明：检测当前环境是否安装了 transformers 库；若未安装，相关对照测试将被跳过（见下方 skipif 标记）。
transformers_installed = importlib.util.find_spec("transformers") is not None


def _hf_ids(obj):
    """Normalize HF chat-template outputs across Transformers versions."""
    # 中文说明：不同版本的 HuggingFace transformers 在 apply_chat_template 等接口上
    # 返回的数据结构并不统一（有的是纯 list，有的是字典/类字典对象，有的是 tensor 或 tuple，
    # 有的还会多包一层 batch 维度）。本函数的作用是把这些五花八门的返回值统一“拍平”成
    # 一个普通的 Python int 列表，方便后续测试直接做相等性比较。
    if isinstance(obj, Mapping):
        # 若是标准映射类型（如 dict），优先取 "input_ids"，其次取 "ids" 字段。
        if "input_ids" in obj:
            obj = obj["input_ids"]
        elif "ids" in obj:
            obj = obj["ids"]
    elif hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        # Some HF containers behave like mappings but don't register as Mapping.
        # 中文说明：部分 HF 容器行为上像字典（有 keys/__getitem__），但并未继承自 Mapping，
        # 因此需要单独兼容处理；用 try/except 防止意外抛错影响测试。
        try:
            if "input_ids" in obj:
                obj = obj["input_ids"]
            elif "ids" in obj:
                obj = obj["ids"]
        except Exception:
            pass
    if hasattr(obj, "input_ids"):
        # 某些对象（如 BatchEncoding）以属性形式暴露 input_ids。
        obj = obj.input_ids
    if hasattr(obj, "ids"):
        # 某些 tokenizers 库的 Encoding 对象以 .ids 属性暴露 token id 列表。
        obj = obj.ids
    if isinstance(obj, torch.Tensor):
        # 若结果是张量，转换为普通嵌套列表。
        obj = obj.tolist()
    if isinstance(obj, tuple):
        # 若结果是元组，转换为列表，方便后续统一处理。
        obj = list(obj)
    # Some HF versions return a batched structure even for a single prompt.
    # 中文说明：某些 HF 版本即便只传入单条 prompt，也会返回带 batch 维度的嵌套列表
    # （即 [[...]] 形式），这里将其“解包”为单层列表。
    if isinstance(obj, list) and obj and isinstance(obj[0], list) and len(obj) == 1:
        obj = obj[0]
    return list(obj)


@pytest.fixture
def dummy_input():
    """
    中文说明：构造一个用于快速前向传播测试的“假”输入张量。
    使用固定随机种子保证测试结果可复现；返回形状为 (batch_size=1, seq_len=8) 的
    随机 token id 张量，取值范围 [0, 100)。
    """
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    """
    中文说明：提供一个规模极小的基础模型配置字典，用于快速单元测试（不含 MoE，num_experts=0）。
    该配置刻意将各维度（emb_dim、hidden_dim、n_layers 等）设置得很小，
    以保证测试运行速度快、显存/内存占用低，同时仍能覆盖 Qwen3Model 的核心前向逻辑。
    """
    return {
        "vocab_size": 100,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 8,
        "n_kv_groups": 1,
        "qk_norm": False,
        "dtype": torch.float32,
        "rope_base": 1000000,
        "context_length": 64,
        "num_experts": 0,
    }


@pytest.fixture
def dummy_cfg_moe(dummy_cfg_base):
    """
    中文说明：在 dummy_cfg_base 基础上派生出的 MoE（混合专家）配置。
    通过 num_experts、num_experts_per_tok、moe_intermediate_size 等字段
    开启 MoEFeedForward 前馈层，用于测试模型在 MoE 模式下的前向传播是否正常。
    """
    cfg = dummy_cfg_base.copy()
    cfg.update({
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
    })
    return cfg


@torch.inference_mode()
def test_dummy_qwen3_forward(dummy_cfg_base, dummy_input):
    """
    中文说明：验证使用基础（非 MoE）配置构建的 Qwen3Model 能正常完成一次前向传播，
    且输出 logits 的形状符合预期 (batch_size, seq_len, vocab_size)。
    """
    torch.manual_seed(123)
    model = Qwen3Model(dummy_cfg_base)
    out = model(dummy_input)
    # 断言：输出形状必须是 (1, 序列长度, 词表大小)，否则说明前向传播中维度处理有误。
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"


@torch.inference_mode()
def test_dummy_qwen3_moe_forward(dummy_cfg_moe, dummy_input):
    """
    中文说明：验证使用 MoE 配置构建的 Qwen3Model 能正常完成前向传播，
    输出形状正确，并且模型内部的 Transformer block 确实使用了 MoEFeedForward
    （通过检查 block.ff 是否具有 "gate" 属性来判断，因为 MoEFeedForward 内部有门控网络 gate）。
    """
    torch.manual_seed(123)
    model = Qwen3Model(dummy_cfg_moe)
    out = model(dummy_input)
    # 断言：输出形状同样应为 (1, 序列长度, 词表大小)。
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_moe["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"
    # 断言：至少有一个 Transformer block 的前馈层（ff）是 MoEFeedForward（拥有 gate 属性），
    # 确认 MoE 分支确实被启用，而不是退化成普通前馈网络。
    assert any(hasattr(block.ff, "gate") for block in model.trf_blocks), \
        "Expected MoEFeedForward in at least one transformer block"


@torch.inference_mode()
def test_moe_forward_matches_reference(dummy_cfg_moe):
    """
    中文说明：手工按照 MoE（混合专家）的原始定义重新实现一遍前向计算逻辑
    （门控打分 -> top-k 专家选择 -> softmax 归一化权重 -> 各专家独立计算 -> 加权求和），
    并将结果与 MoEFeedForward.forward 的实际输出做数值对比，
    以验证 MoEFeedForward 内部的高效实现（可能使用了向量化/融合技巧）在数学上与朴素实现完全等价。
    """
    torch.manual_seed(0)
    moe = MoEFeedForward(dummy_cfg_moe)
    x = torch.randn(2, 5, dummy_cfg_moe["emb_dim"])

    # 门控网络对每个 token 在所有专家上打分。
    scores = moe.gate(x)
    # 选出得分最高的 top-k（num_experts_per_tok）个专家及其索引。
    topk_scores, topk_indices = torch.topk(scores, moe.num_experts_per_tok, dim=-1)
    # 对被选中的 top-k 专家得分做 softmax，得到归一化的加权系数。
    topk_probs = torch.softmax(topk_scores, dim=-1)

    # 手动计算每一个专家（不论是否被选中）对所有 token 的输出，供后续按门控概率加权求和。
    expert_outputs = []
    for e in range(moe.num_experts):
        # 专家的前馈计算：SiLU(fc1(x)) * fc2(x) 后经 fc3 投影，属于 SwiGLU 风格的前馈结构。
        hidden = torch.nn.functional.silu(moe.fc1[e](x)) * moe.fc2[e](x)
        out = moe.fc3[e](hidden)
        expert_outputs.append(out.unsqueeze(-2))
    expert_outputs = torch.cat(expert_outputs, dim=-2)

    # 将 top-k 的归一化权重“散射”回完整的 (num_experts,) 维度上，未被选中的专家权重为 0。
    gating_probs = torch.zeros_like(scores)
    for i in range(moe.num_experts_per_tok):
        indices = topk_indices[..., i:i+1]
        prob = topk_probs[..., i:i+1]
        gating_probs.scatter_(dim=-1, index=indices, src=prob)
    gating_probs = gating_probs.unsqueeze(-1)

    # 按门控权重对所有专家的输出做加权求和，得到参考（朴素）实现的最终输出。
    expected = (gating_probs * expert_outputs).sum(dim=-2)

    # 调用 MoEFeedForward 的实际前向实现。
    actual = moe(x)
    # 断言：手工朴素实现与模块实际实现在数值上应高度接近（容差 1e-5）。
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
@pytest.mark.parametrize("cfg_name", ["dummy_cfg_base", "dummy_cfg_moe"])
def test_qwen3_kvcache_equivalence(cfg_name, request):
    """
    中文说明：验证“逐 token 使用 KV 缓存推理”的结果与“一次性整体前向传播”的结果完全等价。
    该测试同时在基础配置（dummy_cfg_base）与 MoE 配置（dummy_cfg_moe）上参数化运行，
    以确保 KV 缓存机制在两种前馈网络模式下都不会引入数值偏差。
    在 Linux 平台上，MoE 的专家路由可能存在不确定性（例如并行计算顺序导致的浮点误差累积），
    因此该场景会被跳过，避免测试结果不稳定。
    """
    cfg = request.getfixturevalue(cfg_name)

    # 中文说明：Linux 上 MoE 专家路由可能不确定，跳过该组合以避免测试抖动（flaky）。
    if cfg["num_experts"] > 0 and platform.system() == "Linux":
        pytest.skip("Skipping MoE KV equivalence test on Linux due to nondeterministic expert routing")

    torch.manual_seed(123)
    model_regular = Qwen3Model(cfg)
    model_regular.eval()

    # 构建带 KV 缓存的模型版本，并让其权重与普通模型完全一致（load_state_dict 拷贝参数）。
    model_kv = Qwen3ModelKV(cfg)
    model_kv.eval()
    model_kv.load_state_dict(model_regular.state_dict())
    model_kv.reset_kv_cache()
    cache = KVCache(n_layers=cfg["n_layers"])

    torch.manual_seed(123)
    input_ids = torch.randint(0, cfg["vocab_size"], (1, 6))

    # 普通模型：一次性对整段输入做前向传播，得到完整的 logits。
    out_full = model_regular(input_ids)

    # KV 缓存模型：逐个 token 输入，每次只喂入当前 token，依赖缓存中的历史 K/V 完成注意力计算。
    logits_stepwise = []
    for t in range(input_ids.size(1)):
        input_token = input_ids[:, t:t + 1]
        logits = model_kv(input_token, cache=cache)
        logits_stepwise.append(logits)
    out_kv = torch.cat(logits_stepwise, dim=1)

    # 断言：两种方式得到的输出形状必须一致。
    assert out_full.shape == out_kv.shape, f"Shape mismatch: {out_full.shape} vs {out_kv.shape}"
    # 断言：两种方式得到的数值结果应在给定容差内近似相等，证明 KV 缓存实现正确。
    assert torch.allclose(out_full, out_kv, atol=1e-5, rtol=1e-3)


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
@pytest.mark.parametrize("context_len", [1024, 8192, 40960])
def test_rope(context_len):
    """
    中文说明：验证本项目自实现的 RoPE（旋转位置编码，Rotary Position Embedding）
    与 HuggingFace transformers 官方 Qwen3RotaryEmbedding / apply_rotary_pos_emb
    在不同上下文长度（1024、8192、40960）下的数值结果完全一致。
    该测试仅在安装了 transformers 库时运行，否则自动跳过。
    """

    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    # Settings
    # 中文说明：测试用的基础超参数设置。
    batch_size = 1
    num_heads = 4
    head_dim = 16
    rope_theta = 1_000_000

    # Instantiate RoPE parameters (our implementation)
    # 中文说明：使用本项目自实现的 compute_rope_params 计算 cos/sin 位置编码表。
    cos, sin = compute_rope_params(
        head_dim=head_dim,
        theta_base=rope_theta,
        context_length=context_len,
    )

    # Dummy query and key tensors
    # 中文说明：构造随机的 query / key 张量，用于测试 RoPE 的旋转变换效果。
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary embeddings with our implementation
    # 中文说明：用自实现的 apply_rope 对 query/key 做旋转位置编码变换。
    queries_rot = apply_rope(queries, cos, sin)
    keys_rot = apply_rope(keys, cos, sin)

    # Generate reference RoPE via HF
    # 中文说明：构造一个模拟的配置类，以适配 HuggingFace Qwen3RotaryEmbedding 的构造参数要求
    # （因为 transformers 的 RoPE 初始化映射表中并没有专门为 "qwen3" 注册类型，需要手动兼容）。
    class RoPEConfig:
        # Transformers' RoPE init map does not include "qwen3".
        rope_type = "default"
        factor = 1.0
        dim: int = head_dim
        rope_theta = 1_000_000
        max_position_embeddings = context_len
        hidden_size = head_dim * num_heads
        num_attention_heads = num_heads

        def __init__(self):
            # Transformers >=5.0.0 expects `rope_parameters` on the instance.
            # 中文说明：兼容 transformers >= 5.0.0 版本要求实例上必须存在 rope_parameters 属性。
            self.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta, "factor": 1.0}

        def standardize_rope_params(self):
            # 中文说明：将 rope_parameters 标准化，补齐缺失的 rope_type / rope_theta 字段，
            # 以兼容不同版本 transformers 对 RoPE 配置格式的要求差异。
            params = dict(getattr(self, "rope_parameters", {}) or {})
            if "rope_type" not in params:
                params["rope_type"] = getattr(self, "rope_type", "default")
            if "rope_theta" not in params:
                params["rope_theta"] = getattr(self, "rope_theta")
            self.rope_parameters = params
            return params

    config = RoPEConfig()

    # 中文说明：用 HuggingFace 官方实现生成参考的 cos/sin，以及旋转后的 query/key，作为“标准答案”。
    rot_emb = Qwen3RotaryEmbedding(config=config)
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    # torch.testing.assert_close(sin, ref_sin.squeeze(0), rtol=1e-5, atol=1e-6)
    # torch.testing.assert_close(cos, ref_cos.squeeze(0), rtol=1e-5, atol=1e-6)

    # torch.testing.assert_close(keys_rot, ref_keys_rot, rtol=1e-5, atol=1e-6)A
    # torch.testing.assert_close(queries_rot, ref_queries_rot, rtol=1e-5, atol=1e-6)

    # 断言：自实现的 sin/cos 表必须与 HF 参考实现完全相等（严格 equal，而非近似）。
    assert torch.equal(sin, ref_sin.squeeze(0))
    assert torch.equal(cos, ref_cos.squeeze(0))

    # 断言：自实现对 key/query 做旋转变换后的结果，必须与 HF 参考实现完全相等。
    assert torch.equal(keys_rot, ref_keys_rot)
    assert torch.equal(queries_rot, ref_queries_rot)


@pytest.fixture(scope="session")
def qwen3_weights_path(tmp_path_factory):
    """Creates and saves a deterministic model for testing."""
    # 中文说明：该 fixture 的作用域为整个测试会话（session），意味着它只会被执行一次，
    # 生成的权重文件会被后续多个测试用例（如 test_model_variants）共享复用，
    # 从而避免每个测试都重新初始化一次完整的 Qwen3-0.6B 规模模型（节省时间）。
    # 使用固定随机种子（123）保证权重是确定性、可复现的。
    path = tmp_path_factory.mktemp("models") / "qwen3_test_weights.pt"

    if not path.exists():
        torch.manual_seed(123)
        model = Qwen3Model(QWEN_CONFIG_06_B)
        torch.save(model.state_dict(), path)

    return path


@pytest.mark.parametrize("ModelClass", [Qwen3Model, Qwen3ModelKV])
@pytest.mark.parametrize("generate_fn", [generate_text_simple])
def test_model_variants(ModelClass, qwen3_weights_path, generate_fn):
    """
    中文说明：使用真实规模的 Qwen3-0.6B 配置（QWEN_CONFIG_06_B）及真实分词器，
    分别用普通 Qwen3Model 与带 KV 缓存的 Qwen3ModelKV 加载相同的（确定性）权重，
    对同一段 prompt 做贪心生成（generate_text_simple），并验证生成出的 token id 序列
    与预先记录好的期望结果（expect）完全一致。
    该测试通过参数化同时覆盖两种模型实现，确保它们在真实生成任务上行为一致。
    """

    torch.manual_seed(123)
    model = ModelClass(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(qwen3_weights_path))
    model.eval()

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path="tokenizer-base.json",
        repo_id="rasbt/qwen3-from-scratch",
        add_generation_prompt=False,
        add_thinking=False
    )

    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids = torch.tensor([input_token_ids])

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", prompt)
    print("Encoded input text:", input_token_ids)
    print("encoded_tensor.shape:", input_token_ids.shape)

    # 中文说明：调用生成函数，贪心解码新增 5 个 token。
    out = generate_fn(
        model=model,
        idx=input_token_ids,
        max_new_tokens=5,
        context_size=QWEN_CONFIG_06_B["context_length"]
    )
    print("Encoded output text:", out)
    # 中文说明：这是提前记录好的、在固定随机种子与固定权重下应当生成的期望 token id 序列（黄金标准）。
    expect = torch.tensor([
        [151644, 872, 198, 35127, 752, 264, 2805, 16800, 311,
         3460, 4128,  4119, 13, 151645, 198, 112120, 83942, 60483,
         102652, 7414]
    ])
    # 断言：实际生成结果必须与期望结果逐 token 完全一致。
    assert torch.equal(expect, out)


def test_model_KV_noKV():
    """
    中文说明：在一个自定义的小规模配置下（基于 QWEN_CONFIG_06_B 修改若干超参数），
    分别用带 KV 缓存的模型（Qwen3ModelKV + generate_text_simple_cached）
    与不带 KV 缓存的普通模型（Qwen3Model + generate_text_simple）对同一 prompt 做生成，
    验证两者生成出的 token 序列完全一致，从而确认 KV 缓存机制不会改变生成结果。
    """
    cfg = QWEN_CONFIG_06_B.copy()
    cfg.update({
        "n_layers": 2,
        "emb_dim": 64,
        "hidden_dim": 128,
        "n_heads": 4,
        "n_kv_groups": 2,
        "head_dim": 16,
        "dtype": torch.float32,
    })

    torch.manual_seed(123)
    model_KV = Qwen3ModelKV(cfg)
    model_KV.eval()

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path="tokenizer-base.json",
        repo_id="rasbt/qwen3-from-scratch",
        add_generation_prompt=False,
        add_thinking=False
    )

    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids = torch.tensor([input_token_ids])

    # 中文说明：使用带 KV 缓存的生成函数完成推理。
    out_KV = generate_text_simple_cached(
        model=model_KV,
        idx=input_token_ids,
        max_new_tokens=5,
        context_size=cfg["context_length"]
    )
    del model_KV

    # 中文说明：重新设置相同的随机种子，构建不带 KV 缓存的普通模型，保证权重初始化一致。
    torch.manual_seed(123)
    model_noKV = Qwen3Model(cfg)
    model_noKV.eval()

    out_noKV = generate_text_simple(
        model=model_noKV,
        idx=input_token_ids,
        max_new_tokens=5,
        context_size=cfg["context_length"]
    )

    # 断言：两种推理方式（有/无 KV 缓存）生成的 token 序列必须完全相同。
    assert torch.equal(out_noKV, out_KV)


def test_model_batched_KV():
    """
    中文说明：验证支持批量（batch）推理的 KV 缓存模型（Qwen3ModelKVBatched）
    在两种场景下均能得到正确结果：
    1）batch size = 1 时，其生成结果应与单条 KV 缓存模型（Qwen3ModelKV）的结果完全一致；
    2）batch size = 2（两条相同 prompt，含 padding 对齐）时，批量生成结果中每一条
       都应与单条推理的结果一致，验证批处理与 padding 机制没有引入数值污染或串扰。
    """
    cfg = QWEN_CONFIG_06_B.copy()
    cfg.update({
        "n_layers": 2,
        "emb_dim": 64,
        "hidden_dim": 128,
        "n_heads": 4,
        "n_kv_groups": 2,
        "head_dim": 16,
        "dtype": torch.float32,
    })

    torch.manual_seed(123)
    model_KV = Qwen3ModelKV(cfg)
    model_KV.eval()

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path="tokenizer-base.json",
        repo_id="rasbt/qwen3-from-scratch",
        add_generation_prompt=False,
        add_thinking=False
    )

    # Batch size 1

    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)
    input_token_ids = torch.tensor([input_token_ids])

    # 中文说明：先用非批量的 Qwen3ModelKV 生成基准结果（batch size 恒为 1）。
    out_KV = generate_text_simple_cached(
        model=model_KV,
        idx=input_token_ids,
        max_new_tokens=5,
        context_size=cfg["context_length"]
    )
    del model_KV

    # 中文说明：使用相同随机种子重新构建支持批量推理的模型版本 Qwen3ModelKVBatched。
    torch.manual_seed(123)
    model_KV_batched = Qwen3ModelKVBatched(cfg)
    model_KV_batched.eval()

    # 中文说明：先用 batch size = 1 的输入跑一遍批量推理接口，验证其与非批量版本结果一致。
    out_KV_bs_1 = generate_text_simple_batched(
        model=model_KV_batched,
        idx=input_token_ids,
        max_new_tokens=5,
        context_size=cfg["context_length"]
    )

    # 断言：批量接口在 batch size=1 时的输出应与非批量接口完全一致。
    assert torch.equal(out_KV, out_KV_bs_1)

    # Batch size 2
    # 中文说明：构造两条完全相同的 prompt，组成 batch size = 2 的输入，
    # 并对分词后长度不同的情况做右侧 padding 对齐（此处两条相同故长度也相同，但仍走通用 padding 逻辑）。
    prompts = [
        "Give me a short introduction to large language models.",
        "Give me a short introduction to large language models."
    ]
    tokenized_prompts = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(t) for t in tokenized_prompts)
    padded_token_ids = [
        t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
    ]
    input_tensor = torch.tensor(padded_token_ids)
    out_KV_bs_2 = generate_text_simple_batched(
        model=model_KV_batched,
        idx=input_tensor,
        max_new_tokens=5,
        context_size=cfg["context_length"],
    )
    # 断言：batch size=2 时批内第一条样本的生成结果，应与 batch size=1 时的结果完全一致，
    # 证明批处理（含 padding/attention mask 处理）没有引入样本间的相互干扰。
    assert torch.equal(out_KV.squeeze(0), out_KV_bs_2[0]), (out_KV.squeeze(0).shape, out_KV_bs_2[0].shape)


def test_rmsnorm_equivalence():
    """
    中文说明：验证本项目自实现的 RMSNorm 与从 HuggingFace 搬运过来的参考实现
    Qwen3RMSNorm，在相同随机输入上产生的输出数值应高度一致（在给定容差范围内）。
    """
    torch.manual_seed(42)

    hidden_size = 64
    batch_size = 8
    seq_len = 16

    rms_norm = RMSNorm(hidden_size)
    ref_norm = Qwen3RMSNorm(hidden_size)

    # Sync weights
    # 中文说明：这里将 ref_norm 的权重复制给自身（保持默认的全 1 初始化），
    # 目的是确保两个归一化模块使用相同（默认）的权重值，从而对比结果只反映算法本身的差异。
    with torch.no_grad():
        ref_norm.weight.copy_(ref_norm.weight)

    x = torch.randn(batch_size, seq_len, hidden_size)

    out1 = rms_norm(x)
    out2 = ref_norm(x)

    # 断言：两种 RMSNorm 实现在相同输入下的输出应在给定容差内近似相等。
    torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
@pytest.mark.parametrize("repo_id, tok_file", [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B/tokenizer.json"),  # Chat / Reasoning
    ("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base/tokenizer.json"),  # Base
])
def test_all_special_tokens_roundtrip(repo_id, tok_file):
    """
    中文说明：针对 Chat/Reasoning 版本与 Base 版本两种 Qwen3-0.6B 分词器配置，
    验证本项目 Qwen3Tokenizer 中所有可用的特殊 token（special tokens）：
    1）都能正确编码为单一、有效的 token id，并能反向解码回原始字符串（round-trip）；
    2）嵌入到普通文本中间时，编码/解码后仍能保持原有的文本边界（不会被拆散或吞掉周围文本）；
    3）EOS / PAD 特殊 token 与 HuggingFace 官方分词器的定义完全一致；
    4）仅 Chat/Reasoning 模型应包含 <think> / </think> 思考标记，Base 模型不应包含。
    """
    from transformers import AutoTokenizer as HFTokenizer
    hf_tok = HFTokenizer.from_pretrained(repo_id)

    qt = Qwen3Tokenizer(
        tokenizer_file_path=tok_file,
        repo_id=repo_id,
        add_generation_prompt=False,
        add_thinking=False,
    )

    # Use the instance's actually-available specials
    # 中文说明：只使用该分词器实例上实际可用的特殊 token 集合（不同变体可能有所差异）。
    active_specials = list(qt._special_to_id.keys())

    # Every available special has a concrete id and round-trips
    # 中文说明：逐一验证每个特殊 token：其 id 必须是合法非负整数；编码该 token 字符串
    # 应得到仅包含其自身 id 的单元素列表；将该 id 解码回来应得到原字符串本身。
    for sp, sp_id in qt._special_to_id.items():
        assert isinstance(sp_id, int) and sp_id >= 0, f"{sp} missing or invalid id"
        assert qt.encode(sp) == [sp_id], f"{sp} must encode to its single id"
        assert qt.decode([sp_id]) == sp, f"{sp} must decode back to itself"

    # Inline use preserves boundaries for available specials
    # 中文说明：验证特殊 token 嵌入普通文本中间时（如 "hello <token> world"），
    # 编码结果中必须包含该 token 对应的 id，并且解码回来的文本要与原文本完全一致，
    # 说明特殊 token 的边界被正确识别，没有污染前后的普通文本。
    for sp in active_specials:
        s = f"hello {sp} world"
        ids = qt.encode(s, chat_wrapped=False)
        sp_id = qt._special_to_id[sp]
        assert sp_id in ids, f"{sp} id not found inline"
        assert qt.decode(ids) == s, f"Inline decode mismatch for {sp}"

    # EOS / PAD expectations
    # 中文说明：Base 模型与 Chat/Reasoning 模型的 EOS token 不同
    # （Base 用 <|endoftext|>，Chat 用 <|im_end|>），PAD token 统一为 <|endoftext|>。
    is_base = ("Base" in repo_id)
    expected_eos = "<|endoftext|>" if is_base else "<|im_end|>"
    expected_pad = "<|endoftext|>"

    # 断言：自实现分词器与 HuggingFace 官方分词器在 EOS/PAD id 及其解码文本上应完全一致。
    assert qt.decode([qt.eos_token_id]) == expected_eos
    assert qt.decode([qt.pad_token_id]) == expected_pad
    assert hf_tok.eos_token_id == qt.eos_token_id
    assert hf_tok.pad_token_id == qt.pad_token_id
    assert hf_tok.decode([hf_tok.eos_token_id], skip_special_tokens=False) == expected_eos
    assert hf_tok.decode([hf_tok.pad_token_id], skip_special_tokens=False) == expected_pad

    # Thinking tokens only on chat models
    # 中文说明：<think> / </think> 思考标记仅应出现在 Chat/Reasoning 模型中，
    # 其对应的固定 id（151667 / 151668）应可验证；Base 模型则完全不应包含这两个标记。
    if not is_base:
        assert qt._tok.token_to_id("<think>") == 151667
        assert qt._tok.token_to_id("</think>") == 151668
        assert qt.encode("<think>") == [151667]
        assert qt.encode("</think>") == [151668]
    else:
        assert "<think>" not in active_specials and "</think>" not in active_specials


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
@pytest.mark.parametrize("add_gen, add_think", [(True, True), (True, False), (False, False)])
def test_chat_wrap_and_equivalence(add_gen, add_think):
    """
    中文说明：验证本项目 Qwen3Tokenizer 的 encode 方法在自动应用聊天模板（chat template）
    包装单轮用户消息时，与 HuggingFace AutoTokenizer.apply_chat_template 的编码结果一致；
    同时验证解码结果、EOS/PAD id 也保持一致。
    该测试通过 add_generation_prompt / add_thinking 两个开关的三种组合分别验证，
    并同时覆盖 Chat 版本与 Base 版本两个仓库的分词器。
    注：(add_gen=True, add_think=False) 是刻意跳过精确比较的“边界情况”
    （实际使用中不会采用该组合），仅做提示，不做强断言。
    """
    from transformers import AutoTokenizer

    prompt = "Give me a short introduction to large language models."
    messages = [{"role": "user", "content": prompt}]

    for repo_id, tok_file in [
        ("Qwen/Qwen3-0.6B", "Qwen3-0.6B/tokenizer.json"),
        ("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base/tokenizer.json"),
    ]:
        hf_tok = AutoTokenizer.from_pretrained(repo_id)
        qt = Qwen3Tokenizer(
            tokenizer_file_path=tok_file,
            repo_id=repo_id,
            add_generation_prompt=add_gen,
            add_thinking=add_think,
        )

        # Our encode vs HF template
        # 中文说明：分别用自实现分词器与 HF 官方 chat template 对同一 prompt 编码，用于比较。
        ours = qt.encode(prompt)
        ref = _hf_ids(hf_tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_gen,
            enable_thinking=add_think,
        ))

        if add_gen and not add_think:
            pass  # skip edge case as this is not something we use in practice
        else:
            # 断言：在非边界情况下，自实现编码结果必须与 HF 官方模板编码结果完全一致。
            assert ours == ref, (repo_id, add_gen, add_think)

        # Round-trip decode equality
        # 中文说明：非边界情况下，验证解码结果也应完全一致。
        if not (add_gen and not add_think):
            assert qt.decode(ours) == hf_tok.decode(ref)

        # EOS/PAD parity
        # 断言：EOS / PAD 的 token id 在自实现分词器与 HF 分词器之间应保持一致。
        assert qt.eos_token_id == hf_tok.eos_token_id
        assert qt.pad_token_id == hf_tok.pad_token_id


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
@pytest.mark.parametrize("repo_id, tok_file", [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B/tokenizer.json"),
    ("Qwen/Qwen3-0.6B-Base", "Qwen3-0.6B-Base/tokenizer.json"),
])
@pytest.mark.parametrize("add_gen, add_think", [
    (True, True),
    (False, False),
])
def test_multiturn_equivalence(repo_id, tok_file, add_gen, add_think):
    """
    中文说明：验证在多轮对话（包含 system / user / assistant / user 四条消息）场景下，
    本项目 Qwen3Tokenizer 对 HuggingFace 官方 chat template 渲染出的原始文本
    （而非直接对消息列表编码）进行编码后，得到的 token id 序列与解码文本
    均应与 HF 官方 tokenize=True 的结果完全一致。
    该测试同时覆盖 Chat/Base 两种仓库，以及 (add_gen, add_think) 的两种开关组合。
    """
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(repo_id)
    qt = Qwen3Tokenizer(
        tokenizer_file_path=tok_file,
        repo_id=repo_id,
        add_generation_prompt=add_gen,
        add_thinking=add_think,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize transformers in one sentence."},
        {"role": "assistant", "content": "Transformers use attention to model long-range dependencies efficiently."},
        {"role": "user", "content": "Now add one concrete example."},
    ]

    # HF reference (ids and raw template text)
    # 中文说明：分别获取 HF 官方对多轮对话消息列表的 token id 编码结果，以及渲染出的原始模板文本。
    ref_ids = hf_tok.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=add_gen, enable_thinking=add_think
    )
    ref_ids = _hf_ids(ref_ids)
    ref_text = hf_tok.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=add_gen, enable_thinking=add_think
    )

    # Our encode over HF's raw template text
    # 中文说明：让本项目分词器直接对 HF 渲染出的原始文本进行编码（chat_wrapped=False 表示
    # 不再重复套用聊天模板，因为文本本身已经是模板渲染结果）。
    ours_ids = qt.encode(ref_text, chat_wrapped=False)

    # 断言：在相同的原始模板文本上，自实现编码结果应与 HF 官方编码结果完全一致。
    assert ours_ids == ref_ids, f"mismatch for ({repo_id}, add_gen={add_gen}, add_think={add_think})"

    # Round-trip decode equality
    # 断言：解码结果也应完全一致。
    ours_dec = qt.decode(ours_ids)
    ref_dec = hf_tok.decode(ref_ids, skip_special_tokens=False)
    assert ours_dec == ref_dec


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_tokenizer_equivalence():
    """
    中文说明：这是一个较为全面的分词器等价性测试，会实际从 HuggingFace Hub 下载
    Qwen3-0.6B（Chat）与 Qwen3-0.6B-Base 两个仓库的 tokenizer.json 文件，
    并在“是否应用聊天模板”（apply_chat_template）与
    “是否添加生成提示/思考模式”（add_generation_prompt, add_thinking）的
    多种组合下，逐一验证：
    1）编码得到的 token id 序列与 HF 官方结果一致；
    2）解码得到的文本与 HF 官方结果一致；
    3）关键特殊 token（<|endoftext|>、<|im_end|>）编码正确；
    4）EOS / PAD token 的解码文本符合预期（区分 Base 与 Chat 版本）。
    由于涉及网络下载，该测试仅在安装了 transformers 时运行，且运行较慢。
    """
    from transformers import AutoTokenizer

    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt},
    ]

    for apply_chat_template in (True, False):
        for s in ("-Base", ""):
            repo_id = f"Qwen/Qwen3-0.6B{s}"
            tokenizer_ref = AutoTokenizer.from_pretrained(repo_id)
            tokenizer_url = f"https://huggingface.co/Qwen/Qwen3-0.6B{s}/resolve/main/tokenizer.json"
            # 中文说明：从 HuggingFace Hub 下载对应仓库的 tokenizer.json 到当前目录。
            download_file(tokenizer_url, out_dir=".")

            old_name = "tokenizer.json"

            if not s:
                new_name = "tokenizer-reasoning.json"
            else:
                new_name = "tokenizer-base.json"

            # 中文说明：将下载下来的通用文件名 tokenizer.json 重命名为区分 Base/Reasoning 的文件名，
            # 以便同一目录下可以同时保留两个仓库的分词器文件而不互相覆盖。
            # 若 shutil.move 因跨设备等原因失败，则退化为“复制到临时文件再原子替换”的方式。
            try:
                shutil.move(old_name, new_name)
            except Exception:
                with tempfile.NamedTemporaryFile(delete=False, dir=".") as tmp_file:
                    shutil.copyfile(old_name, tmp_file.name)
                    os.replace(tmp_file.name, new_name)
                os.remove(old_name)

            for states in ((True, True), (False, False)):
                tokenizer = Qwen3Tokenizer(
                    tokenizer_file_path=new_name,
                    repo_id=repo_id,
                    apply_chat_template=apply_chat_template,
                    add_generation_prompt=states[0],
                    add_thinking=states[1]
                )
                input_token_ids = tokenizer.encode(prompt)

                if apply_chat_template:
                    # 中文说明：若启用聊天模板，则与 HF 官方 apply_chat_template 的编码结果比较。
                    input_token_ids_ref = tokenizer_ref.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=states[0],
                        enable_thinking=states[1],
                    )
                    input_token_ids_ref = _hf_ids(input_token_ids_ref)
                else:
                    # 中文说明：若不启用聊天模板，则参考结果就是自身编码结果（无对照，仅走通用断言逻辑）。
                    input_token_ids_ref = input_token_ids

                # 断言：编码 token id 序列必须一致。
                assert input_token_ids == input_token_ids_ref, states

                output_text = tokenizer.decode(input_token_ids)
                out_text_ref = tokenizer_ref.decode(input_token_ids_ref)
                # 断言：解码文本必须一致。
                assert output_text == out_text_ref, states

                # 断言：关键特殊 token 的编码必须正确对应其内部 id 映射表中的值。
                assert tokenizer.encode("<|endoftext|>") == [tokenizer._special_to_id["<|endoftext|>"]]
                assert tokenizer.encode("<|im_end|>") == [tokenizer._special_to_id["<|im_end|>"]]

                expected_eos_token = "<|im_end|>" if "base" not in new_name else "<|endoftext|>"
                expected_pad_token = "<|endoftext|>"
                # 断言：EOS / PAD token 的解码文本应符合预期（依据文件名判断是否为 base 版本）。
                assert tokenizer.decode([tokenizer.eos_token_id]) == expected_eos_token
                assert tokenizer.decode([tokenizer.pad_token_id]) == expected_pad_token


@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
@pytest.mark.parametrize("repo_id, tok_file", [
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B/tokenizer.json"),
])
@pytest.mark.parametrize("add_gen, add_think", [
    (True, True),
    (False, False),
])
def test_multiturn_prefix_stability(repo_id, tok_file, add_gen, add_think):
    """
    中文说明：验证在多轮对话逐步增长（每次新增一条消息）的场景下，本项目分词器
    对 HF 官方模板渲染文本的编码结果满足以下三条性质：
    1）每一阶段的完整编码结果都应与 HF 官方编码结果完全一致；
    2）“前缀稳定性”：若当前阶段 HF 渲染出的文本仍然是上一阶段文本的前缀延伸
       （即历史文本没有被 HF 模板逻辑改写，例如插入 <think> 标记等特殊情况），
       那么当前阶段编码结果的前缀部分应与上一阶段编码结果完全相同
       （这对增量式对话缓存、KV 缓存复用等场景非常关键）；
    3）每一阶段的解码结果都应与 HF 官方解码结果一致。
    当某一阶段的编码结果与参考不一致时，测试会打印详细的逐 token 差异定位信息，方便调试。
    """
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(repo_id)
    qt = Qwen3Tokenizer(
        tokenizer_file_path=tok_file,
        repo_id=repo_id,
        add_generation_prompt=add_gen,
        add_thinking=add_think,
    )

    turns = [
        [{"role": "user", "content": "Define perplexity briefly."}],
        [{"role": "assistant", "content": "A measure of how well a language model predicts a sample."}],
        [{"role": "user", "content": "And why lower is better?"}],
    ]

    prev_ids_qt, prev_ids_hf = None, None
    prev_ref_text = None
    running = []  # grows turn-by-turn
    # 中文说明：running 列表模拟对话历史的逐轮累积，每次循环新增一轮对话内容。

    for delta in turns:
        running += delta

        ref_ids = hf_tok.apply_chat_template(
            running, tokenize=True,
            add_generation_prompt=add_gen, enable_thinking=add_think
        )
        ref_ids = _hf_ids(ref_ids)
        ref_text = hf_tok.apply_chat_template(
            running, tokenize=False,
            add_generation_prompt=add_gen, enable_thinking=add_think
        )

        # Normalize line endings to match our encoder's assumptions
        # 中文说明：统一换行符为 \n，以匹配本项目编码器对换行符格式的假设，避免因 \r\n 差异导致误判。
        ref_text_norm = ref_text.replace("\r\n", "\n").replace("\r", "\n")

        # Our encode over HF's raw template text
        # 中文说明：对归一化后的 HF 模板文本直接编码（不再重复应用聊天模板）。
        ours_ids = qt.encode(ref_text_norm, chat_wrapped=False)

        # 1) Exact equality per stage
        # 中文说明：若当前阶段编码结果与 HF 参考不一致，则逐 token 定位第一个不同的位置，
        # 并打印该位置附近的 id、token 文本及解码结果，便于快速排查差异原因。
        if ours_ids != ref_ids:
            # Lightweight inline diff to aid debugging
            from itertools import zip_longest
            for i, (a, b) in enumerate(zip_longest(ours_ids, ref_ids, fillvalue=None)):
                if a != b:
                    slice_lo, slice_hi = max(0, i-6), i+6
                    ours_slice = ours_ids[slice_lo:slice_hi]
                    ref_slice = ref_ids[slice_lo:slice_hi]
                    ours_toks = [qt._tok.id_to_token(x) if x is not None else None for x in ours_slice]
                    ref_toks = hf_tok.convert_ids_to_tokens(ref_slice, skip_special_tokens=False)
                    raise AssertionError(
                        f"Stage mismatch for ({repo_id}, add_gen={add_gen}, add_think={add_think}) at index {i}\n"
                        f"OURS ids: {ours_slice}\nREF  ids: {ref_slice}\n"
                        f"OURS tok: {ours_toks}\nREF  tok: {ref_toks}\n"
                        f"OURS dec: {qt.decode(ours_slice)}\nREF  dec: {hf_tok.decode(ref_slice, skip_special_tokens=False)}"
                    )
        # If no raise, they match
        # 断言：（若上面未提前抛出更详细的错误信息）当前阶段编码结果必须与 HF 参考完全一致。
        assert ours_ids == ref_ids

        # 2) Prefix stability only when HF's own *text* remained a prefix
        # 中文说明：只有当 HF 本轮渲染出的文本确实是上一轮文本的前缀延伸时，
        # 才检验编码 id 序列的前缀部分是否保持稳定；否则说明 HF 模板逻辑本身
        # 改写了历史部分（例如动态插入/移除 <think> 标记），此时跳过前缀检验是合理的。
        if prev_ids_hf is not None and prev_ref_text is not None:
            if ref_text.startswith(prev_ref_text):
                assert ours_ids[:len(prev_ids_qt)] == prev_ids_qt
                assert ref_ids[:len(prev_ids_hf)] == prev_ids_hf
            # else: HF modified earlier boundaries (e.g., inserted <think>), so skip prefix checks

        # 3) Decode parity at each step
        # 断言：每一轮的解码结果都必须与 HF 官方解码结果一致。
        assert qt.decode(ours_ids) == hf_tok.decode(ref_ids, skip_special_tokens=False)

        prev_ids_qt, prev_ids_hf = ours_ids, ref_ids
        prev_ref_text = ref_text


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_qwen3_base_equivalence_with_transformers():
    """
    中文说明：这是端到端的“最终对照测试”——构造一个极小规模但结构完整
    （含分组查询注意力 n_kv_groups、QK 归一化 qk_norm 等特性）的配置，
    分别用本项目的 Qwen3Model 与 HuggingFace 官方 Qwen3ForCausalLM 构建模型，
    通过 load_weights_into_qwen 把 HF 模型的权重手动映射并加载进自实现模型，
    最后在相同的随机输入上比较两者输出的 logits 是否数值一致。
    这是验证本项目 Qwen3 架构实现（包括权重命名映射关系）是否与官方实现完全对应的关键测试。
    """

    from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    # Tiny config so the test is fast
    # 中文说明：使用极小配置（词表 257、上下文长度 8 等）以保证测试运行速度快。
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        "qk_norm": True,
        "n_kv_groups": 2,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,
    }
    model = Qwen3Model(cfg)

    # 中文说明：将本项目配置字典中的各字段映射为 HuggingFace Qwen3Config 所需的对应参数名。
    hf_cfg = Qwen3Config(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_groups"],
        rope_theta=cfg["rope_base"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    hf_model = Qwen3ForCausalLM(hf_cfg)

    # 中文说明：取出 HF 模型的随机初始化权重字典，通过本项目提供的
    # load_weights_into_qwen 工具函数，将其按命名规则映射并加载进自实现的 Qwen3Model 中，
    # 从而保证两个模型使用完全相同的参数值，比较结果才有意义。
    hf_state = hf_model.state_dict()
    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    load_weights_into_qwen(model, param_config, hf_state)

    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    # 断言：在完全相同的权重和输入下，自实现模型与 HuggingFace 官方模型的输出 logits
    # 应在给定容差范围内数值一致，这是对整体架构实现正确性的最终验证。
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
