# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块用于验证 Qwen3（带 KV cache 版本）notebook 中定义的模型实现是否正确。

主要做两件事：
1. 通过 `import_definitions_from_notebook` 工具函数，从
   `standalone-qwen3-plus-kvcache.ipynb` 这个 notebook 文件中动态导入
   其中定义的类和函数（例如 `Qwen3Model`、`load_weights_into_qwen`），
   使得可以像普通 Python 模块一样在 pytest 中对其进行单元测试。
2. 提供两个测试用例：
   - 使用一个很小的随机配置，检查模型前向传播输出的张量形状是否符合预期；
   - 在安装了 `transformers` 库的前提下，将本 notebook 中手写实现的 Qwen3
     模型与 HuggingFace 官方 `transformers` 库中的 `Qwen3ForCausalLM`
     进行数值等价性比较，确保自实现的注意力、RoPE、归一化等逻辑与官方
     实现完全一致。
"""

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


# 检测当前环境中是否安装了 transformers 库，用于后面跳过依赖该库的测试
transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def import_notebook_defs():
    # 定位到 notebook 所在目录（当前测试文件的上两级目录）
    nb_dir = Path(__file__).resolve().parents[1]
    # 从 notebook 文件中动态导入其中定义的类/函数，返回一个类似模块的对象
    mod = import_definitions_from_notebook(nb_dir, "standalone-qwen3-plus-kvcache.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    # 固定随机种子，保证测试结果可复现
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    # 一个尽量精简的基础模型配置，仅用于快速跑通前向传播，不追求真实规模
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
        "rope_base": 10000,
        "context_length": 64,
        "num_experts": 0,
    }


@pytest.fixture
def dummy_cfg_moe(dummy_cfg_base):
    # 在基础配置之上，追加 MoE（混合专家）相关配置，得到 MoE 版本的配置
    cfg = dummy_cfg_base.copy()
    cfg.update({
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
    })
    return cfg


@torch.inference_mode()
def test_dummy_qwen3_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    """使用极小的随机配置构建 Qwen3Model，验证模型前向传播能够正常运行，
    并且输出张量的形状符合预期的 (batch_size, seq_len, vocab_size)。"""
    torch.manual_seed(123)
    model = import_notebook_defs.Qwen3Model(dummy_cfg_base)
    out = model(dummy_input)
    # 校验输出形状：批大小为1、序列长度与输入一致、最后一维为词表大小
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_qwen3_base_equivalence_with_transformers(import_notebook_defs):
    """在未安装 transformers 时跳过本测试；已安装时，构建一个微型配置，
    分别用 notebook 中自实现的 Qwen3Model 和 HuggingFace 官方的
    Qwen3ForCausalLM 初始化模型，将官方模型的权重加载进自实现模型后，
    对相同输入比较两者输出的 logits 是否在数值上足够接近，用以验证
    自实现模型（包括注意力、RoPE、归一化等模块）与官方实现等价。"""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    # Tiny config so the test is fast
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
        "rope_local_base": 10_000.0,
        "sliding_window": 4,
        "layer_types": ["full_attention", "full_attention"],
        "dtype": torch.float32,
        "query_pre_attn_scalar": 256,
    }
    # 使用 notebook 中自实现的 Qwen3Model 构建模型
    model = import_notebook_defs.Qwen3Model(cfg)

    # 使用相同的超参数构建 HuggingFace 官方的 Qwen3Config
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
        rope_local_base_freq=cfg["rope_local_base"],
        layer_types=cfg["layer_types"],
        sliding_window=cfg["sliding_window"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
        rope_scaling={"rope_type": "default"},
    )
    # 用官方配置初始化官方模型（权重为随机初始化）
    hf_model = Qwen3ForCausalLM(hf_cfg)

    # 取出官方模型的权重字典，稍后加载到自实现模型中，保证两者权重完全一致
    hf_state = hf_model.state_dict()
    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    import_notebook_defs.load_weights_into_qwen(model, param_config, hf_state)

    # 构造相同的随机输入，分别喂给两个模型
    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    # 核心断言：在加载了相同权重的前提下，自实现模型与官方模型的输出 logits
    # 应在给定的相对/绝对误差范围内一致，从而验证自实现的正确性
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
