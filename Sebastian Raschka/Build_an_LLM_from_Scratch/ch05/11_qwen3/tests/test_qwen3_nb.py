# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块:针对 Qwen3 教学 notebook(standalone-qwen3.ipynb)中定义的模型代码进行单元测试。

主要验证内容:
1. 使用一个很小的“dummy”配置构建 Qwen3Model,检查前向传播输出的张量形状是否符合预期
   (batch_size, seq_len, vocab_size)。
2. 在安装了 `transformers` 库的前提下,将本仓库手写实现的 Qwen3Model 与 HuggingFace
   官方的 Qwen3ForCausalLM 在相同(随机初始化后被覆盖为一致)的权重下进行数值等价性对比,
   确保自实现的注意力机制、RoPE、滑动窗口等逻辑与官方实现保持一致。

测试中使用 `import_definitions_from_notebook` 工具函数,直接从 notebook 文件中动态导入
类和函数定义(而不是从 .py 模块导入),这样可以保证测试始终针对最新的 notebook 内容。
"""

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


# 检测当前环境中是否安装了 transformers 库;未安装时,依赖它的等价性测试会被跳过
transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def import_notebook_defs():
    # 定位到 notebook 所在目录(当前测试文件的上两级目录),并动态导入其中的类/函数定义
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-qwen3.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    # 固定随机种子以保证测试可复现
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    # 一个体积很小的基础配置,仅用于快速跑通前向传播,不代表真实 Qwen3 的规模
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
    # 在基础配置之上,追加混合专家(MoE)相关的配置项,得到 MoE 版本的配置
    cfg = dummy_cfg_base.copy()
    cfg.update({
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
    })
    return cfg


@torch.inference_mode()
def test_dummy_qwen3_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    """使用小型 dummy 配置构建 Qwen3Model,验证前向传播输出张量的形状是否符合预期。"""
    torch.manual_seed(123)
    model = import_notebook_defs.Qwen3Model(dummy_cfg_base)
    out = model(dummy_input)
    # 断言输出形状为 (batch_size, seq_len, vocab_size),即模型对每个位置都输出词表大小的 logits
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_qwen3_base_equivalence_with_transformers(import_notebook_defs):
    """
    将本仓库手写的 Qwen3Model 与 HuggingFace transformers 官方的 Qwen3ForCausalLM
    进行数值等价性验证:构建结构相同的小型配置,把 HF 模型的随机初始化权重加载到
    自实现模型中,再用相同的输入分别做前向传播,比较两者输出的 logits 是否足够接近。
    若未安装 transformers,则跳过本测试。
    """
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
    # 构建本仓库手写实现的 Qwen3 模型
    model = import_notebook_defs.Qwen3Model(cfg)

    # 构建结构参数一致的 HuggingFace Qwen3 配置,并据此实例化官方模型
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
    hf_model = Qwen3ForCausalLM(hf_cfg)

    # 取出 HF 官方模型的随机初始化权重,并按照本仓库的权重加载逻辑灌入自实现模型,
    # 从而保证两个模型在相同权重下比较,排除初始化差异带来的干扰
    hf_state = hf_model.state_dict()
    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    import_notebook_defs.load_weights_into_qwen(model, param_config, hf_state)

    # 构造相同的随机输入 token 序列,分别送入两个模型
    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    # 断言两者输出的 logits 在给定的相对/绝对误差范围内一致,验证自实现的正确性
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
