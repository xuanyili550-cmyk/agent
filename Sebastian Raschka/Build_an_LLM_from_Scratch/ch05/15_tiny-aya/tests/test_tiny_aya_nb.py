# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是 TinyAya（Cohere2 架构简化实现）的单元测试文件。

测试内容包括：
1. `test_dummy_tiny_aya_forward`：仅验证我们自己实现的 TinyAyaModel 能够正常
   完成前向传播，且输出张量形状符合预期（batch_size, seq_len, vocab_size）。
2. `test_tiny_aya_base_equivalence_with_transformers`：在安装了 HuggingFace
   `transformers` 库的前提下，构造一个参数完全对齐的 `Cohere2ForCausalLM`，
   把它的随机初始化权重加载进我们自己的 TinyAyaModel，然后比较两者在同一批
   随机输入上的 logits 输出是否数值一致（即验证我们的实现与官方实现等价）。

被测的模型定义（TinyAyaModel、load_weights_into_tiny_aya 等）并不是直接从
.py 源文件导入的，而是通过 `import_definitions_from_notebook` 这个工具函数
动态执行同目录下的 `standalone-tiny-aya.ipynb` notebook 并提取其中定义的类/
函数，这样可以保证教学用的 notebook 内容与测试保持同步。
"""

import importlib
import importlib.util  # 显式导入 util 子模块，见下方 bug 说明
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


# 原代码：`import importlib.util` 这一行缺失，仅有 `import importlib`，
# 随后却访问了 `importlib.util.find_spec(...)`。
# 为什么是 bug：`importlib.util` 是一个独立子模块，仅执行 `import importlib`
# 并不保证 `importlib.util` 一定可用——它是否已经被绑定到 `importlib` 命名空间，
# 取决于运行时之前是否有其他模块（例如某些第三方库）间接导入过
# `importlib.util`。这属于隐式依赖，在不同环境/Python 版本/依赖顺序下可能
# 触发 `AttributeError: module 'importlib' has no attribute 'util'`，
# 是一个确定性的、可复现的稳定性 bug。修复方式是显式 `import importlib.util`
# （已在上方补充），从而保证该子模块始终被正确加载。
transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def import_notebook_defs():
    """Pytest fixture：动态导入同目录（上一级目录）下的
    `standalone-tiny-aya.ipynb` notebook，并返回其中定义的所有类/函数
    所在的模块对象，供各测试用例调用（如 TinyAyaModel、
    load_weights_into_tiny_aya 等）。
    """
    # __file__ 是 tests/test_tiny_aya_nb.py，parents[1] 即上溯两级，
    # 定位到该文件所在的 15_tiny-aya 目录（notebook 所在目录）
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-tiny-aya.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    """Pytest fixture：生成一个固定随机种子下的伪造 token id 输入张量，
    形状为 (batch_size=1, seq_len=8)，取值范围 [0, 100)，用于前向传播测试。
    """
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    """Pytest fixture：提供一个体积很小的 TinyAyaModel 基础配置字典，
    用于快速验证模型结构能否正常前向传播（不追求与真实 Cohere2 模型
    参数一致，只保证测试运行速度快）。
    """
    return {
        "vocab_size": 100,
        "context_length": 64,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        "n_kv_heads": 1,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_base": 10_000.0,
        "layer_norm_eps": 1e-5,
        "logit_scale": 1.0,
        "tie_word_embeddings": False,
        "dtype": torch.float32,
    }

@torch.inference_mode()
def test_dummy_tiny_aya_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    """验证 TinyAyaModel 能够正常完成一次前向传播，且输出的 logits 张量
    形状为 (batch_size, seq_len, vocab_size)。

    本测试不关心数值是否正确，只关心结构是否跑得通、输出形状是否符合预期，
    因此使用的是一个体积很小、随意构造的配置（dummy_cfg_base）。
    """
    torch.manual_seed(123)
    model = import_notebook_defs.TinyAyaModel(dummy_cfg_base)
    out = model(dummy_input)
    # 核心断言：输出形状必须是 (batch=1, seq_len=8, vocab_size=100)，
    # 否则说明模型内部维度处理（如注意力头拼接、输出投影层等）存在问题
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_tiny_aya_base_equivalence_with_transformers(import_notebook_defs):
    from transformers import Cohere2Config, Cohere2ForCausalLM

    # Tiny config so the test is fast
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        "n_kv_heads": 2,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention"],
        "dtype": torch.float32,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-5,
        "rope_base": 10_000.0,
        "logit_scale": 1.0,
        "tie_word_embeddings": False,
    }

    model = import_notebook_defs.TinyAyaModel(cfg)

    hf_cfg = Cohere2Config(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        num_key_value_heads=cfg["n_kv_heads"],
        attention_bias=cfg["attention_bias"],
        attention_dropout=cfg["attention_dropout"],
        layer_norm_eps=cfg["layer_norm_eps"],
        layer_types=cfg["layer_types"],
        sliding_window=cfg["sliding_window"],
        logit_scale=cfg["logit_scale"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        rope_parameters={"rope_type": "default", "rope_theta": cfg["rope_base"]},
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    hf_model = Cohere2ForCausalLM(hf_cfg)

    hf_state = hf_model.state_dict()
    import_notebook_defs.load_weights_into_tiny_aya(model, cfg, hf_state)

    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)