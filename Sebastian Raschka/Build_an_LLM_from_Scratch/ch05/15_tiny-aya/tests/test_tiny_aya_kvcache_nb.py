# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是针对 `standalone-tiny-aya-plus-kv-cache.ipynb` 这个 Jupyter Notebook
中定义的 TinyAya（Cohere2 架构简化实现，支持 KV 缓存）模型所写的 pytest 测试
文件。

它做了两件事：
1. `test_dummy_tiny_aya_forward`：
   用一个体积很小、随意构造的配置（vocab_size/emb_dim/层数等都很小）构造
   TinyAyaModel，跑一次前向传播，检查输出张量的形状是否符合预期
   （batch_size, seq_len, vocab_size）。这是一个“冒烟测试”（smoke test），
   只验证模型结构能跑通、输出形状正确，不验证数值精度。

2. `test_tiny_aya_base_equivalence_with_transformers`：
   构造一个参数完全对齐的 HuggingFace `transformers` 版 `Cohere2ForCausalLM`，
   把该 HF 模型随机初始化的权重加载进我们自己实现的 TinyAyaModel
   （通过 `load_weights_into_tiny_aya`），然后比较两者在同一批随机输入上的
   logits 输出是否数值一致（在给定的相对/绝对误差范围内）。这是一个
   “等价性测试”，用来验证本地从零实现的 TinyAya 与官方 Cohere2 实现在数学上
   是等价的。该测试依赖 `transformers` 库，如果环境中没有安装则会被跳过。

被测的模型定义（TinyAyaModel、load_weights_into_tiny_aya 等）并不是直接从
.py 源文件导入的，而是通过 `import_definitions_from_notebook` 这个工具函数
动态执行同目录（上一级目录）下的 `standalone-tiny-aya-plus-kv-cache.ipynb`
notebook 并提取其中定义的类/函数，这样可以保证教学用的 notebook 内容与测试
保持同步。

注意（本次审阅发现的一处风险点，按要求未做修改，仅在此标注并上报）：
    第 15 行 `transformers_installed = importlib.util.find_spec(...)` 只执行
    了 `import importlib`，并没有显式 `import importlib.util`。
    `importlib.util` 是 `importlib` 包下的一个独立子模块，Python 语言规范
    并不保证仅靠 `import importlib` 就一定能通过属性访问到
    `importlib.util`——它当前之所以能正常工作，是因为 pytest 自身的模块
    导入/断言重写机制（以及可能的 torch 等其他依赖）在加载本测试文件之前，
    已经作为副作用导入过 `importlib.util`，从而把它挂载到了 `importlib`
    这个包对象的命名空间下。也就是说，这是一处“隐式依赖导入顺序”的写法：
    在本项目固定通过 pytest 收集执行的场景下，实际观察不到失败，但它并非
    被语言规范保证成立，属于跨环境/跨版本可能存在差异的风险点，而不是一个
    在当前实际调用路径下必然复现的确定性 bug，因此这里选择保持原样、仅作
    标注，不做改动（如需彻底消除该风险，可显式加入 `import importlib.util`）。
"""

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


# 检测当前环境是否安装了 transformers 库，用于下面等价性测试的 skipif 条件。
# （见上方模块 docstring 中的风险说明：这里隐式依赖 importlib.util 已被间接导入。）
transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def import_notebook_defs():
    """Pytest fixture：动态导入同目录（上一级目录）下的
    `standalone-tiny-aya-plus-kv-cache.ipynb` notebook，并返回其中定义的所有
    类/函数所在的模块对象，供各测试用例调用（如 TinyAyaModel、
    load_weights_into_tiny_aya 等）。
    """
    # __file__ 是 tests/test_tiny_aya_kvcache_nb.py，parents[1] 即上溯两级，
    # 定位到该文件所在的 15_tiny-aya 目录（notebook 所在目录）
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-tiny-aya-plus-kv-cache.ipynb")
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
    """验证我们自己实现的 TinyAyaModel 与 HuggingFace 官方
    `Cohere2ForCausalLM` 在数学上是等价的。

    做法：用完全相同的一组超参数分别构造 TinyAyaModel 和 HF 版
    Cohere2ForCausalLM；把 HF 模型（随机初始化）的权重通过
    `load_weights_into_tiny_aya` 搬运进我们自己的模型；再用同一批随机 token
    输入分别跑前向传播，比较两者输出的 logits 是否在给定误差范围内一致。
    如果两者数值一致，说明我们从零实现的注意力、RoPE、滑动窗口、归一化等
    逻辑与官方实现等价。若当前环境未安装 `transformers`，该测试会被跳过。
    """
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

    # 构造我们自己实现的模型（来自 notebook 中定义的 TinyAyaModel 类）
    model = import_notebook_defs.TinyAyaModel(cfg)

    # 构造与上面 cfg 参数一一对应的 HuggingFace Cohere2 配置，
    # 确保两边模型在结构、维度、RoPE 参数、滑动窗口等设置上完全一致，
    # 这样才能做出有意义的数值等价性比较
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

    # 取出 HF 模型（随机初始化）的权重字典，搬运进我们自己的实现里，
    # 这样两边模型的参数值完全一致，输出差异只可能来自实现逻辑本身
    hf_state = hf_model.state_dict()
    import_notebook_defs.load_weights_into_tiny_aya(model, cfg, hf_state)

    # 用同一批随机 token 输入分别跑两边模型的前向传播
    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    # 核心断言：两边 logits 在给定的相对/绝对误差范围内必须一致，
    # 这是验证“自实现与官方实现等价”的关键判据
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
