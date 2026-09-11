# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是针对 `standalone-gemma3.ipynb` 这个 Jupyter Notebook 中定义的
Gemma3 模型实现所写的 pytest 测试文件。

它做了两件事：
1. `test_dummy_gemma3_forward`：
   用一个很小的随机配置（vocab/emb/层数等都很小）构造 Gemma3Model，
   跑一次前向传播，检查输出张量的形状是否符合预期
   （batch_size, seq_len, vocab_size）。这是一个"冒烟测试"（smoke test），
   只验证模型能跑通、形状对，不验证数值精度。

2. `test_gemma3_base_equivalence_with_transformers`：
   在本地实现之上，构造一个参数完全对齐的 HuggingFace `transformers`
   版 Gemma3ForCausalLM，把 HF 模型的权重加载进本地实现，
   然后比较两者在同一批随机输入上的 logits 输出是否数值一致
   （在给定的相对/绝对误差范围内）。这是一个"等价性测试"，
   用来验证本地从零实现的 Gemma3 与官方实现在数学上是等价的。
   该测试依赖 `transformers` 库，如果环境中没有安装则会被跳过。

注意（本次审阅发现的一处风险点，未做修改）：
    第 15 行 `importlib.util.find_spec(...)` 只 `import importlib`，
    并没有显式 `import importlib.util`。`importlib.util` 是一个子模块，
    Python 并不保证仅靠 `import importlib` 就能访问到 `importlib.util`
    属性——它之所以能工作，是因为运行时其他已导入的库（如 pytest/torch
    等）作为副作用提前导入了 `importlib.util`，使其被挂载到
    `importlib` 命名空间下。这属于"依赖导入顺序、跨环境/跨版本不一定
    稳定复现"的隐患，而不是一个确定性 bug（当前环境下可以正常跑通），
    因此按要求不做修改，仅在此标注、并在返回结果中上报。
"""

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


# 检测当前环境是否安装了 transformers 库；用于下面等价性测试的 skipif 条件。
# （见上方模块 docstring 中的风险说明：这里隐式依赖 importlib.util 已被间接导入。）
transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def import_notebook_defs():
    """
    pytest 夹具（fixture）：动态加载 `standalone-gemma3.ipynb` 中定义的
    所有类和函数（如 Gemma3Model、load_weights_into_gemma 等），
    返回一个类似模块的对象，供测试函数通过属性访问方式使用
    （例如 `mod.Gemma3Model`）。

    这样做的好处是无需把 notebook 手动转换成 .py 文件即可对其中的
    代码进行单元测试，保证 notebook 与测试用例始终基于同一份源码。
    """
    # notebook 所在目录：当前测试文件的上两级目录（tests/ 的父目录）
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-gemma3.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    """
    pytest 夹具：生成一个用于前向传播冒烟测试的随机整数张量，
    模拟 token id 序列。固定随机种子以保证测试结果可复现。
    """
    torch.manual_seed(123)  # 固定随机种子，保证每次运行生成同样的输入
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    """
    pytest 夹具：提供一份"迷你版" Gemma3 模型配置字典，用于
    `test_dummy_gemma3_forward` 的快速冒烟测试。所有维度都被
    设置得很小（vocab_size=100, emb_dim=32 等），目的是让测试
    运行得足够快，同时仍然覆盖 Gemma3 特有的结构性特性
    （如 qk_norm、分组查询注意力 n_kv_groups、滑动窗口等）。
    """
    return {
        "vocab_size": 100,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 8,
        "n_kv_groups": 1,
        "qk_norm": True,                # Gemma3 uses q/k RMSNorm
        "dtype": torch.float32,
        "rope_base": 1_000_000.0,       # global RoPE base
        "rope_local_base": 10_000.0,    # local RoPE base (unused in these tests)
        "context_length": 64,
        "sliding_window": 16,
        "layer_types": ["full_attention", "full_attention"],
        "query_pre_attn_scalar": 256,
    }


@torch.inference_mode()  # 关闭梯度计算，加速推理且节省显存/内存
def test_dummy_gemma3_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    """
    冒烟测试：使用迷你配置构造 Gemma3Model，执行一次前向传播，
    只验证输出张量的形状是否为 (batch_size, seq_len, vocab_size)。

    该测试不校验具体数值是否正确，只确认模型各层维度衔接无误、
    能够端到端跑通，属于最基础的"能不能跑"级别的验证。
    """
    torch.manual_seed(123)  # 固定随机种子，保证模型权重初始化可复现
    model = import_notebook_defs.Gemma3Model(dummy_cfg_base)
    out = model(dummy_input)
    # 断言输出形状与预期一致：批大小1、序列长度与输入相同、最后一维为词表大小
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"])


@torch.inference_mode()  # 关闭梯度计算，仅做前向对比
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_gemma3_base_equivalence_with_transformers(import_notebook_defs):
    """
    等价性测试：验证 notebook 中从零实现的 Gemma3Model 与 HuggingFace
    官方 `transformers` 库中的 Gemma3ForCausalLM 在相同权重、相同输入下
    是否输出数值上一致（在给定误差范围内）。

    步骤：
    1. 用同一份小配置分别构造本地 Gemma3Model 和 HF 版 Gemma3ForCausalLM；
    2. 把 HF 模型的 state_dict 通过 `load_weights_into_gemma` 转换/加载进
       本地模型，确保两者使用完全相同的权重；
    3. 用同一批随机 token 输入两个模型，比较输出 logits 是否在误差范围内一致。

    若当前环境未安装 transformers，则通过 skipif 标记跳过本测试。
    """
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM

    # Tiny config so the test is fast
    # 迷你配置：保证测试运行速度快，同时覆盖 GQA（n_kv_groups=2）、
    # qk_norm、局部/全局 RoPE、滑动窗口等 Gemma3 关键特性
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
    # 构造本地（notebook 中定义的）Gemma3 模型实现
    model = import_notebook_defs.Gemma3Model(cfg)

    # 构造与本地配置参数一一对应的 HuggingFace Gemma3TextConfig，
    # 确保两边模型结构（层数、维度、头数、RoPE 参数等）完全一致，
    # 这样才能公平比较数值输出。
    hf_cfg = Gemma3TextConfig(
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
        attn_implementation="eager",     # 使用最朴素的注意力实现，避免融合算子带来的数值差异
        torch_dtype=torch.float32,
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
        rope_scaling={"rope_type": "default"},
        ignore_keys_at_rope_validation={"full_attention", "sliding_attention"},
    )
    # HF 模型此时是随机初始化权重的（尚未与本地模型对齐）
    hf_model = Gemma3ForCausalLM(hf_cfg)

    # 取出 HF 模型的权重字典，将其"翻译"/加载进本地实现，
    # 使本地模型与 HF 模型拥有完全相同的参数值，
    # 这样后续 logits 的差异就只能来自实现逻辑本身，而非权重不同。
    hf_state = hf_model.state_dict()
    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    import_notebook_defs.load_weights_into_gemma(model, param_config, hf_state)

    # 构造同一批随机 token 输入，形状为 (batch=2, seq_len=context_length)
    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)          # 本地实现的前向输出
    theirs_logits = hf_model(x).logits  # HF 官方实现的前向输出
    # 核心断言：两个实现在相同权重、相同输入下的 logits 应在给定误差范围内一致，
    # 这是判定"本地实现与官方实现等价"的关键校验
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
