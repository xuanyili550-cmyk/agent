# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块用于测试 `standalone-llama32.ipynb` notebook 中实现的 Llama 3.2 模型定义。

测试内容包括:
1. 使用一个小型「dummy」配置对 Llama3Model 做一次前向传播,检查输出张量形状是否符合预期
   (batch_size, seq_len, vocab_size)。
2. 在安装了 `transformers` 库的前提下,将我们自己实现的 Llama3Model 与 HuggingFace 官方的
   LlamaForCausalLM 进行数值等价性对比:先把 HuggingFace 模型的随机初始化权重加载到我们的
   模型中,再用相同的输入做前向传播,断言两者输出的 logits 在给定容差内一致。

依赖 `llms_from_scratch.utils.import_definitions_from_notebook` 工具函数,
该函数会从 notebook 文件中动态导入类/函数定义(如 Llama3Model、load_weights_into_llama 等),
从而无需将 notebook 转换为 .py 文件即可对其中的代码进行单元测试。
"""

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


# 检测当前环境是否安装了 transformers 库,若未安装则跳过依赖该库的对比测试
transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def import_notebook_defs():
    # 定位到 notebook 所在目录(当前测试文件的上上级目录),并从中动态导入所需的类/函数定义
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-llama32.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    # 固定随机种子以保证测试可复现
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    # 构造一个体积很小的模型配置,仅用于快速验证前向传播是否能跑通、输出形状是否正确
    return {
        "vocab_size": 100,
        "emb_dim": 32,            # hidden_size
        "hidden_dim": 64,         # intermediate_size (FFN)
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 8,
        "n_kv_groups": 1,
        "dtype": torch.float32,
        "rope_base": 500_000.0,
        "rope_freq": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        },
        "context_length": 64,
    }


@torch.inference_mode()
def test_dummy_llama3_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    """使用小型 dummy 配置测试 Llama3Model 的前向传播是否能正常运行,
    并验证输出张量的形状是否为 (batch_size, seq_len, vocab_size)。"""
    torch.manual_seed(123)
    model = import_notebook_defs.Llama3Model(dummy_cfg_base)
    out = model(dummy_input)
    # 断言输出形状符合预期: (批大小=1, 序列长度=8, 词表大小=100)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"])


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_llama3_base_equivalence_with_transformers(import_notebook_defs):
    """在安装了 transformers 库的条件下,验证我们自己实现的 Llama3Model
    与 HuggingFace 官方 LlamaForCausalLM 在加载相同权重后,对相同输入
    产生的 logits 是否在数值上等价(即实现是否正确对齐官方实现)。"""
    from transformers.models.llama import LlamaConfig, LlamaForCausalLM
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

    # 构建我们自己(notebook 中定义)的 Llama3Model 实例
    ours = import_notebook_defs.Llama3Model(cfg)

    # 构建与之对应的 HuggingFace LlamaConfig,确保各超参数与上面的 cfg 一一对应
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
    # 使用随机初始化权重构建 HuggingFace 官方模型
    theirs = LlamaForCausalLM(hf_cfg)

    # 取出 HuggingFace 模型的随机初始化权重,并将其加载进我们自己实现的模型中,
    # 以便两者在相同权重下进行输出对比
    hf_state = theirs.state_dict()
    import_notebook_defs.load_weights_into_llama(ours, {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}, hf_state)

    # 构造相同的随机输入 token id 序列,分别喂给两个模型
    x = torch.randint(0, cfg["vocab_size"], (2, 8), dtype=torch.long)
    ours_logits = ours(x)
    theirs_logits = theirs(x).logits.to(ours_logits.dtype)

    # 核心断言: 在加载了相同权重的前提下,两套实现对相同输入应给出数值上几乎一致的 logits
    # (rtol/atol 容差用于容忍浮点运算顺序不同带来的微小数值误差)
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
