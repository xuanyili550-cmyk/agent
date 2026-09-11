# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是对 `standalone-olmo3-plus-kv-cache.ipynb` notebook 中定义的
Olmo3（带 KV 缓存版本）模型实现的单元测试。

主要覆盖两方面：
1. `test_dummy_olmo3_forward`：仅用一个很小的随机配置，验证
   `Olmo3Model` 能正常完成前向传播，且输出张量形状符合预期
   （不依赖 HuggingFace `transformers`，用于快速冒烟测试）。
2. `test_olmo3_base_equivalence_with_transformers`：在安装了
   `transformers` 库的前提下，将 notebook 中手写的 Olmo3 实现与
   HuggingFace 官方 `Olmo3ForCausalLM` 实现进行数值对齐测试——
   把 HuggingFace 随机初始化的权重搬运（load）到我们自己的模型中，
   再比较两者在同一输入下的 logits 输出是否一致，从而验证我们的
   实现（包括注意力、RoPE、归一化、滑动窗口等细节）与官方实现等价。

测试所需的模型定义（`Olmo3Model`、`load_weights_into_olmo` 等）并不是
直接从 .py 文件导入的，而是通过 `import_definitions_from_notebook`
工具函数，动态执行同目录下的 notebook 代码单元格后获取的模块对象，
这样可以保证测试始终针对 notebook 中"最新"的代码，而不是可能过时的
副本。
"""

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


# 检测当前环境是否安装了 HuggingFace `transformers` 库，用于后面对
# 依赖 transformers 的等价性测试做 skipif 条件判断。
# 注意（风险标注，未修改）：这里只 `import importlib`，随后却直接访问
# `importlib.util.find_spec(...)`。`importlib.util` 是一个子模块，
# 按 Python 的导入机制，仅 `import importlib` 并不保证
# `importlib.util` 属性一定存在——它之所以能在当前环境下正常工作，
# 是因为在此之前 `import torch` / `pytest` 等库已经在进程内隐式
# 导入过 `importlib.util`，从而把该子模块挂到了 `importlib` 对象上。
# 这依赖的是"其他库的导入副作用"这一实现细节，并非语言规范保证的行为，
# 在不同版本/不同导入顺序下存在细微的跨版本风险，因此这里按要求只标注、
# 不做修改（若要彻底规避，应显式改成 `import importlib.util`）。
transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def import_notebook_defs():
    """
    Pytest 夹具：动态导入同目录（上一级目录）下的
    `standalone-olmo3-plus-kv-cache.ipynb` notebook，
    执行其中的代码单元格，并返回一个包含所有顶层定义
    （如 `Olmo3Model`、`load_weights_into_olmo` 等类/函数）的模块对象，
    供各测试用例直接引用，避免在测试代码里重复粘贴 notebook 实现。
    """
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-olmo3-plus-kv-cache.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    """
    Pytest 夹具：生成一个形状为 (1, 8) 的随机整数张量，
    模拟 batch_size=1、序列长度=8 的 token id 输入，
    用固定随机种子保证测试结果可复现。
    """
    torch.manual_seed(123)
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    """
    Pytest 夹具：提供一份体积很小、用于快速前向测试的 Olmo3 模型配置字典。
    仅用于 `test_dummy_olmo3_forward`（不涉及与 HuggingFace 权重对齐），
    因此各维度都刻意设置得很小，以加快测试速度。
    """
    return {
        "vocab_size": 100,          # 词表大小（很小，仅测试用）
        "context_length": 64,       # 最大上下文长度
        "emb_dim": 32,               # 词嵌入 / 隐藏层维度
        "n_heads": 4,                 # 注意力头（query 头）数量
        "n_layers": 2,                # Transformer 层数
        "hidden_dim": 64,            # 前馈网络（FFN）中间层维度
        "head_dim": 8,                 # 每个注意力头的维度
        "n_kv_heads": 1,  # 4 query heads, 1 KV groups -> group_size = 4
        "attention_bias": False,     # 注意力线性层是否使用 bias
        "attention_dropout": 0.0,    # 注意力 dropout 概率
        "sliding_window": 4,          # 滑动窗口注意力的窗口大小
        "layer_types": ["full_attention"] * 2,  # 每层的注意力类型（此处两层均为全量注意力）

        # RoPE config
        "rope_base": 10_000.0,        # RoPE 旋转位置编码的基数 theta
        "rope_attention_factor": 1.0,  # RoPE 注意力缩放因子
        "rope_type": "default",       # RoPE 类型（默认，不做长度外推缩放）
        "rope_factor": 1.0,           # RoPE 长度外推缩放因子
        "rope_orig_max": 64,          # RoPE 原始训练时的最大上下文长度
        "rms_norm_eps": 1e-6,          # RMSNorm 的数值稳定项 epsilon
        "dtype": torch.float32,       # 模型参数与计算使用的数据类型
    }

@torch.inference_mode()
def test_dummy_olmo3_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    """
    冒烟测试：使用极小的随机配置构建 `Olmo3Model`，对一个随机 token
    序列做一次前向传播，只校验输出 logits 的形状是否为
    (batch_size, seq_len, vocab_size)，不涉及与外部实现的数值比对，
    也不依赖是否安装了 `transformers`。
    """
    torch.manual_seed(123)
    model = import_notebook_defs.Olmo3Model(dummy_cfg_base)
    out = model(dummy_input)
    # 校验输出形状：(batch=1, seq_len=输入序列长度, vocab_size=词表大小)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"


@torch.inference_mode()
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
def test_olmo3_base_equivalence_with_transformers(import_notebook_defs):
    """
    等价性测试：验证 notebook 中手写的 Olmo3 实现与 HuggingFace 官方
    `Olmo3ForCausalLM` 实现在数值上是否一致。

    做法：
    1. 用同一份小配置分别构建"我们自己的" `Olmo3Model` 和 HuggingFace
       的 `Olmo3ForCausalLM`（后者初始权重是随机的）。
    2. 通过 `load_weights_into_olmo` 把 HuggingFace 模型的 state_dict
       权重搬运到我们自己的模型中，确保两者权重完全一致。
    3. 用相同的随机 token 序列分别跑一次前向传播，比较两者输出的
       logits 是否在给定误差范围内相等（`torch.testing.assert_close`）。

    若该测试通过，说明我们手写实现的注意力机制、RoPE、RMSNorm、
    滑动窗口掩码等细节均与官方实现在数值上等价。
    仅当环境中安装了 `transformers` 时才会执行（否则被 skip）。
    """
    from transformers import Olmo3Config, Olmo3ForCausalLM

    # Tiny config so the test is fast
    # 一份很小的配置，保证测试运行速度快
    cfg = {
        "vocab_size": 257,             # 词表大小
        "context_length": 8,            # 最大上下文长度（同时也是本测试用的序列长度）
        "emb_dim": 32,                  # 隐藏层维度
        "n_heads": 4,                    # query 注意力头数
        "n_layers": 2,                   # Transformer 层数
        "hidden_dim": 64,                # FFN 中间层维度
        "head_dim": 8,                    # 每个注意力头维度
        "qk_norm": True,                 # 是否对 Q/K 做归一化（Olmo3 特性之一）
        "n_kv_heads": 2,                  # KV 头（分组）数量
        "sliding_window": 4,              # 滑动窗口大小
        "layer_types": ["full_attention", "full_attention"],  # 两层均为全量注意力
        "dtype": torch.float32,           # 计算数据类型
        "query_pre_attn_scalar": 256,     # Olmo3 中用于缩放注意力分数的标量

        # required by TransformerBlock
        # TransformerBlock 构造时必需的字段
        "attention_bias": False,

        # required by RMSNorm and RoPE setup in Olmo3Model
        # Olmo3Model 中 RMSNorm 与 RoPE 初始化所必需的字段
        "rms_norm_eps": 1e-6,
        "rope_base": 1_000_000.0,
        "rope_attention_factor": 1.0,
        "rope_type": "default",
        "rope_factor": 1.0,
        "rope_orig_max": 8,

        # extra HF-only stuff
        # 仅 HuggingFace 侧配置需要的额外字段（我们自己的实现不使用）
        "rope_local_base": 10_000.0,
    }

    # 构建"我们自己的" Olmo3 模型（此时权重仍为随机初始化，
    # 后面会被 HuggingFace 的权重覆盖）
    model = import_notebook_defs.Olmo3Model(cfg)

    # 构建与上面配置等价的 HuggingFace Olmo3Config / Olmo3ForCausalLM，
    # 逐字段把我们的 cfg 映射到 HuggingFace 对应的参数名上
    hf_cfg = Olmo3Config(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_heads"],
        rope_theta=cfg["rope_base"],
        rope_local_base_freq=cfg["rope_local_base"],
        layer_types=cfg["layer_types"],
        sliding_window=cfg["sliding_window"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
        rope_scaling={"rope_type": "default"},
        qk_norm=cfg["qk_norm"],
        rms_norm_eps=cfg["rms_norm_eps"],
    )
    # HuggingFace 模型此时使用随机初始化权重
    hf_model = Olmo3ForCausalLM(hf_cfg)

    # 取出 HuggingFace 模型的权重字典，准备搬运到我们自己的模型中
    hf_state = hf_model.state_dict()
    param_config = {
        "n_layers": cfg["n_layers"],
        "hidden_dim": cfg["hidden_dim"],
    }
    # 将 HuggingFace 权重按名称/形状对应关系拷贝进我们自己实现的模型，
    # 使两者权重完全一致，从而后续的输出差异只可能来自实现逻辑本身
    import_notebook_defs.load_weights_into_olmo(model, param_config, hf_state)

    # 构造同一份随机 token 序列，形状 (batch=2, seq_len=context_length)
    x = torch.randint(
        0,
        cfg["vocab_size"],
        (2, cfg["context_length"]),
        dtype=torch.long,
    )
    ours_logits = model(x)
    theirs_logits = hf_model(x).logits
    # 核心断言：在权重完全一致的前提下，两套实现对同一输入的 logits
    # 输出应在给定的相对/绝对误差范围内几乎相等，用于验证我们手写实现
    # 与 HuggingFace 官方实现在数值上等价
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
