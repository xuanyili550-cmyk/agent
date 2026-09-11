"""
本模块是 OLMo3 独立实现笔记本（standalone-olmo3.ipynb）的 pytest 测试文件。

测试思路：
1. 通过 `import_definitions_from_notebook` 工具函数，把 Jupyter Notebook 中定义的
   `Olmo3Model`（以及权重加载函数 `load_weights_into_olmo`）当作普通 Python 模块导入，
   这样可以直接对 notebook 里的代码做单元测试，而不必手工把代码复制成 .py 文件。
2. `test_dummy_olmo3_forward`：用一个很小的随机配置做前向传播冒烟测试（smoke test），
   只检查输出张量的形状是否符合预期，不检查数值是否正确。
3. `test_olmo3_base_equivalence_with_transformers`：在本地安装了 HuggingFace
   `transformers` 库的前提下，构造一个参数完全对齐的 HuggingFace `Olmo3ForCausalLM`，
   把其随机初始化的权重加载进我们自己实现的 `Olmo3Model`，然后比较两者在同一输入下的
   输出 logits 是否在数值误差范围内一致，用来验证自研实现与官方实现在数学上等价。

风险/跨版本提示（按用户要求：不改动代码，仅标注上报，见下方 `transformers_installed` 赋值处的具体位置注释）：
- `importlib.util.find_spec(...)`：本文件只写了 `import importlib`，
  并没有显式 `import importlib.util`。`importlib.util` 是否能通过属性访问拿到，
  依赖于运行期间是否已有其他模块（如 pytest / torch）先一步导入了 `importlib.util`
  从而把它注册为 `importlib` 包的属性——这不是 Python 语言规范保证的行为，只是
  CPython 目前的实际效果（本机 Python 3.13 环境下经过实测可以正常工作）。
  在极端情况下（例如更换解释器实现、或未来某个 Python 版本的导入机制发生变化、
  或以某种方式单独执行本模块而不先导入 pytest/torch）该属性访问可能会抛出
  `AttributeError: module 'importlib' has no attribute 'util'`。
  这是一个跨版本/跨环境风险项，且原始 rasbt 仓库中的姊妹测试文件
  （test_olmo3_kvcache_nb.py）也采用了完全相同的写法，属于该代码库的既有风格，
  因此按用户要求不做修改，仅在此标注并上报。
"""

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook


# 检测当前 Python 环境是否安装了 transformers 库；
# 若未安装，则下方依赖 transformers 的等价性测试会被跳过（见 pytest.mark.skipif）。
# 【风险标注，不修改】：此处通过 `importlib.util` 访问的是尚未被本文件显式导入的
# 子模块，详见模块顶部 docstring 中的说明。
transformers_installed = importlib.util.find_spec("transformers") is not None


@pytest.fixture
def import_notebook_defs():
    """
    pytest 夹具（fixture）：将 standalone-olmo3.ipynb 笔记本中定义的类和函数
    （例如 Olmo3Model、load_weights_into_olmo）作为一个 Python 模块动态导入并返回，
    供下面的测试函数直接调用，等价于 `import standalone_olmo3 as mod`。
    """
    # 笔记本文件所在目录：当前测试文件在 .../13_olmo3/tests/ 下，
    # parents[1] 即上溯两级目录得到 .../13_olmo3/，笔记本就存放在该目录下。
    nb_dir = Path(__file__).resolve().parents[1]
    mod = import_definitions_from_notebook(nb_dir, "standalone-olmo3.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    """
    pytest 夹具：生成一个固定随机种子下的伪造输入 token id 张量，
    形状为 (batch_size=1, seq_len=8)，取值范围 [0, 100)，用于前向传播冒烟测试。
    """
    torch.manual_seed(123)  # 固定随机种子，保证每次测试生成的输入完全一致，结果可复现
    return torch.randint(0, 100, (1, 8))  # batch size 1, seq length 8


@pytest.fixture
def dummy_cfg_base():
    """
    pytest 夹具：提供一份体积很小的 OLMo3 模型配置字典，用于快速前向传播测试。
    该配置刻意把各维度（词表、隐藏维度、层数等）设置得很小，以保证测试运行迅速，
    并不追求和真实 OLMo3 模型配置完全一致。
    """
    return {
        "vocab_size": 100,
        "context_length": 64,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 64,
        "head_dim": 8,
        # 原注释：4 query heads, 1 KV groups -> group_size = 4
        # 说明：n_kv_heads 指的是「KV 头的数量」而非「KV 分组数量」，
        # 4 个 query heads 共享 1 个 kv head，因此分组大小 group_size = 4 / 1 = 4；
        # 原注释表述为「1 KV groups」用词不够精确，但不影响实际取值，
        # 因此保留原注释不做修改，仅在此附加说明。
        "n_kv_heads": 1,  # 4 query heads, 1 KV groups -> group_size = 4
        "attention_bias": False,
        "attention_dropout": 0.0,
        "sliding_window": 4,
        "layer_types": ["full_attention"] * 2,  # 两层都使用全量注意力（非滑动窗口局部注意力）

        # RoPE config
        "rope_base": 10_000.0,
        "rope_attention_factor": 1.0,
        "rope_type": "default",
        "rope_factor": 1.0,
        "rope_orig_max": 64,
        "rms_norm_eps": 1e-6,
        "dtype": torch.float32,
    }

@torch.inference_mode()  # 推理模式：关闭梯度追踪，加快前向传播速度、降低显存占用
def test_dummy_olmo3_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    """
    冒烟测试：使用极小配置构造 Olmo3Model，跑一次前向传播，
    只验证输出张量的形状是否为 (batch_size, seq_len, vocab_size)，
    不校验具体数值是否正确（数值正确性由下面的等价性测试负责）。
    """
    torch.manual_seed(123)  # 固定种子，确保模型权重初始化可复现
    model = import_notebook_defs.Olmo3Model(dummy_cfg_base)
    out = model(dummy_input)
    # 断言输出形状：批大小恒为 1，序列长度应与输入一致，最后一维应为词表大小（每个位置的 logits）
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), \
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"


@torch.inference_mode()  # 推理模式：只做前向对比，不需要梯度
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")  # 未安装 transformers 时跳过本测试
def test_olmo3_base_equivalence_with_transformers(import_notebook_defs):
    """
    等价性测试：验证本仓库自研的 Olmo3Model 与 HuggingFace 官方 Olmo3ForCausalLM
    在参数配置对齐、且加载同一套随机初始化权重的前提下，对同一批输入是否输出
    数值上（在给定误差范围内）完全一致的 logits。

    测试步骤：
    1. 构造一份很小的自定义配置 cfg（词表、层数、维度都很小，测试更快）；
    2. 用 cfg 实例化自研的 Olmo3Model；
    3. 用等价参数构造 HuggingFace 的 Olmo3Config 与 Olmo3ForCausalLM（权重随机初始化）；
    4. 通过 load_weights_into_olmo 把 HuggingFace 模型的随机权重复制进自研模型，
       确保两个模型使用完全相同的参数值，这样才有意义比较输出是否一致；
    5. 用同一份随机 token 输入分别跑两个模型的前向传播；
    6. 用 torch.testing.assert_close 比较两者输出的 logits 是否在给定误差范围内相等。
    """
    from transformers import Olmo3Config, Olmo3ForCausalLM

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
        "n_kv_heads": 2,
        "sliding_window": 4,
        "layer_types": ["full_attention", "full_attention"],
        "dtype": torch.float32,
        "query_pre_attn_scalar": 256,

        # required by TransformerBlock
        "attention_bias": False,

        # required by RMSNorm and RoPE setup in Olmo3Model
        "rms_norm_eps": 1e-6,
        "rope_base": 1_000_000.0,
        "rope_attention_factor": 1.0,
        "rope_type": "default",
        "rope_factor": 1.0,
        "rope_orig_max": 8,

        # extra HF-only stuff
        "rope_local_base": 10_000.0,  # HuggingFace 对局部（滑动窗口）注意力层使用的独立 RoPE base，本仓库自研实现未用到该字段
    }

    # 1) 用自定义小配置实例化自研模型（此时权重仍是随机初始化的，尚未与 HF 对齐）
    model = import_notebook_defs.Olmo3Model(cfg)

    # 2) 用完全对应的参数构造 HuggingFace 官方配置，
    #    保证两边的模型结构（层数、头数、RoPE 参数等）严格一致，
    #    这样后面才能把 HF 的权重直接搬到自研模型上做数值比较。
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
        attn_implementation="eager",  # 强制使用朴素 Python 实现的注意力（而非 flash-attn 等融合算子），确保数值可比对
        torch_dtype=torch.float32,
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
        rope_scaling={"rope_type": "default"},
        qk_norm=cfg["qk_norm"],
        rms_norm_eps=cfg["rms_norm_eps"],
    )
    # HuggingFace 模型此时会用其内部默认的随机初始化方式生成权重
    hf_model = Olmo3ForCausalLM(hf_cfg)

    # 3) 把 HuggingFace 模型的权重字典取出来，
    #    通过 notebook 中定义的 load_weights_into_olmo 函数灌入自研模型，
    #    确保两个模型此后共享完全相同的参数值（否则输出必然不同，比较没有意义）。
    hf_state = hf_model.state_dict()
    param_config = {
        "n_layers": cfg["n_layers"],
        "hidden_dim": cfg["hidden_dim"],
    }
    import_notebook_defs.load_weights_into_olmo(model, param_config, hf_state)

    # 4) 构造同一份随机 token 输入（batch_size=2，序列长度=context_length），供两个模型共用
    x = torch.randint(
        0,
        cfg["vocab_size"],
        (2, cfg["context_length"]),
        dtype=torch.long,
    )
    ours_logits = model(x)  # 自研 Olmo3Model 的前向输出
    theirs_logits = hf_model(x).logits  # HuggingFace 官方实现的前向输出（.logits 取出预测分布对应的原始分数）
    # 核心断言：两者输出的 logits 应在给定的相对/绝对误差范围内数值一致，
    # 从而证明自研实现与 HuggingFace 官方实现在数学上是等价的。
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)