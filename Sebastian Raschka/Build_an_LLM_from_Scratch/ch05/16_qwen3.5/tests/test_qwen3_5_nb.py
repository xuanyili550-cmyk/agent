# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块：qwen3.5 教学 notebook 的单元测试。

用途：
1. 从 `qwen3.5.ipynb` notebook 中动态导入模型定义（`Qwen3_5Model` 等），
   在不依赖把 notebook 转换成 .py 文件的前提下，对其中定义的类和函数做常规
   pytest 单元测试。
2. 如果本机环境安装了 HuggingFace `transformers` 且其中包含 `qwen3_5` 模型
   实现（或仓库根目录下存在本地源码 `transformers-main/src`），则额外做一次
   "自研实现 vs. 官方 transformers 实现" 的数值等价性测试，用随机权重初始化
   官方模型，把权重原样搬到自研实现里，再比较两者在同一输入上的输出 logits
   是否一致（在数值容差范围内）。若 transformers 不可用，则跳过该测试
   （通过 `pytest.mark.skipif` 实现）。

注意：本文件仅添加中文注释与一处确定性 bug 修复（见下方 `import importlib.util`
处标注），未改变任何测试逻辑与断言语义。
"""

import importlib
import importlib.util  # 【bug修复】原代码只 `import importlib`，却在下方直接使用 `importlib.util.find_spec(...)`；
# `importlib.util` 是子模块，Python 不保证 `import importlib` 会自动让 `importlib.util` 可用
# （是否可用取决于是否有其他第三方库作为副作用提前 import 了它，纯属偶然、不可靠）。
# 在某些环境/Python 版本下未被间接导入时会直接 AttributeError: module 'importlib' has no attribute 'util'。
# 这是全库统一做法：显式补上 `import importlib.util`，消除对隐式副作用导入的依赖。
import sys
from pathlib import Path

import pytest
import torch

from llms_from_scratch.utils import import_definitions_from_notebook  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)


def _import_qwen3_5_classes():
    """
    尝试导入 HuggingFace transformers 中的 Qwen3.5 相关类。

    优先从当前已安装的 `transformers` 包里导入 `Qwen3_5TextConfig` 与
    `Qwen3_5ForCausalLM`；如果当前安装的 transformers 版本还没有收录
    qwen3_5 模型（导入失败，抛出任意 Exception），则退而求其次，尝试从仓库
    同级目录下的本地源码 `transformers-main/src`（一份手动 clone/构建的、
    包含 qwen3_5 支持的 transformers 源码树）动态加载。

    做法：
    - 先把 sys.modules 中所有已缓存的 "transformers" 及其子模块清掉，避免
      混用「已安装版本的部分缓存」和「本地源码版本」导致的不一致行为。
    - 把本地源码目录插入 sys.path 最前面，让后续 `import transformers.*`
      优先命中本地源码。

    风险项（不改，仅标注）：
    - `except Exception:` 捕获范围很宽，会掩盖除 ImportError/AttributeError
      以外的其他异常（比如本地源码本身存在语法错误等），排查时不易定位。
    - 直接删改 `sys.modules` 是全局副作用，若测试与其他用例共享进程/夹具
      顺序不当，可能影响之后再导入 transformers 的其他测试。
    - 这是在做 "猴子补丁" 式的模块替换，跨 Python/transformers 版本行为
      可能不稳定（例如本地源码目录结构变化、依赖的私有 API 发生变化等）。

    返回：
        (Qwen3_5TextConfig, Qwen3_5ForCausalLM) 两个类的元组。
    """
    try:
        # 优先尝试：假设当前安装的 transformers 版本已经内置支持 qwen3_5
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        return Qwen3_5TextConfig, Qwen3_5ForCausalLM
    except Exception:
        # 退路：改用仓库旁边的本地 transformers 源码树（须手动准备好）
        repo_root = Path(__file__).resolve().parents[3]
        local_src = repo_root / "transformers-main" / "src"
        if not local_src.exists():
            # 本地源码也不存在，说明确实没有 qwen3_5 支持，把原始异常继续抛出
            raise

        # 清空已缓存的 transformers 相关模块，防止新旧版本混用
        for name in list(sys.modules):
            if name == "transformers" or name.startswith("transformers."):
                del sys.modules[name]
        sys.path.insert(0, str(local_src))  # 让本地源码优先于已安装包被 import

        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        return Qwen3_5TextConfig, Qwen3_5ForCausalLM


# 模块级探测：判断当前环境是否安装了 transformers 包（不管是否含 qwen3_5 支持）
transformers_installed = importlib.util.find_spec("transformers") is not None
if transformers_installed:
    try:
        # 即便装了 transformers，也可能版本太旧不含 qwen3_5，因此需要再实际尝试导入一次
        Qwen3_5TextConfig, Qwen3_5ForCausalLM = _import_qwen3_5_classes()
    except Exception:
        # 导入失败（含本地源码回退也失败）时，视为"transformers 不可用"，后续对比测试将被跳过
        transformers_installed = False
        Qwen3_5TextConfig, Qwen3_5ForCausalLM = None, None
else:
    Qwen3_5TextConfig, Qwen3_5ForCausalLM = None, None


@pytest.fixture
def import_notebook_defs():
    """
    夹具：从 `qwen3.5.ipynb` notebook 文件中动态导入其中定义的类/函数
    （例如 `Qwen3_5Model`、`load_weights_into_qwen3_5`），返回一个类似
    模块对象，供测试函数以 `import_notebook_defs.XXX` 的形式访问。

    实现细节：把 notebook 所在目录（当前测试文件的上两级目录）加入
    sys.path，以便 notebook 内部若有相对 import 也能正常工作。
    """
    nb_dir = Path(__file__).resolve().parents[1]
    if str(nb_dir) not in sys.path:
        sys.path.insert(0, str(nb_dir))

    mod = import_definitions_from_notebook(nb_dir, "qwen3.5.ipynb")
    return mod


@pytest.fixture
def dummy_input():
    """夹具：生成一个固定随机种子下的假输入 token 序列，形状为 (batch=1, seq_len=8)。"""
    torch.manual_seed(123)  # 固定随机种子，保证测试结果可复现
    return torch.randint(0, 100, (1, 8))


@pytest.fixture
def dummy_cfg_base():
    """
    夹具：一份用于快速前向验证的极小规模 Qwen3.5 配置字典。

    刻意把各维度（emb_dim/hidden_dim/n_layers/n_heads 等）设得很小，
    只是为了让模型能在 CPU 上快速跑通一次前向传播，不追求真实模型规模。
    其中 `layer_types` 混合了 "linear_attention" 与 "full_attention" 两种
    层类型，用来覆盖 Qwen3.5 中混合注意力架构的两条分支代码路径。
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
        "rope_base": 10_000.0,
        "context_length": 64,
        "partial_rotary_factor": 1.0,
        "rms_norm_eps": 1e-6,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "layer_types": ["linear_attention", "full_attention"],
    }


@torch.inference_mode()  # 关闭梯度追踪与 autograd 记录，测试仅做前向推理，加速且省内存
def test_dummy_qwen3_5_forward(dummy_cfg_base, dummy_input, import_notebook_defs):
    """
    冒烟测试：仅验证 notebook 中定义的 `Qwen3_5Model` 能够正常前向传播，
    且输出张量形状符合预期 (batch_size, seq_len, vocab_size)。

    不涉及与 HuggingFace 官方实现的数值对比，纯粹检查"模型能跑通、形状对不对"。
    """
    torch.manual_seed(123)  # 固定随机种子，保证模型权重初始化可复现
    model = import_notebook_defs.Qwen3_5Model(dummy_cfg_base)
    out = model(dummy_input)
    # 断言：输出形状必须是 (batch=1, seq_len=dummy_input 的序列长度, vocab_size)
    assert out.shape == (1, dummy_input.size(1), dummy_cfg_base["vocab_size"]), (
        f"Expected shape (1, seq_len, vocab_size), got {out.shape}"
    )


@torch.inference_mode()  # 同上，只做前向推理对比，不需要梯度
@pytest.mark.skipif(not transformers_installed, reason="transformers not installed")
# 若当前环境未安装 transformers，或安装的版本不包含 qwen3_5 支持（且没有可用的本地源码回退），
# 则跳过本测试，避免因环境缺依赖而误报失败
def test_qwen3_5_base_equivalence_with_transformers(import_notebook_defs):
    """
    等价性测试：验证 notebook 中自研的 `Qwen3_5Model` 与 HuggingFace 官方
    `Qwen3_5ForCausalLM` 实现，在使用完全相同的权重与输入时，输出的 logits
    是否数值一致（在给定容差内）。

    步骤：
    1. 用一份小规模配置分别构造自研模型 `model` 与官方模型 `hf_model`
       （官方模型的配置项通过 `Qwen3_5TextConfig` 显式映射自同一份 cfg，
       确保两者结构完全对应，包括 RoPE 参数、混合线性/全量注意力层类型等）。
    2. 取出官方模型（随机初始化）的 `state_dict`，通过
       `load_weights_into_qwen3_5` 把这些权重搬运/映射进自研模型，
       确保两个模型使用完全相同的参数值。
    3. 用同一份随机整数 token 序列分别喂给两个模型，比较输出 logits。
    """
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
        "partial_rotary_factor": 1.0,
        "rms_norm_eps": 1e-6,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "dtype": torch.float32,
    }
    model = import_notebook_defs.Qwen3_5Model(cfg)  # 自研（notebook 中定义）的模型实现

    # 用同一份 cfg 的参数值构造官方 HuggingFace 配置对象，字段名不同但语义一一对应
    hf_cfg = Qwen3_5TextConfig(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_groups"],
        layer_types=cfg["layer_types"],
        linear_conv_kernel_dim=cfg["linear_conv_kernel_dim"],
        linear_key_head_dim=cfg["linear_key_head_dim"],
        linear_value_head_dim=cfg["linear_value_head_dim"],
        linear_num_key_heads=cfg["linear_num_key_heads"],
        linear_num_value_heads=cfg["linear_num_value_heads"],
        tie_word_embeddings=False,
        use_cache=False,  # 测试只做单次前向，不需要 KV cache
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": cfg["rope_base"],
            "partial_rotary_factor": cfg["partial_rotary_factor"],
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        torch_dtype=torch.float32,
    )
    hf_cfg._attn_implementation = "eager"  # 强制官方模型使用朴素 eager 实现的注意力，而非融合/flash 版本，确保数值路径可复现比较
    hf_model = Qwen3_5ForCausalLM(hf_cfg)  # 官方实现，随机初始化权重

    hf_state = hf_model.state_dict()  # 取出官方模型的随机初始化权重
    param_config = {"n_layers": cfg["n_layers"], "layer_types": cfg["layer_types"]}
    # 把官方模型的权重按名称/结构映射后加载进自研模型，使二者参数完全一致
    import_notebook_defs.load_weights_into_qwen3_5(model, param_config, hf_state)

    # 构造同一份随机 token 序列（batch=2, seq_len=context_length），分别喂给两个模型
    x = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]), dtype=torch.long)
    ours_logits = model(x)
    theirs_logits = hf_model(x, use_cache=False).logits
    # 核心断言：在权重、输入完全相同的前提下，自研实现与官方实现的输出 logits
    # 必须在给定相对/绝对容差内数值一致，用以验证自研实现的正确性
    torch.testing.assert_close(ours_logits, theirs_logits, rtol=1e-5, atol=1e-5)
