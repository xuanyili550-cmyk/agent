# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

"""
本模块是内部单元测试文件,用于验证「从 GPT 改造为 Llama」系列笔记本中
RoPE（旋转位置编码，Rotary Position Embedding）以及 SiLU、RMSNorm 等
组件实现的正确性。

核心验证思路：
1. 从对应的 Jupyter Notebook（.ipynb）中动态导入自定义实现的函数/类
   （precompute_rope_params、compute_rope、SiLU、RMSNorm）；
2. 分别与业界权威的两种参考实现做数值对比：
   - HuggingFace transformers 库中的 LlamaRotaryEmbedding /
     apply_rotary_pos_emb（官方 Llama 实现）；
   - LitGPT 项目中的 litgpt_build_rope_cache / litgpt_apply_rope
     （另一套独立的开源实现，本文件中直接复制了其源码）；
3. 使用 torch.testing.assert_close / torch.allclose 逐一比较
   cos、sin 缓存以及应用旋转位置编码后的 queries、keys 张量，
   只要三方实现在数值上高度吻合，就说明自定义 RoPE 实现是正确的。

文件覆盖了三种 RoPE 场景：
- test_rope_llama2:      Llama 2 标准 RoPE（theta_base=10000，无频率缩放）；
- test_rope_llama3:      Llama 3 RoPE（theta_base=500000，更大的 base 以支持更长上下文）；
- test_rope_llama3_12:   Llama 3.1/3.2 RoPE（在标准 RoPE 基础上引入了
                         NTK-aware 的频率缩放/平滑插值，用于扩展上下文长度）。
"""

import io
import os
import sys
import types
import nbformat
from packaging import version
from typing import Optional, Tuple
import torch
import pytest
import transformers
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


transformers_version = transformers.__version__  # 记录当前安装的 transformers 版本号，用于后面根据版本差异选择不同的 API 调用方式

# LitGPT code function `litgpt_build_rope_cache` from https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
# LitGPT is licensed under Apache v2: https://github.com/Lightning-AI/litgpt/blob/main/LICENSE


def litgpt_build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
    """
    # 这是 LitGPT 项目中构建 RoPE cos/sin 缓存的独立参考实现（第三方来源，
    # 与本项目笔记本中的 precompute_rope_params 各自独立编写），
    # 用作与自定义实现进行交叉验证的“第二参考答案”。

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    # 上面一行计算 RoPE 的逆频率 theta_i = 1 / base^(2i/d)，i 从 0 到 n_elem/2 - 1，
    # 这是 RoPE 论文中的标准定义，频率随维度索引增大而指数衰减

    if extra_config is not None:
        # 进入该分支代表启用了 Llama 3.1 / 3.2 的 NTK-aware 频率缩放（用于将预训练时的
        # 上下文长度外推到更长的上下文），下面是对高频/低频分量做平滑插值的核心逻辑
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        wavelen = 2 * torch.pi / theta
        ratio = orig_context_len / wavelen
        smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)
        # smooth_factor 被裁剪到 [0, 1] 区间：0 表示完全使用缩放后的低频（做外推），
        # 1 表示完全保留原始高频（不缩放），中间值则做线性插值平滑过渡

        # Compute adjusted_theta without masked indexing
        adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
        theta = adjusted_theta
        # 用平滑因子在“缩放后的频率”和“原始频率”之间做线性混合，
        # 避免在频率边界处出现突变（这正是 Llama 3.1/3.2 相较 Llama 3 的关键改进点）

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    # 生成位置索引序列 [0, 1, ..., seq_len-1]，condense_ratio 可用于压缩位置间隔（默认不压缩）

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    # 对位置索引和频率做外积，得到 (seq_len, n_elem/2) 的角度矩阵，
    # 再沿最后一维复制一份拼接为 (seq_len, n_elem)，
    # 因为 RoPE 需要对头维度的前后两半使用相同的一组角度（对应旋转矩阵的两个分量）

    return torch.cos(idx_theta), torch.sin(idx_theta)
    # 返回预先计算好的余弦、正弦缓存表，后续在 litgpt_apply_rope 中直接查表使用


# LitGPT code from https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
# LitGPT is licensed under Apache v2: https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
def litgpt_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # LitGPT 对 RoPE 的应用函数：把输入张量 x 的最后一维（头维度）一分为二，
    # 通过“旋转半区”（rotate_half）的方式实现旋转位置编码，这是与
    # 「复数乘法形式」等价的另一种常见实现方式
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2:]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    # rotated 即“旋转半区”结果：把后半部分取负后放到前面，前半部分放到后面，
    # 对应二维旋转矩阵中 (-x2, x1) 的构造方式
    if cos.dim() > 1:
        # batch dimensions must align
        # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
        # we count from back because all of apply_rope does
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)
        # 在倒数第三维插入一个维度，用于与 (B, nh, T, hs) 形状的 x 广播对齐（对齐注意力头维度 nh）

    roped = (x * cos) + (rotated * sin)
    # 这一步就是 RoPE 的核心旋转公式：new_x = x * cos(theta) + rotate_half(x) * sin(theta)，
    # 等价于把每一对 (x_i, x_{i+d/2}) 看作复数分量，乘以 e^{i*theta} 做旋转
    return roped.to(dtype=x.dtype)


@pytest.fixture(scope="module")
def notebook():
    # 该 fixture 用于从磁盘上的 .ipynb 笔记本文件中动态“抽取”出指定的函数/类定义，
    # 并将其作为可导入的 Python 模块返回，从而在不直接运行整本笔记本的前提下，
    # 单独测试笔记本里实现的 RoPE、SiLU、RMSNorm 等组件
    def import_definitions_from_notebook(notebooks):
        imported_modules = {}

        for fullname, names in notebooks.items():
            # Get the directory of the current test file
            current_dir = os.path.dirname(__file__)
            path = os.path.join(current_dir, "..", fullname + ".ipynb")
            path = os.path.normpath(path)

            # Load the notebook
            if not os.path.exists(path):
                raise FileNotFoundError(f"Notebook file not found at: {path}")
            # 若目标笔记本文件不存在，直接抛出异常，避免后续静默失败

            with io.open(path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # Create a module to store the imported functions and classes
            mod = types.ModuleType(fullname)
            sys.modules[fullname] = mod
            # 创建一个空的动态模块，并注册到 sys.modules，
            # 这样后面 exec 执行的代码块中如果引用了模块级名字，也能正确解析

            # Go through the notebook cells and only execute function or class definitions
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell_code = cell.source
                    for name in names:
                        # Check for function or class definitions
                        if f"def {name}" in cell_code or f"class {name}" in cell_code:
                            exec(cell_code, mod.__dict__)
                            # 只有当某个代码单元格中包含目标函数/类定义时才执行该单元格，
                            # 避免执行笔记本中无关的示例代码（如打印、下载数据等）

            imported_modules[fullname] = mod

        return imported_modules

    notebooks = {
        "converting-gpt-to-llama2": ["SiLU", "RMSNorm", "precompute_rope_params", "compute_rope"],
        "converting-llama2-to-llama3": ["precompute_rope_params"]
    }
    # 声明需要从哪些笔记本中提取哪些函数/类：
    # - converting-gpt-to-llama2.ipynb 提供 Llama2 版本的 RoPE 及 SiLU、RMSNorm
    # - converting-llama2-to-llama3.ipynb 提供 Llama3（含 3.1/3.2 频率缩放）版本的 RoPE

    return import_definitions_from_notebook(notebooks)


@pytest.fixture(autouse=True)
def set_seed():
    # 自动应用于每个测试函数的 fixture：固定随机种子，保证每次运行测试时
    # 生成的随机 queries/keys 张量完全一致，从而使数值对比结果可复现
    torch.manual_seed(123)


def test_rope_llama2(notebook):
    """
    测试 Llama 2 版本的 RoPE 实现。

    通过与 HuggingFace transformers 的 LlamaRotaryEmbedding/apply_rotary_pos_emb
    以及 LitGPT 的参考实现（litgpt_build_rope_cache/litgpt_apply_rope）分别对比，
    验证自定义 precompute_rope_params + compute_rope 生成的 cos/sin 缓存，
    以及对 queries、keys 应用旋转位置编码后的结果在数值上完全一致
    （theta_base=10000，即标准 Llama 2 配置，不含任何频率缩放）。
    """

    this_nb = notebook["converting-gpt-to-llama2"]

    # Settings
    batch_size = 1
    context_len = 4096
    num_heads = 4
    head_dim = 16
    theta_base = 10_000

    # Instantiate RoPE parameters
    cos, sin = this_nb.precompute_rope_params(head_dim=head_dim, context_length=context_len)
    # 调用笔记本中自定义实现的 precompute_rope_params，预先计算好 cos/sin 位置编码表

    # Dummy query and key tensors
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)
    # 构造随机的 queries/keys 张量作为测试输入（种子已被 set_seed fixture 固定）

    # Apply rotary position embeddings
    queries_rot = this_nb.compute_rope(queries, cos, sin)
    keys_rot = this_nb.compute_rope(keys, cos, sin)
    # 用自定义实现对 queries/keys 应用旋转位置编码

    # Generate reference RoPE via HF

    if version.parse(transformers_version) < version.parse("4.48"):
        # transformers 旧版本 API：直接传 dim/max_position_embeddings/base 构造
        rot_emb = LlamaRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=context_len,
            base=theta_base
        )
    else:
        # transformers 新版本 API 改为要求传入一个 config 对象，这里构造一个满足接口的最小配置类
        class RoPEConfig:
            dim: int = head_dim
            rope_theta = theta_base
            max_position_embeddings: int = 8192
            hidden_size = head_dim * num_heads
            num_attention_heads = num_heads
            rope_parameters = {"rope_type": "default", "rope_theta": theta_base}

            def standardize_rope_params(self):
                return

        config = RoPEConfig()
        rot_emb = LlamaRotaryEmbedding(config=config)

    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)
    # 用 HuggingFace 官方实现生成参考的 cos/sin 缓存，并对相同的 queries/keys 应用旋转编码
    torch.testing.assert_close(sin, ref_sin.squeeze(0))  # 校验自定义 sin 缓存与 HF 参考 sin 缓存数值一致
    torch.testing.assert_close(cos, ref_cos.squeeze(0))  # 校验自定义 cos 缓存与 HF 参考 cos 缓存数值一致
    torch.testing.assert_close(keys_rot, ref_keys_rot)  # 校验旋转编码后的 keys 与 HF 参考结果一致
    torch.testing.assert_close(queries_rot, ref_queries_rot)  # 校验旋转编码后的 queries 与 HF 参考结果一致

    # Generate reference RoPE via LitGPT
    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(context_len, n_elem=head_dim, base=10_000)
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)
    # 再用 LitGPT 的独立实现生成第二份参考结果，进一步交叉验证

    torch.testing.assert_close(sin, litgpt_sin)  # 校验自定义 sin 缓存与 LitGPT 参考 sin 缓存数值一致
    torch.testing.assert_close(cos, litgpt_cos)  # 校验自定义 cos 缓存与 LitGPT 参考 cos 缓存数值一致
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)  # 校验旋转编码后的 keys 与 LitGPT 参考结果一致
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)  # 校验旋转编码后的 queries 与 LitGPT 参考结果一致


def test_rope_llama3(notebook):
    """
    测试 Llama 3 版本的 RoPE 实现。

    与 test_rope_llama2 思路一致，区别在于 Llama 3 使用了更大的
    theta_base=500000（而非 Llama 2 的 10000），以配合更长的上下文长度
    （context_len=8192）。同样分别对照 HuggingFace 与 LitGPT 两种参考实现，
    验证 cos/sin 缓存及旋转编码后的 queries/keys 数值完全一致。
    """

    nb1 = notebook["converting-gpt-to-llama2"]
    nb2 = notebook["converting-llama2-to-llama3"]

    # Settings
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    theta_base = 500_000

    # Instantiate RoPE parameters
    cos, sin = nb2.precompute_rope_params(
        head_dim=head_dim,
        context_length=context_len,
        theta_base=theta_base
    )
    # 使用 Llama3 笔记本中的 precompute_rope_params（支持传入更大的 theta_base）

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = nb1.compute_rope(queries, cos, sin)
    keys_rot = nb1.compute_rope(keys, cos, sin)
    # compute_rope 的应用逻辑与 Llama2 版本共用（沿用 nb1 中的实现）

    # Generate reference RoPE via HF
    if version.parse(transformers_version) < version.parse("4.48"):
        rot_emb = LlamaRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=context_len,
            base=theta_base
        )
    else:
        class RoPEConfig:
            dim: int = head_dim
            rope_theta = theta_base
            max_position_embeddings: int = 8192
            hidden_size = head_dim * num_heads
            num_attention_heads = num_heads
            rope_parameters = {"rope_type": "default", "rope_theta": theta_base}

            def standardize_rope_params(self):
                return

        config = RoPEConfig()
        rot_emb = LlamaRotaryEmbedding(config=config)

    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))  # 校验自定义 sin 缓存与 HF 参考 sin 缓存数值一致（theta_base=500000）
    torch.testing.assert_close(cos, ref_cos.squeeze(0))  # 校验自定义 cos 缓存与 HF 参考 cos 缓存数值一致
    torch.testing.assert_close(keys_rot, ref_keys_rot)  # 校验旋转编码后的 keys 与 HF 参考结果一致
    torch.testing.assert_close(queries_rot, ref_queries_rot)  # 校验旋转编码后的 queries 与 HF 参考结果一致

    # Generate reference RoPE via LitGPT
    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(context_len, n_elem=head_dim, base=theta_base)
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    torch.testing.assert_close(sin, litgpt_sin)  # 校验自定义 sin 缓存与 LitGPT 参考 sin 缓存数值一致
    torch.testing.assert_close(cos, litgpt_cos)  # 校验自定义 cos 缓存与 LitGPT 参考 cos 缓存数值一致
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)  # 校验旋转编码后的 keys 与 LitGPT 参考结果一致
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)  # 校验旋转编码后的 queries 与 LitGPT 参考结果一致


def test_rope_llama3_12(notebook):
    """
    测试 Llama 3.1 / 3.2 版本的 RoPE 实现（带 NTK-aware 频率缩放）。

    相较于标准 Llama 3 RoPE，这里额外引入了 freq_config（factor、
    low_freq_factor、high_freq_factor、original_context_length），
    用于在“低频分量做外推缩放”和“高频分量保持不变”之间做平滑插值，
    从而在不重新训练的情况下将模型的有效上下文长度扩展到更大范围。
    测试方式依旧是分别与 HuggingFace（rope_type="llama3"）以及 LitGPT
    （通过 extra_config 参数）两种参考实现做数值比对。
    """

    nb1 = notebook["converting-gpt-to-llama2"]
    nb2 = notebook["converting-llama2-to-llama3"]

    # Settings
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    rope_theta = 500_000

    rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
    # 自定义实现所使用的频率缩放配置：factor 控制整体缩放倍数，
    # low_freq_factor/high_freq_factor 划定平滑插值的边界

    # Instantiate RoPE parameters
    cos, sin = nb2.precompute_rope_params(
        head_dim=head_dim,
        theta_base=rope_theta,
        context_length=context_len,
        freq_config=rope_config,
    )
    # 传入 freq_config 即可启用 Llama 3.1/3.2 风格的频率缩放版 RoPE

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = nb1.compute_rope(queries, cos, sin)
    keys_rot = nb1.compute_rope(keys, cos, sin)

    # Generate reference RoPE via HF
    hf_rope_params = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    }
    # HuggingFace 侧对应的频率缩放参数，字段名与自定义实现略有不同
    # （original_max_position_embeddings vs original_context_length），
    # 但含义完全一致，用于确保两边参数语义对齐

    class RoPEConfig:
        rope_type = "llama3"
        rope_scaling = hf_rope_params
        factor = 1.0
        dim: int = head_dim
        rope_theta = 500_000
        max_position_embeddings: int = 8192
        hidden_size = head_dim * num_heads
        num_attention_heads = num_heads
        rope_parameters = {**hf_rope_params, "rope_theta": rope_theta}

        def standardize_rope_params(self):
            return

    config = RoPEConfig()
    # 构造 HuggingFace LlamaRotaryEmbedding 所需的 config，
    # rope_type="llama3" 会触发官方实现内部对应的频率缩放逻辑

    rot_emb = LlamaRotaryEmbedding(config=config)
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    torch.testing.assert_close(sin, ref_sin.squeeze(0))  # 校验带频率缩放的自定义 sin 缓存与 HF 参考一致
    torch.testing.assert_close(cos, ref_cos.squeeze(0))  # 校验带频率缩放的自定义 cos 缓存与 HF 参考一致
    torch.testing.assert_close(keys_rot, ref_keys_rot)  # 校验旋转编码后的 keys 与 HF 参考结果一致
    torch.testing.assert_close(queries_rot, ref_queries_rot)  # 校验旋转编码后的 queries 与 HF 参考结果一致

    # Generate reference RoPE via LitGPT
    litgpt_rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_seq_len": 8192
    }
    # LitGPT 侧对应的频率缩放参数（字段名 original_max_seq_len），
    # 会传给 litgpt_build_rope_cache 的 extra_config 参数触发平滑插值逻辑

    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(
        context_len,
        n_elem=head_dim,
        base=rope_theta,
        extra_config=litgpt_rope_config
    )
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    torch.testing.assert_close(sin, litgpt_sin)  # 校验带频率缩放的自定义 sin 缓存与 LitGPT 参考一致
    torch.testing.assert_close(cos, litgpt_cos)  # 校验带频率缩放的自定义 cos 缓存与 LitGPT 参考一致
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)  # 校验旋转编码后的 keys 与 LitGPT 参考结果一致
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)  # 校验旋转编码后的 queries 与 LitGPT 参考结果一致


def test_silu(notebook):
    """
    测试自定义 SiLU（Sigmoid Linear Unit，也称 Swish）激活函数的正确性。

    直接与 PyTorch 官方的 torch.nn.functional.silu 做逐元素数值比对，
    验证自定义实现的公式 x * sigmoid(x) 计算无误。
    """
    example_batch = torch.randn(2, 3, 4)
    silu = notebook["converting-gpt-to-llama2"].SiLU()
    assert torch.allclose(silu(example_batch), torch.nn.functional.silu(example_batch))
    # 断言自定义 SiLU 的输出与 PyTorch 官方实现在数值上（默认容差内）完全一致


@pytest.mark.skipif(torch.__version__ < "2.4", reason="Requires PyTorch 2.4 or newer")
def test_rmsnorm(notebook):
    """
    测试自定义 RMSNorm（Root Mean Square Layer Normalization）实现的正确性。

    与 PyTorch 2.4+ 内置的 torch.nn.RMSNorm 做数值比对；若当前 PyTorch
    版本低于 2.4（不含官方 RMSNorm 实现），则自动跳过该测试。
    """
    example_batch = torch.randn(2, 3, 4)
    rms_norm = notebook["converting-gpt-to-llama2"].RMSNorm(emb_dim=example_batch.shape[-1], eps=1e-5)
    rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)

    assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))
    # 断言自定义 RMSNorm 的输出与 PyTorch 官方 RMSNorm 在数值上（默认容差内）完全一致
