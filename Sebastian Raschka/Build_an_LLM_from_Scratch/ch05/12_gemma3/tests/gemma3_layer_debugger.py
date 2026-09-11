"""Gemma3 逐层调试器模块。

本模块用于对比「自实现的 Gemma3Model」（来自配套 notebook `standalone-gemma3.ipynb`）
与 HuggingFace `transformers` 官方参考实现（`Gemma3ForCausalLM`）在**同一份权重、同一份输入**下，
逐层（embedding -> 每个 transformer block -> final_norm -> logits）产生的中间激活值是否一致。

核心调试思路：
    1. 构造一对模型：自实现模型 `ours` 与 HF 参考模型 `hf_model`，并保证二者权重完全对齐
       （通过 `load_weights_into_gemma` 把 HF 的 state_dict 灌入自实现模型）。
    2. 用 forward hook 分别在两个模型的关键子模块（embedding 层、每个 block、最终归一化层、
       输出头）上挂钩子，拦截它们的输出张量，存入字典 `traces`。
    3. 用同一份输入 `input_ids` 分别跑一次前向传播，两边的 hook 会记录下每一层的输出。
    4. 按层名对齐后逐层比较两边激活的形状与数值误差（最大绝对误差、平均绝对误差、
       是否在给定容差内 allclose），从而定位「自实现模型从哪一层开始与参考实现出现偏差」。
    5. `format_report` 把上述比较结果格式化成人类可读的文本报告，标记每层是 OK / DIFF /
       SHAPE（形状不匹配）/ MISSING（某一侧缺失该层输出）。

这种「逐层激活值比对」是调试从零实现的 LLM 是否与官方实现等价的标准手段：
一旦发现某一层开始 DIFF，就可以把排查范围缩小到该层的具体实现（例如 RoPE 频率、
QK-Norm、滑动窗口注意力、embedding 缩放系数等 Gemma3 特有细节）。
"""

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import torch

from llms_from_scratch.utils import import_definitions_from_notebook

try:
    # 优先尝试导入 HuggingFace transformers 中的 Gemma3 官方实现，
    # 作为本调试器的「参考基准」（ground truth）。
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
except ImportError:
    # 如果环境中没有安装 transformers，则将两个引用置空，
    # 后续用到时会显式抛出 ImportError 提示用户安装依赖。
    Gemma3ForCausalLM = None
    Gemma3TextConfig = None


def tiny_debug_config():
    """返回一份体积极小的 Gemma3 调试用配置字典。

    该配置刻意把词表、层数、隐藏维度等都设置得很小（vocab_size=257、n_layers=2 等），
    目的是让「自实现模型」与「HF 参考模型」都能快速构建、快速前向传播，
    从而可以在几秒内完成一次完整的逐层激活值比对，便于调试。
    """
    return {
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


def _hf_config_from_dict(cfg):
    """把自实现模型所用的配置字典 `cfg` 转换成 HuggingFace 的 `Gemma3TextConfig` 对象。

    这样可以保证在「没有提供预训练权重」的情况下，随机初始化的 HF 模型
    与自实现模型使用完全对齐的超参数（层数、头数、RoPE 基数、滑动窗口大小等），
    从而使二者在结构上具备可比性。
    """
    if Gemma3TextConfig is None:
        # 未安装 transformers 时无法构造 HF 配置，直接报错提示。
        raise ImportError("transformers is required for the Gemma 3 debugger.")

    return Gemma3TextConfig(
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
        attn_implementation="eager",  # 使用 eager 实现而非融合 kernel，方便逐层 hook 拦截
        torch_dtype=cfg.get("dtype", torch.float32),
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
        rope_scaling={"rope_type": "default"},
        ignore_keys_at_rope_validation={"full_attention", "sliding_attention"},
    )


def load_notebook_defs(nb_name="standalone-gemma3.ipynb"):
    """从配套 notebook 中动态导入自实现的 Gemma3 相关定义（模型类、权重加载函数等）。

    `nb_dir` 指向本文件所在目录的上一级目录（即 `ch05/12_gemma3/`），
    默认从中加载 `standalone-gemma3.ipynb`，这样调试脚本无需把 notebook 里的
    代码手动复制成 .py 文件，而是直接复用 notebook 中已经写好、验证过的实现。
    """
    nb_dir = Path(__file__).resolve().parents[1]
    return import_definitions_from_notebook(nb_dir, nb_name)


def build_gemma3_pair(import_notebook_defs, cfg, hf_checkpoint=None):
    """构建一对可直接比较的模型：自实现的 `Gemma3Model` 与 HF 的 `Gemma3ForCausalLM`。

    关键步骤（这是保证「逐层比对」有意义的前提）：
        1. 先用 `cfg` 实例化自实现模型 `ours`。
        2. 构建 HF 模型：如果提供了 `hf_checkpoint`（预训练权重路径/名称），
           则直接从预训练权重加载；否则根据 `cfg` 随机初始化一个同结构的 HF 模型。
        3. 调用 `load_weights_into_gemma`，把 HF 模型的 `state_dict`（无论是预训练还是随机初始化的）
           按照命名映射规则灌入自实现模型 `ours`，确保两个模型的权重完全一致。
        4. 都切换到 `eval()` 模式，关闭 dropout 等训练态行为，保证前向传播结果可复现、可比较。

    只有权重严格对齐之后，后续逐层比较激活值的差异才能真实反映「实现逻辑」的差异，
    而不是「权重不同」造成的差异。
    """
    if Gemma3ForCausalLM is None:
        raise ImportError("transformers is required for the Gemma 3 debugger.")

    ours = import_notebook_defs.Gemma3Model(cfg)

    if hf_checkpoint:
        # 如果指定了具体的预训练权重仓库/路径，直接加载真实的 Gemma3 权重。
        hf_model = Gemma3ForCausalLM.from_pretrained(
            hf_checkpoint,
            torch_dtype=cfg.get("dtype", torch.float32),
            attn_implementation="eager",
        )
    else:
        # 没有指定预训练权重时，根据 cfg 构造一个结构相同、权重随机初始化的 HF 模型，
        # 随后会把它的随机权重同步灌入自实现模型，只比较「实现逻辑」是否一致。
        hf_cfg = _hf_config_from_dict(cfg)
        hf_model = Gemma3ForCausalLM(hf_cfg)

    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    # 把 HF 模型的权重按照自实现模型的参数命名规则拷贝过去，
    # 这是保证两个模型「同权重、只比实现」的关键一步。
    import_notebook_defs.load_weights_into_gemma(ours, param_config, hf_model.state_dict())

    ours.eval()
    hf_model.eval()
    return ours, hf_model


def _attach_debug_hooks(model, is_hf):
    """在给定模型的关键子模块上注册 forward hook，用于捕获逐层输出激活值。

    这是「逐层调试」的核心机制：PyTorch 的 `register_forward_hook` 允许我们在
    不修改模型代码的前提下，拦截某个子模块 forward 计算完成后的输出张量。
    本函数分别处理两种模型结构（HF 官方实现 vs. 自实现），把它们的
    embedding 层、每一个 transformer block、最终归一化层（final_norm）、
    输出头（lm_head / out_head）都挂上同名的 hook，方便之后按「层名」对齐比较。

    参数:
        model: 待挂钩子的模型实例（HF 模型或自实现模型）。
        is_hf: 布尔值，标记 `model` 是 HuggingFace 官方实现（True）还是自实现（False），
               因为两者的子模块命名/结构不同，需要分别处理。

    返回:
        traces: 字典，键为层名（如 "embedding"、"block_0"、"final_norm"、"logits"），
                值为该层输出的张量（已 detach、转 float32、搬到 CPU，避免影响后续计算图）。
        handles: hook 句柄列表，调用方在比较完成后需要手动 `remove()` 释放，避免内存泄漏
                 或影响后续对同一模型的其他前向调用。
    """
    traces = {}
    handles = []

    def hook(name, scale=None):
        """构造一个具名的 forward hook 闭包，把输出张量记录到 `traces[name]`。

        `scale` 参数用于处理 Gemma 系列模型的一个特殊细节：
        自实现的 embedding 层输出未做缩放，而 Gemma 官方做法是在 embedding 之后
        乘以 sqrt(emb_dim) 的缩放系数；为了让两侧在同一层名下可比较，
        这里允许在记录自实现 embedding 输出时手动乘上该缩放系数。
        """

        def _record(_, __, output):
            if isinstance(output, tuple):
                # 有些子模块（如 transformer block）的输出是元组（如 (hidden_states, ...)），
                # 这里只取第一个元素，即真正的隐藏状态张量。
                output = output[0]
            if scale is not None:
                # 对齐 Gemma 的 embedding 缩放约定（乘以 sqrt(emb_dim)）。
                output = output * scale
            # detach 阻断梯度追踪，统一转为 float32 并搬到 CPU，方便跨模型、跨精度做数值比较。
            traces[name] = output.detach().to(torch.float32).cpu()

        return _record

    if is_hf:
        # HuggingFace Gemma3ForCausalLM 的结构：model.model 是骨干网络（含 embed_tokens、
        # layers、norm），model.lm_head 是语言模型输出头。
        core = model.model
        handles.append(core.embed_tokens.register_forward_hook(hook("embedding")))
        for idx, layer in enumerate(core.layers):
            # 逐层挂钩子，层名统一为 "block_{idx}"，与自实现模型的层名对齐，方便后续比较。
            handles.append(layer.register_forward_hook(hook(f"block_{idx}")))
        handles.append(core.norm.register_forward_hook(hook("final_norm")))
        handles.append(model.lm_head.register_forward_hook(hook("logits")))
    else:
        # 自实现模型：从 cfg 中取 emb_dim（若取不到则退回 tok_emb.embedding_dim），
        # 计算 embedding 缩放系数 sqrt(emb_dim)，因为自实现的 tok_emb 输出本身未缩放，
        # 需要在记录时手动补上缩放，才能与 HF 侧（内部已做缩放）的 embedding 输出对齐比较。
        emb_scale = float(getattr(model, "cfg", {}).get("emb_dim", model.tok_emb.embedding_dim) ** 0.5)
        handles.append(model.tok_emb.register_forward_hook(hook("embedding", scale=emb_scale)))
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            # 兼容不同版本自实现模型对 transformer block 列表属性的命名（blocks 或 trf_blocks）。
            blocks = getattr(model, "trf_blocks", None)
        if blocks is None:
            raise AttributeError("Could not locate Gemma 3 blocks on the local model.")
        for idx, block in enumerate(blocks):
            handles.append(block.register_forward_hook(hook(f"block_{idx}")))
        handles.append(model.final_norm.register_forward_hook(hook("final_norm")))
        handles.append(model.out_head.register_forward_hook(hook("logits")))

    return traces, handles


def _layer_sort_key(name):
    """为层名生成排序键，保证报告输出时严格按照
    embedding -> block_0 -> block_1 -> ... -> final_norm -> logits 的模型前向顺序排列，
    而不是按字符串默认排序（那样会把 block_10 排在 block_2 前面等问题）。
    """
    if name == "embedding":
        return (0, 0)
    if name.startswith("block_"):
        idx = int(name.split("_")[1])
        return (1, idx)
    if name == "final_norm":
        return (2, 0)
    if name == "logits":
        return (3, 0)
    return (4, name)


def layerwise_differences(ours, hf_model, input_ids, rtol=1e-5, atol=1e-5):
    """对「自实现模型」与「HF 参考模型」执行同一份输入的前向传播，并逐层比较激活值差异。

    这是本调试器的主流程函数，完整的调试思路如下：
        1. 分别给 `ours` 和 `hf_model` 挂上 forward hook（见 `_attach_debug_hooks`），
           以便捕获两者在 embedding / 各 block / final_norm / logits 处的输出。
        2. 用 `torch.inference_mode()` 关闭梯度追踪，对同一份 `input_ids` 分别跑一次前向传播——
           必须用完全相同的输入，才能保证两侧激活值的差异只来自「实现逻辑」而非输入不同。
        3. 无论比较是否发生异常，都要在 `finally` 块中移除所有 hook 句柄，
           避免 hook 残留影响后续对模型的其他调用（这是使用 forward hook 时的标准防御写法）。
        4. 取两侧记录到的层名并集，按照 `_layer_sort_key` 排序后逐层比较：
           - 若某一层只在一侧存在（缺失），标记为 "missing"；
           - 若两侧形状不一致，标记为 "shape_mismatch"；
           - 否则计算逐元素绝对差值的最大值/均值，并用 `torch.allclose` 判断是否在给定的
             相对容差 `rtol` 与绝对容差 `atol` 内一致，标记为 "ok" 或 "mismatch"。

    通过观察从哪一层开始首次出现 "mismatch"，即可快速定位自实现代码中与参考实现
    行为不一致的具体环节（例如某个 block 内部的注意力计算、RoPE 应用方式等）。

    参数:
        ours: 自实现的 Gemma3Model 实例。
        hf_model: HuggingFace 的 Gemma3ForCausalLM 参考实现实例。
        input_ids: 用于两侧模型前向传播的同一份输入 token id 张量。
        rtol, atol: 传给 `torch.allclose` 的相对/绝对容差，用于判定两层激活是否「足够接近」。

    返回:
        results: 列表，每个元素是一个描述某一层比较结果的字典
                 （包含 name/status/ours_shape/hf_shape/max_diff/mean_abs_diff）。
    """
    ours_traces, ours_handles = _attach_debug_hooks(ours, is_hf=False)
    hf_traces, hf_handles = _attach_debug_hooks(hf_model, is_hf=True)

    try:
        with torch.inference_mode():
            # 用同一份 input_ids 依次跑两个模型的前向传播，触发上面注册的所有 hook，
            # 从而把每一层的输出激活值分别记录进 ours_traces 与 hf_traces。
            ours(input_ids)
            hf_model(input_ids)
    finally:
        # 无论前向传播是否抛异常，都要移除全部 hook，避免影响调用方后续对这两个模型的使用。
        for h in ours_handles + hf_handles:
            h.remove()

    # 取两侧记录到的层名并集（理论上应完全一致），按前向顺序排序，逐层比较。
    layer_names = sorted(set(ours_traces) | set(hf_traces), key=_layer_sort_key)
    results = []
    for name in layer_names:
        ours_tensor = ours_traces.get(name)
        hf_tensor = hf_traces.get(name)

        if ours_tensor is None or hf_tensor is None:
            # 只有一侧记录到了该层的输出，说明两模型结构不对齐（例如层数不同），
            # 直接标记为 missing，不再做数值比较。
            results.append(
                {
                    "name": name,
                    "status": "missing",
                    "ours_shape": None if ours_tensor is None else tuple(ours_tensor.shape),
                    "hf_shape": None if hf_tensor is None else tuple(hf_tensor.shape),
                    "max_diff": None,
                    "mean_abs_diff": None,
                }
            )
            continue

        if ours_tensor.shape != hf_tensor.shape:
            # 形状不一致时无法逐元素比较，直接报告形状不匹配。
            results.append(
                {
                    "name": name,
                    "status": "shape_mismatch",
                    "ours_shape": tuple(ours_tensor.shape),
                    "hf_shape": tuple(hf_tensor.shape),
                    "max_diff": None,
                    "mean_abs_diff": None,
                }
            )
            continue

        # 形状一致时，计算逐元素绝对误差的最大值与均值，作为量化的「偏差程度」指标，
        # 同时用 allclose 给出是否通过容差阈值的布尔判定。
        diff = (ours_tensor - hf_tensor).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        allclose = torch.allclose(ours_tensor, hf_tensor, rtol=rtol, atol=atol)
        results.append(
            {
                "name": name,
                "status": "ok" if allclose else "mismatch",
                "ours_shape": tuple(ours_tensor.shape),
                "hf_shape": tuple(hf_tensor.shape),
                "max_diff": max_diff,
                "mean_abs_diff": mean_diff,
            }
        )
    return results


def format_report(differences):
    """把 `layerwise_differences` 返回的逐层比较结果格式化为易读的多行文本报告。

    每一行对应一层，前缀标签含义：
        [OK]      两侧该层激活在给定容差内一致；
        [DIFF]    两侧该层激活数值上存在超出容差的差异（附最大/平均绝对误差）；
        [SHAPE]   两侧该层输出形状不一致，无法数值比较；
        [MISSING] 该层只在一侧模型中被记录到输出。

    调试时通常从上到下查看该报告，第一处出现 [DIFF]/[SHAPE]/[MISSING] 的层，
    就是自实现代码与参考实现开始产生偏差的位置，可据此进一步深入排查。
    """
    lines = []
    for diff in sorted(differences, key=lambda d: _layer_sort_key(d["name"])):
        if diff["status"] == "ok":
            lines.append(f"[OK] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}")
        elif diff["status"] == "mismatch":
            lines.append(f"[DIFF] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}")
        elif diff["status"] == "shape_mismatch":
            lines.append(f"[SHAPE] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}")
        else:
            lines.append(f"[MISSING] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 脚本入口：检查 transformers 是否已安装（因为参考实现依赖它），
    # 未安装则直接退出并提示安装。
    transformers_available = importlib.util.find_spec("transformers") is not None
    if not transformers_available:
        raise SystemExit("transformers is not installed; install it to run the debugger.")

    # 从 notebook 动态加载自实现的 Gemma3 相关定义（模型类、权重加载函数）。
    import_notebook_defs = load_notebook_defs()
    # 使用极小规模的调试配置，保证脚本能快速跑完一次完整对比。
    cfg = tiny_debug_config()

    # 构建权重对齐的自实现模型与 HF 参考模型。
    ours_model, hf_model = build_gemma3_pair(import_notebook_defs, cfg)
    torch.manual_seed(0)  # 固定随机种子，保证输入 token 序列可复现
    input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), dtype=torch.long)
    # 执行逐层前向传播与激活值比较。
    diffs = layerwise_differences(ours_model, hf_model, input_ids)
    # 打印格式化后的逐层比对报告，用于人工检查从哪一层开始出现偏差。
    print(format_report(diffs))
