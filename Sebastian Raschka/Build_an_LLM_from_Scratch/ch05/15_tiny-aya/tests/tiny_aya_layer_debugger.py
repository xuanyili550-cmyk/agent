# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""Tiny Aya (Cohere2) 逐层调试脚本。

本模块的作用：构建一对"参数完全对齐"的模型——
    1) `ours`：notebook（standalone-tiny-aya.ipynb）里手写的 TinyAyaModel 实现；
    2) `hf_model`：HuggingFace transformers 官方的 Cohere2ForCausalLM 实现，
       并把同一份权重（要么随机初始化、要么从真实 checkpoint 加载）灌入两边。

然后对同一个输入 `input_ids` 分别跑一次前向传播，利用 forward hook 把
embedding / 每个 Transformer block / 最终 LayerNorm / 输出头（logits）
这几个关键节点的输出张量都"拍照"记录下来，逐层比较两边的数值是否一致
（在给定的相对/绝对误差范围内）。这样如果手写实现有 bug，可以立刻定位到
是哪一层第一次出现了偏差，而不用等到最终 logits 出现明显不对时再去排查。

简言之：这是一个"手写实现 vs 官方实现"的数值对拍（diff）工具，用于验证
从零实现的 Tiny Aya 模型是否与 HuggingFace 官方 Cohere2 模型完全等价。
"""

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import torch
from llms_from_scratch.utils import import_definitions_from_notebook

try:
    # transformers 是可选依赖：只有装了它才能构造官方的 Cohere2 模型用于对拍。
    from transformers import Cohere2Config, Cohere2ForCausalLM
except ImportError:
    # 没装 transformers 时把两个符号置空，后面用到的地方会显式抛出更友好的错误提示。
    Cohere2Config = None
    Cohere2ForCausalLM = None


def tiny_debug_config():
    """返回一个"迷你"版的 Tiny Aya 配置字典，专门用于调试/单测。

    真实的 Tiny Aya（Cohere2 风格）模型体积很大（vocab 26 万+、36 层等），
    直接拿来做逐层数值比对既慢又占内存。这里把各维度都缩小到个位数/几十的
    量级（vocab_size=257、emb_dim=32、n_layers=2 ...），但保留了模型结构上
    所有关键特性：
      - 分组查询注意力（n_heads=4, n_kv_heads=2）；
      - 滑动窗口 + 全局注意力交替的层类型（layer_types）；
      - RoPE 位置编码、logit 缩放等。
    这样跑起来又快，又能覆盖到需要验证的所有分支逻辑。
    """
    return {
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


def _hf_config_from_dict(cfg):
    """把我们自己风格的配置字典 `cfg` 转换成 HuggingFace 的 `Cohere2Config`。

    这里做的其实是"字段名翻译"：我们自己的命名（如 emb_dim、n_heads）
    和 HF 官方命名（hidden_size、num_attention_heads）不一样，需要逐个
    对应起来，确保两边模型的结构参数（层数、头数、滑窗大小等）完全一致，
    这样后面权重复制和逐层输出比较才有意义。
    """
    if Cohere2Config is None:
        # 提前给出清晰的报错信息，而不是等到下面用 Cohere2Config(...) 时
        # 才因为 None 不可调用而抛出一个让人摸不着头脑的 TypeError。
        raise ImportError("transformers is required for the Tiny Aya debugger.")

    return Cohere2Config(
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
        sliding_window=cfg["sliding_window"],
        layer_types=cfg["layer_types"],
        logit_scale=cfg["logit_scale"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        rope_parameters={"rope_type": "default", "rope_theta": cfg["rope_base"]},
        torch_dtype=cfg.get("dtype", torch.float32),
        # --- bug 修复 ---
        # 原代码：这里没有传 attn_implementation="eager"。
        # 为什么是 bug：本文件里另一条路径（build_tiny_aya_pair 中从真实
        # checkpoint 加载时）显式指定了 attn_implementation="eager"；
        # 同一目录下其它模型的调试脚本（olmo3 / qwen3.5 / gemma3 / gemma4
        # 的 layer_debugger.py）在构造 HF Config 时也都统一强制使用 eager
        # 实现。如果这里不设置，HF 在"从零构造模型"（没有传 checkpoint）
        # 这条分支上会退回默认的 attention 后端（如 SDPA / 融合 kernel），
        # 其数值结果与我们手写实现里显式构造的 mask + 手动矩阵乘法注意力
        # 并不保证逐元素一致，会让本工具"逐层比较数值是否相等"这个核心目的
        # 失效（可能报出假的 mismatch，或者对于自定义的滑窗/全局布尔 mask
        # 支持不一致而报错）。因此在此显式加上 eager，使前向计算路径与
        # 手写实现完全对齐，也与本文件内另一条构造路径保持一致。
        attn_implementation="eager",
    )


def load_notebook_defs(nb_name="standalone-tiny-aya.ipynb"):
    """从同目录上一级的 notebook 文件中动态导入手写实现的类/函数。

    `parents[1]` 是因为本文件位于 `.../15_tiny-aya/tests/xxx.py`，
    上一级目录 `15_tiny-aya` 才是 notebook 所在的位置。
    返回的对象上会挂载 notebook 里定义的 TinyAyaModel、
    load_weights_into_tiny_aya 等符号，供后面直接调用。
    """
    nb_dir = Path(__file__).resolve().parents[1]
    return import_definitions_from_notebook(nb_dir, nb_name)


def build_tiny_aya_pair(import_notebook_defs, cfg, hf_checkpoint=None):
    """构建"手写实现"与"HF 官方实现"这一对模型，并让二者权重完全一致。

    参数:
        import_notebook_defs: `load_notebook_defs()` 返回的模块对象，
            里面包含 notebook 中定义的 TinyAyaModel 类和权重加载函数。
        cfg: 模型结构配置字典（通常是 `tiny_debug_config()` 的返回值）。
        hf_checkpoint: 若提供（例如 HF Hub 上的仓库名或本地路径），
            则加载真实预训练权重的 HF 模型；否则用随机初始化的权重
            （这种情况下我们自己的模型权重会被硬拷贝成与 HF 模型一致，
            从而排除"初始化不同导致数值不同"的干扰，只测试计算逻辑本身）。

    返回:
        (ours, hf_model): 均已切换到 eval() 模式、权重严格对齐的一对模型。
    """
    if Cohere2ForCausalLM is None:
        raise ImportError("transformers is required for the Tiny Aya debugger.")

    # 先按配置实例化手写版模型（此时权重是随机初始化的，稍后会被覆盖）。
    ours = import_notebook_defs.TinyAyaModel(cfg)
    hf_cfg = _hf_config_from_dict(cfg)

    if hf_checkpoint:
        # 从真实预训练权重加载 HF 模型；显式指定 eager 注意力实现，
        # 避免融合 kernel（如 flash-attn/SDPA）带来的浮点误差影响逐层对拍。
        hf_model = Cohere2ForCausalLM.from_pretrained(
            hf_checkpoint,
            torch_dtype=cfg.get("dtype", torch.float32),
            attn_implementation="eager",
        )
    else:
        # 没有指定 checkpoint 时，直接用（迷你）配置随机初始化一个 HF 模型。
        hf_model = Cohere2ForCausalLM(hf_cfg)

    # 核心步骤：把 HF 模型的权重（state_dict）按照命名映射关系，
    # 逐个张量拷贝进我们手写实现对应的参数里，确保两边权重逐比特一致。
    import_notebook_defs.load_weights_into_tiny_aya(ours, cfg, hf_model.state_dict())

    # 调试对拍要求确定性的前向计算，因此都切到 eval 模式，
    # 关闭 dropout 等随机性（本配置里 attention_dropout=0.0，但仍是好习惯）。
    ours.eval()
    hf_model.eval()
    return ours, hf_model


def _attach_debug_hooks(model, is_hf):
    """给模型的关键层挂上 forward hook，用于抓取每一层的输出张量。

    参数:
        model: 待挂钩子的模型（可以是我们手写的 TinyAyaModel，
            也可以是 HF 的 Cohere2ForCausalLM）。
        is_hf: 是否是 HF 官方模型。两边模型的属性命名不同
            （例如 HF 是 `model.model.layers`，我们是 `model.trf_blocks`），
            需要分支处理才能找到对应的子模块。

    返回:
        (traces, handles):
            traces: dict，key 是层名（如 "embedding"、"block_0"、
                "final_norm"、"logits"），value 是该层前向输出（已 detach、
                转 float32、搬到 CPU，方便后续用 torch.allclose 之类的
                函数直接比较，且不会持有计算图、不占用 GPU 显存）。
            handles: hook 句柄列表，调用方用完之后需要逐个 remove()，
                否则钩子会一直挂在模型上，多次调用会导致重复记录/内存泄漏。
    """
    traces = {}
    handles = []

    def hook(name):
        # 闭包工厂：为每个层名生成一个独立的 hook 函数，
        # 这样多个层可以复用同一套逻辑，只是把结果存到 traces[name] 里。
        def _record(_, __, output):
            # forward hook 的签名是 (module, input, output)，
            # 这里用 _ 和 __ 分别丢弃 module 和 input，只关心输出张量。
            # 注意：这里假设 output 一定是单个 Tensor（而不是 tuple）。
            # 风险提示（不改动，仅标注）：在当前安装的 transformers 版本
            # （5.5.0）中，Cohere2DecoderLayer.forward 确实只返回一个
            # Tensor，所以这里能正常工作；但其类型注解写的是
            # `tuple[FloatTensor, tuple|None]`，说明不同 transformers
            # 版本/不同配置下这个层的返回值格式可能会变成 tuple。
            # 如果未来升级 transformers 后返回值变成了 tuple，
            # `output.detach()` 会因为 tuple 没有 detach 方法而抛出
            # AttributeError。这是一个跨版本兼容性风险，不属于本次
            # "确定性 bug"，故按要求只标注、不修改。
            traces[name] = output.detach().to(torch.float32).cpu()

        return _record

    if is_hf:
        # HF Cohere2ForCausalLM 的子模块层级：
        #   model.model.embed_tokens  -> 词嵌入
        #   model.model.layers[i]     -> 第 i 个 Transformer decoder layer
        #   model.model.norm          -> 最终的 LayerNorm/RMSNorm
        #   model.lm_head             -> 输出投影到词表维度（得到 logits）
        core = model.model
        handles.append(core.embed_tokens.register_forward_hook(hook("embedding")))
        for idx, layer in enumerate(core.layers):
            handles.append(layer.register_forward_hook(hook(f"block_{idx}")))
        handles.append(core.norm.register_forward_hook(hook("final_norm")))
        handles.append(model.lm_head.register_forward_hook(hook("logits")))
    else:
        # 手写 TinyAyaModel 的子模块层级：
        #   model.tok_emb      -> 词嵌入
        #   model.trf_blocks[i]-> 第 i 个 Transformer block
        #   model.final_norm   -> 最终归一化层
        #   model.out_head     -> 输出投影头
        handles.append(model.tok_emb.register_forward_hook(hook("embedding")))
        blocks = getattr(model, "trf_blocks", None)
        if blocks is None:
            # 兼容性兜底：万一未来 notebook 里把属性名从 trf_blocks
            # 改成了更通用的 blocks，这里也能找到，不至于直接报错。
            blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise AttributeError("Could not locate Tiny Aya blocks on the local model.")
        for idx, block in enumerate(blocks):
            handles.append(block.register_forward_hook(hook(f"block_{idx}")))
        handles.append(model.final_norm.register_forward_hook(hook("final_norm")))
        handles.append(model.out_head.register_forward_hook(hook("logits")))

    return traces, handles


def _layer_sort_key(name):
    """为层名生成排序键，使得比较结果能按照"网络前向顺序"打印，而不是字典乱序。

    顺序约定：embedding(0) < block_0, block_1, ...(1, idx) < final_norm(2) < logits(3)；
    其它未知名字统一排在最后（4, name），按字符串排序兜底。
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
    """核心对拍逻辑：对两份模型跑同一个输入，逐层比较输出张量的数值差异。

    参数:
        ours: 手写实现模型。
        hf_model: HF 官方实现模型（权重已与 ours 对齐）。
        input_ids: 共同的输入 token id 张量，形状 [batch, seq_len]。
        rtol, atol: torch.allclose 使用的相对/绝对误差容忍度，
            用于判定两层输出是否"足够接近"（浮点计算不可能完全按位相等）。

    返回:
        results: list[dict]，每个元素描述一个层的比较结果，
            status 取值有四种：
              - "ok"：两边输出形状一致且数值在容差范围内。
              - "mismatch"：形状一致但数值超出容差（说明该层实现有问题）。
              - "shape_mismatch"：两边输出形状都不一样（结构性错误）。
              - "missing"：某一方没有抓到该层的输出（钩子没挂上/层不存在）。
    """
    # 分别给两个模型挂上调试钩子，拿到各自的 traces 字典和 handle 列表。
    ours_traces, ours_handles = _attach_debug_hooks(ours, is_hf=False)
    hf_traces, hf_handles = _attach_debug_hooks(hf_model, is_hf=True)

    try:
        # inference_mode 比 no_grad 更彻底地关闭autograd 相关开销，
        # 这里只做前向推理、不需要反向传播，用它更快也更安全。
        with torch.inference_mode():
            ours(input_ids)
            hf_model(input_ids)
    finally:
        # 无论前向是否抛异常，都要把钩子摘掉，避免钩子残留导致
        # 之后再次调用模型时污染 traces 或产生副作用。
        for h in ours_handles + hf_handles:
            h.remove()

    # 取两边都出现过的层名的并集，再按前向顺序排序，逐层比较。
    layer_names = sorted(set(ours_traces) | set(hf_traces), key=_layer_sort_key)
    results = []
    for name in layer_names:
        ours_tensor = ours_traces.get(name)
        hf_tensor = hf_traces.get(name)

        if ours_tensor is None or hf_tensor is None:
            # 只有一边抓到了这一层的输出，说明层名对不上或者钩子没挂成功。
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
            # 形状都不一样就没法逐元素比较了，直接标记为结构性错误。
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

        # 形状一致时才计算逐元素绝对差异，取最大值和平均值两个指标，
        # 既能看出"最坏情况偏差多大"，也能看出"整体偏差程度"。
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
    """把 `layerwise_differences` 返回的结果列表格式化成人类可读的多行文本报告。

    每一行前缀标签含义：
        [OK]      该层数值在容差范围内一致；
        [DIFF]    该层形状一致但数值有明显差异，是重点排查对象；
        [SHAPE]   该层输出形状本身就不一致（结构性 bug）；
        [MISSING] 某一方缺失该层的输出记录。
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
    # 作为脚本直接运行时的入口：构建一对迷你模型、跑一次前向、打印逐层对拍报告。
    transformers_available = importlib.util.find_spec("transformers") is not None
    if not transformers_available:
        raise SystemExit("transformers is not installed; install it to run the debugger.")

    # 动态从 notebook 里加载手写实现的类和权重加载函数。
    import_notebook_defs = load_notebook_defs()
    cfg = tiny_debug_config()

    # 构建权重对齐的一对模型（随机初始化权重，因为没传 hf_checkpoint）。
    ours_model, hf_model = build_tiny_aya_pair(import_notebook_defs, cfg)

    # 固定随机种子以获得可复现的输入，方便反复调试时结果一致。
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), dtype=torch.long)

    # 逐层比较并打印报告，任何一层出现 [DIFF]/[SHAPE]/[MISSING] 都说明
    # 手写实现在该层的计算逻辑与 HF 官方实现不一致，需要重点排查。
    diffs = layerwise_differences(ours_model, hf_model, input_ids)
    print(format_report(diffs))
