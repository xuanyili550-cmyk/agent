# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Gemma 3 逐层（layer-by-layer）调试对比工具。

本模块的作用：把书中自己实现的 Gemma3Model（"ours"，来自 standalone-gemma3.ipynb
notebook）与 HuggingFace transformers 官方实现的 Gemma3ForCausalLM（"hf"）放在同一份
随机权重（或同一份预训练权重）下跑一次前向传播，并通过 forward hook 抓取两边网络
中每一层、每一个子模块（embedding、每个 Transformer block 内部的 LayerNorm/注意力
投影/前馈网络等）的输出张量，逐一比较数值是否一致（allclose）、形状是否一致，从而
定位"自己实现的模型到底是哪一层开始和官方实现的结果对不上"的问题。

核心流程：
1. tiny_debug_config()      —— 构造一个很小的调试用配置（层数、维度都很小，跑得快）。
2. _hf_config_from_dict()   —— 把上面的自定义配置转换成 HuggingFace 的 Gemma3TextConfig。
3. build_gemma3_pair()      —— 分别构建"我们的模型"和"HF 模型"，并把 HF 模型的权重
                                 转换/加载进我们的模型里，保证两者权重完全一致，这样
                                 前向输出的任何差异就只能来自实现逻辑本身的不同，而不是
                                 权重不同导致的。
4. _attach_debug_hooks()    —— 给两个模型的各层挂上 forward hook，运行一次前向后就能
                                 拿到每一层的中间输出。
5. layerwise_differences()  —— 对比两边同名层的输出，计算最大绝对误差 / 平均绝对误差，
                                 判断是否在给定的 rtol/atol 容差内一致。
6. format_report()          —— 把上面的比较结果格式化成一份人类可读的报告，方便快速
                                 定位第一处出现明显数值偏差（DIFF/SHAPE/MISSING）的层。

【本次改动说明】仅补充中文注释，未改变原有代码行为，除了下方 format_report() 中
一处已确认的重复判断（duplicate/dead code）bug 修复，具体说明见该函数内注释。
"""

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import torch

from llms_from_scratch.utils import import_definitions_from_notebook

try:
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
except ImportError:
    # 如果环境里没装 transformers，就把这两个符号设为 None，
    # 后面用到的地方会显式抛出更友好的 ImportError 提示信息。
    Gemma3ForCausalLM = None
    Gemma3TextConfig = None


def tiny_debug_config():
    """
    构造一个"迷你"版 Gemma 3 配置，专门用于调试/单元测试。

    刻意把词表、层数、维度都设得很小（vocab_size=257、n_layers=2、emb_dim=32 等），
    这样构建模型和跑前向传播都非常快，适合在调试脚本/CI 里反复运行，
    而不追求真实模型的效果，只追求"结构对齐、数值对齐"。
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
        "rope_base": 1_000_000.0,       # 全局注意力层使用的 RoPE 基频（theta）
        "rope_local_base": 10_000.0,    # 局部/滑动窗口注意力层使用的 RoPE 基频
        "sliding_window": 4,
        # layer_types 长度必须等于 n_layers，逐层标注该层是全局注意力还是滑窗注意力；
        # 这里两层都设为 full_attention，所以本配置下滑窗逻辑实际不会被触发。
        "layer_types": ["full_attention", "full_attention"],
        "dtype": torch.float32,
        "query_pre_attn_scalar": 256,   # Gemma 系列注意力打分前的缩放因子
    }


def _hf_config_from_dict(cfg):
    """
    把本文件里自定义的 dict 配置（tiny_debug_config 的输出格式）转换成
    HuggingFace 官方的 Gemma3TextConfig 对象，以便实例化 Gemma3ForCausalLM。

    两边配置项的命名并不完全一致（例如我们的 "emb_dim" 对应 HF 的 "hidden_size"，
    "n_heads" 对应 "num_attention_heads"），这里做的就是逐项的字段名映射。
    """
    if Gemma3TextConfig is None:
        # transformers 未安装时提前报错，比后面莫名其妙的 NameError 更好定位问题。
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
        # 用 eager 实现（而非 flash-attn/sdpa），是为了能拿到普通 nn.Module 的
        # forward hook，同时避免某些融合实现导致中间张量不可直接比较。
        attn_implementation="eager",
        torch_dtype=cfg.get("dtype", torch.float32),
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
        rope_scaling={"rope_type": "default"},
        # 【风险/版本相关，未改动】以下两个参数（rope_scaling 的写法、
        # ignore_keys_at_rope_validation 这个字段名）依赖当前所用 transformers 库
        # 的具体版本实现细节；不同版本的 Gemma3TextConfig 对 RoPE 校验的处理方式
        # 可能不同，如果升级/降级 transformers 版本后这里报错，需要对照该版本的
        # Gemma3TextConfig 源码检查这两个参数是否仍然存在/含义是否一致。
        ignore_keys_at_rope_validation={"full_attention", "sliding_attention"},
    )


def load_notebook_defs(nb_name="standalone-gemma3.ipynb"):
    """
    从同目录上一级的 standalone-gemma3.ipynb notebook 中动态导入类/函数定义
    （例如 Gemma3Model、load_weights_into_gemma），这样调试脚本可以直接复用
    notebook 里定义的"我们自己实现的"模型代码，无需手动复制一份 .py 文件，
    避免两处代码不同步。
    """
    nb_dir = Path(__file__).resolve().parents[1]
    return import_definitions_from_notebook(nb_dir, nb_name)


def build_gemma3_pair(import_notebook_defs, cfg, hf_checkpoint=None):
    """
    同时构建"我们的实现"和"HuggingFace 官方实现"两个 Gemma 3 模型，
    并让两者使用完全相同的权重，返回 (ours, hf_model) 这一对模型，供后续
    逐层比较使用。

    关键点：如果不把权重对齐，两个模型输出天然就会不同（因为初始化是随机的），
    那样就无法判断输出差异到底是"权重不同"造成的还是"实现逻辑写错了"造成的。
    所以这里的核心步骤是 load_weights_into_gemma(...)，把 HF 模型的
    state_dict 转换/搬运进我们自己的模型里。
    """
    if Gemma3ForCausalLM is None:
        raise ImportError("transformers is required for the Gemma 3 debugger.")

    # 固定随机种子，保证"我们的模型"每次初始化的（后面会被覆盖的）权重可复现。
    torch.manual_seed(123)
    ours = import_notebook_defs.Gemma3Model(cfg)

    if hf_checkpoint:
        # 传入了真实的 checkpoint 名称/路径，则直接加载预训练权重。
        hf_model = Gemma3ForCausalLM.from_pretrained(
            hf_checkpoint,
            torch_dtype=cfg.get("dtype", torch.float32),
            attn_implementation="eager",
        )
    else:
        # 否则用调试用的小配置随机初始化一个 HF 模型，仅用于结构/数值对齐调试。
        hf_cfg = _hf_config_from_dict(cfg)
        hf_model = Gemma3ForCausalLM(hf_cfg)

    # 把 HF 模型的随机（或预训练）权重，按 load_weights_into_gemma 里定义的映射关系，
    # 灌入我们自己实现的模型，确保两个模型此后使用同一套参数值。
    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    import_notebook_defs.load_weights_into_gemma(ours, param_config, hf_model.state_dict())

    # 切到 eval 模式：关闭 dropout 等训练时行为，保证前向传播是确定性的，
    # 这样才能做逐层数值比较（否则 dropout 的随机性会掩盖真正的实现差异）。
    ours.eval()
    hf_model.eval()
    return ours, hf_model


def _register_trace_hook(handles, traces, name, module, scale=None):
    """
    给指定的子模块 module 注册一个 forward hook，在每次前向传播完成后，
    把该模块的输出张量存进 traces[name]，同时把 hook 句柄存进 handles 里
    （方便调用方最后统一 remove，避免 hook 累积导致重复记录/内存泄漏）。

    scale 参数：某些层的输出需要额外乘一个缩放系数才能跟另一边模型的对应输出
    对齐比较（典型例子见下方 embedding 缩放，Gemma 系列会对 embedding 输出
    乘以 sqrt(emb_dim)）。
    """
    if module is None:
        # 允许传入 None（比如某个子模块在某种实现里不存在），直接跳过不挂 hook。
        return

    def _record(_, __, output):
        # forward hook 的标准签名是 (module, input, output)，这里用 _、__
        # 占位表示我们不关心 module 本身和输入，只关心输出 output。
        if isinstance(output, tuple):
            # 有些模块（如注意力层）forward 返回的是 tuple（比如 (hidden_states, attn_weights)），
            # 这里只取第一个元素，即真正的隐藏状态输出。
            output = output[0]
        if scale is not None:
            output = output * scale
        # 统一转成 float32 + detach + cpu，避免不同 dtype/device 导致比较失败，
        # 也避免继续持有计算图引用造成显存/内存占用。
        traces[name] = output.detach().to(torch.float32).cpu()

    handles.append(module.register_forward_hook(_record))


def _attach_debug_hooks(model, is_hf, include_block_details=True):
    """
    给一个模型（可能是 HF 的 Gemma3ForCausalLM，也可能是我们自己的 Gemma3Model）
    的关键子模块统一挂上调试用的 forward hook，返回 (traces, handles)：
      - traces：{层名: 该层输出张量} 的字典，前向传播跑完后才会被填充；
      - handles：所有 hook 句柄的列表，调用方用完之后需要逐个 remove()。

    is_hf 用来区分两种模型的属性命名方式不同（HF 是 model.model.layers[i].self_attn.q_proj，
    我们自己的实现是 model.trf_blocks[i].att.W_query 等），但是给它们记录的
    trace 名字（如 "block_0.att.q_proj"）是统一的，这样才能在 layerwise_differences
    里按同一个 name 做一一对应的比较。
    """
    traces = {}
    handles = []

    if is_hf:
        core = model.model
        _register_trace_hook(handles, traces, "embedding", core.embed_tokens)
        for idx, layer in enumerate(core.layers):
            block_name = f"block_{idx}"
            # 先挂一个"整个 block"的 hook，记录该 Transformer block 的最终输出，
            # 用于快速判断"这一层整体是否对齐"，无需先展开细节。
            _register_trace_hook(handles, traces, block_name, layer)

            if include_block_details:
                # 再逐个挂 block 内部每个子模块的 hook，方便在整体不一致时
                # 进一步定位到底是 LayerNorm、Q/K/V 投影、QK-Norm 还是前馈网络出的问题。
                _register_trace_hook(handles, traces, f"{block_name}.input_layernorm", layer.input_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.att.q_proj", layer.self_attn.q_proj)
                _register_trace_hook(handles, traces, f"{block_name}.att.k_proj", layer.self_attn.k_proj)
                _register_trace_hook(handles, traces, f"{block_name}.att.v_proj", layer.self_attn.v_proj)
                _register_trace_hook(handles, traces, f"{block_name}.att.q_norm", layer.self_attn.q_norm)
                _register_trace_hook(handles, traces, f"{block_name}.att.k_norm", layer.self_attn.k_norm)
                _register_trace_hook(handles, traces, f"{block_name}.att.o_proj", layer.self_attn.o_proj)
                _register_trace_hook(handles, traces, f"{block_name}.att", layer.self_attn)
                _register_trace_hook(handles, traces, f"{block_name}.post_attention_layernorm", layer.post_attention_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.pre_feedforward_layernorm", layer.pre_feedforward_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.ff.gate_proj", layer.mlp.gate_proj)
                _register_trace_hook(handles, traces, f"{block_name}.ff.up_proj", layer.mlp.up_proj)
                _register_trace_hook(handles, traces, f"{block_name}.ff.down_proj", layer.mlp.down_proj)
                _register_trace_hook(handles, traces, f"{block_name}.ff", layer.mlp)
                _register_trace_hook(handles, traces, f"{block_name}.post_feedforward_layernorm", layer.post_feedforward_layernorm)

        _register_trace_hook(handles, traces, "final_norm", core.norm)
        _register_trace_hook(handles, traces, "logits", model.lm_head)
    else:
        # 我们自己实现的模型：embedding 输出需要额外乘以 sqrt(emb_dim) 才能对齐
        # HF 内部对 embedding 的缩放约定（Gemma 系列的常见做法）。
        # 优先从 model.cfg 里取 emb_dim，取不到就退回用 tok_emb 的 embedding_dim 属性。
        emb_scale = float(getattr(model, "cfg", {}).get("emb_dim", model.tok_emb.embedding_dim) ** 0.5)
        _register_trace_hook(handles, traces, "embedding", model.tok_emb, scale=emb_scale)
        # 不同版本/命名习惯下，Transformer block 列表属性名可能是 blocks 或 trf_blocks，
        # 依次尝试，都找不到就说明模型结构变了，直接报错提示。
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            blocks = getattr(model, "trf_blocks", None)
        if blocks is None:
            raise AttributeError("Could not locate Gemma 3 blocks on the local model.")
        for idx, block in enumerate(blocks):
            block_name = f"block_{idx}"
            _register_trace_hook(handles, traces, block_name, block)

            if include_block_details:
                # 这里的属性名（W_query/W_key/W_value/out_proj/fc1/fc2/fc3 等）
                # 对应的是 standalone-gemma3.ipynb 里自定义模块的命名方式，
                # 与上面 HF 分支里的属性名不同，但记录时使用相同的 trace 名字
                # （如 "block_0.att.q_proj"），从而实现跨实现的层级对齐比较。
                _register_trace_hook(handles, traces, f"{block_name}.input_layernorm", block.input_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.att.q_proj", block.att.W_query)
                _register_trace_hook(handles, traces, f"{block_name}.att.k_proj", block.att.W_key)
                _register_trace_hook(handles, traces, f"{block_name}.att.v_proj", block.att.W_value)
                _register_trace_hook(handles, traces, f"{block_name}.att.q_norm", block.att.q_norm)
                _register_trace_hook(handles, traces, f"{block_name}.att.k_norm", block.att.k_norm)
                _register_trace_hook(handles, traces, f"{block_name}.att.o_proj", block.att.out_proj)
                _register_trace_hook(handles, traces, f"{block_name}.att", block.att)
                _register_trace_hook(handles, traces, f"{block_name}.post_attention_layernorm", block.post_attention_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.pre_feedforward_layernorm", block.pre_feedforward_layernorm)
                _register_trace_hook(handles, traces, f"{block_name}.ff.gate_proj", block.ff.fc1)
                _register_trace_hook(handles, traces, f"{block_name}.ff.up_proj", block.ff.fc2)
                _register_trace_hook(handles, traces, f"{block_name}.ff.down_proj", block.ff.fc3)
                _register_trace_hook(handles, traces, f"{block_name}.ff", block.ff)
                _register_trace_hook(handles, traces, f"{block_name}.post_feedforward_layernorm", block.post_feedforward_layernorm)

        _register_trace_hook(handles, traces, "final_norm", model.final_norm)
        _register_trace_hook(handles, traces, "logits", model.out_head)

    return traces, handles


def _layer_sort_key(name):
    """
    给 trace 名字生成一个排序 key，让最终报告按照"网络实际执行顺序"输出，
    而不是按字符串默认字典序（否则 "block_10" 会排在 "block_2" 前面，
    block 内部各子层也会乱序，不便于阅读调试报告）。

    排序优先级：embedding(0) < block_i 整体(1) < block_i 内部细节(2，
    再按 block_detail_order 里定义的执行先后顺序排) < final_norm(3) < logits(4) < 其他(5)。
    """
    block_detail_order = {
        "input_layernorm": 0,
        "att.q_proj": 1,
        "att.k_proj": 2,
        "att.v_proj": 3,
        "att.q_norm": 4,
        "att.k_norm": 5,
        "att.o_proj": 6,
        "att": 7,
        "post_attention_layernorm": 8,
        "pre_feedforward_layernorm": 9,
        "ff.gate_proj": 10,
        "ff.up_proj": 11,
        "ff.down_proj": 12,
        "ff": 13,
        "post_feedforward_layernorm": 14,
    }

    if name == "embedding":
        return (0, 0)
    if name.startswith("block_"):
        # 形如 "block_1.att.q_proj" -> block_name="block_1", detail="att.q_proj"
        block_name, _, detail = name.partition(".")
        idx = int(block_name.split("_")[1])
        if not detail:
            # 没有 "." 的是 block 整体输出（如 "block_1"），排在同一 block 内所有细节之前（-1）。
            return (1, idx, -1)
        return (2, idx, block_detail_order.get(detail, 100), detail)
    if name == "final_norm":
        return (3, 0)
    if name == "logits":
        return (4, 0)
    return (5, name)


def layerwise_differences(ours, hf_model, input_ids, rtol=1e-5, atol=1e-5, include_block_details=True):
    """
    对"我们的模型"和"HF 模型"跑同一份 input_ids 做一次前向传播，
    逐层比较两边同名 trace 的输出张量，返回一个列表，每个元素是一层的比较结果：
      - status: "ok"（在容差内一致）/ "mismatch"（数值不一致）/
                "shape_mismatch"（输出形状都不一样，没法比较数值）/
                "missing"（某一边没有采集到该层的 trace，比如属性名对不上）。
      - max_diff / mean_abs_diff：两边输出的最大绝对误差 / 平均绝对误差。

    这是整个调试脚本"发现 bug"的核心：如果某一层报告 mismatch，
    那么问题大概率就出在这一层对应的实现代码里。
    """
    ours_traces, ours_handles = _attach_debug_hooks(ours, is_hf=False, include_block_details=include_block_details)
    hf_traces, hf_handles = _attach_debug_hooks(hf_model, is_hf=True, include_block_details=include_block_details)

    try:
        # inference_mode 关闭梯度追踪，加速前向且减少内存占用；
        # 两个模型依次跑前向，各自的 hook 会把中间结果记录到对应的 traces 字典里。
        with torch.inference_mode():
            ours(input_ids)
            hf_model(input_ids)
    finally:
        # 无论前向是否抛异常，都要把 hook 摘掉，避免影响后续对这两个模型的其他调用。
        for h in ours_handles + hf_handles:
            h.remove()

    # 取两边 trace 名字的并集，保证"只有一边采集到的层"也会被报告为 missing，
    # 而不是被悄悄忽略掉。
    layer_names = sorted(set(ours_traces) | set(hf_traces), key=_layer_sort_key)
    results = []
    for name in layer_names:
        ours_tensor = ours_traces.get(name)
        hf_tensor = hf_traces.get(name)

        if ours_tensor is None or hf_tensor is None:
            # 两边至少有一边没记录到这一层（例如属性名写错、模块为 None 被跳过等），
            # 直接标记为 missing，不再往下比较数值。
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
            # 形状都不一致就没法逐元素比较数值了，直接报告 shape_mismatch。
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

        # 形状一致时才计算逐元素绝对误差，以及是否落在 rtol/atol 容差范围内。
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


def _format_diff_line(diff, indent=""):
    """
    把单条比较结果（layerwise_differences 返回的一个 dict）渲染成一行可读文本，
    不同 status 用不同的标签（[OK]/[DIFF]/[SHAPE]/[MISSING]）方便肉眼快速扫描。
    """
    if diff["status"] == "ok":
        return f"{indent}[OK] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}"
    if diff["status"] == "mismatch":
        return f"{indent}[DIFF] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}"
    if diff["status"] == "shape_mismatch":
        return f"{indent}[SHAPE] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}"
    return f"{indent}[MISSING] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}"


def format_report(differences, show_block_details=True, details_for_all_blocks=False):
    """
    把 layerwise_differences() 产出的比较结果列表，整理成一份分层缩进的文本报告：
    先打印顶层节点（embedding / block_i 整体 / final_norm / logits），
    如果某个 block 整体状态不是 ok，或者该 block 内部子层有不一致的地方，
    就在其下方用两格缩进展开打印该 block 内部每个子层的详细比较结果，
    便于快速定位"到底是哪个子模块导致这个 block 输出不一致"。

    details_for_all_blocks=True 时，无论该 block 是否一致，都强制展开细节
    （方便逐层核对，而不仅仅是排查错误）。
    """
    lines = []
    # 顶层节点的 trace 名字里不含 "."（如 "embedding"、"block_0"、"final_norm"、"logits"），
    # 子层细节的名字则形如 "block_0.att.q_proj"，含有 "."，因此用是否含 "." 来做区分。
    top_level_diffs = [diff for diff in differences if "." not in diff["name"]]

    for diff in sorted(top_level_diffs, key=lambda d: _layer_sort_key(d["name"])):
        lines.append(_format_diff_line(diff))

        if not show_block_details or not diff["name"].startswith("block_"):
            # 未开启细节展示，或者当前顶层节点不是 block（比如 embedding/final_norm/logits
            # 本身就没有更细的子层可展开），直接跳过，处理下一个顶层节点。
            continue

        detail_prefix = f"{diff['name']}."
        detail_diffs = [
            other for other in differences
            if other["name"].startswith(detail_prefix)
        ]
        if not detail_diffs:
            # 这个 block 没有采集任何子层细节（例如调用时 include_block_details=False），
            # 自然也就没什么可展开的。
            continue

        has_detail_mismatch = any(other["status"] != "ok" for other in detail_diffs)
        # 【bug 修复】原代码在这里之后紧跟着另一行几乎相同的判断：
        #     if not details_for_all_blocks and diff["status"] == "ok":
        #         continue
        # 这一行会在"block 整体输出是 ok"时无条件跳过细节展开——完全不考虑刚刚算出来的
        # has_detail_mismatch。也就是说，即便某个子层（比如某个 LayerNorm 或某个投影层）
        # 数值其实不一致，只要这两个不一致在 block 输出层面“恰好被抵消/掩盖”导致整体
        # 输出仍然 allclose，原代码也会把这个隐藏的子层差异悄悄吞掉、永远不打印出来。
        # 这使得上面刚刚计算的 has_detail_mismatch 变量形同虚设（计算了却从未真正
        # 影响任何分支），是一处明显的重复判断/死代码 bug。已确认删除该重复行、
        # 只保留下面这一条同时检查 has_detail_mismatch 的判断后，程序行为才符合变量
        # 命名所表达的本意，且经 py_compile 验证语法无误。
        if not details_for_all_blocks and diff["status"] == "ok" and not has_detail_mismatch:
            continue

        for other in sorted(detail_diffs, key=lambda d: _layer_sort_key(d["name"])):
            lines.append(_format_diff_line(other, indent="  "))

    return "\n".join(lines)


if __name__ == "__main__":
    # 命令行直接运行本文件时的入口：跑一次完整的调试对比并打印报告。
    transformers_available = importlib.util.find_spec("transformers") is not None
    if not transformers_available:
        raise SystemExit("transformers is not installed; install it to run the debugger.")

    import_notebook_defs = load_notebook_defs()
    cfg = tiny_debug_config()

    ours_model, hf_model = build_gemma3_pair(import_notebook_defs, cfg)
    torch.manual_seed(0)
    # 随机生成一批 token id 作为输入（值域 [0, vocab_size)），仅用于跑通前向、
    # 采集调试用的中间张量，不代表任何真实语义。
    input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), dtype=torch.long)
    diffs = layerwise_differences(ours_model, hf_model, input_ids)
    print(format_report(diffs))
