# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""Qwen3.5 逐层调试工具（layer debugger）。

本模块用于把 notebook（qwen3.5.ipynb）里手写实现的 ``Qwen3_5Model``，与 HuggingFace
``transformers`` 官方实现的 ``Qwen3_5ForCausalLM`` 做逐层数值对比，帮助定位手写实现
和官方实现之间在哪一层开始出现数值偏差（这是排查"复现官方模型权重加载/前向计算是否正确"
的标准调试手法）。

核心调试思路：
1. 构造一个参数量极小的调试配置（``tiny_debug_config``），保证两边模型都能快速前向。
2. 用同一份随机权重（从 HF 模型 state_dict 复制到 notebook 自定义模型里）初始化两个模型，
   确保参数完全一致，这样如果输出不同，就说明是"计算逻辑"错了，而不是"权重不同"导致的。
3. 给两个模型的每一层（embedding、每个 Transformer block、最终归一化、输出头/logits）都注册
   forward hook，记录各层的输出张量。
4. 对两边同名层的输出做逐元素比较（``torch.allclose`` + 最大/平均绝对误差），
   从前往后找到第一个出现明显偏差（``mismatch``/``shape_mismatch``）的层，就能快速定位 bug
   所在的具体子模块。

【bug修复】原为 Build_an_LLM_from_Scratch.* 的 import 路径，该包并不存在（实测 ModuleNotFoundError）；
已改回真实包名 llms_from_scratch（与仓库其余 38 个文件及安装说明一致）——见下方 import 行的注释。
"""

import sys
from pathlib import Path

import torch

from llms_from_scratch.utils import import_definitions_from_notebook  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)


def _import_qwen3_5_classes():
    """尝试导入 HuggingFace 官方的 Qwen3.5 配置类与模型类，并提供本地源码兜底方案。

    优先使用 pip 安装的 ``transformers`` 包里的 Qwen3.5 实现；如果当前安装的 transformers
    版本还没有收录 Qwen3.5（比如版本太旧），则退而求其次，尝试从仓库内 ``transformers-main/src``
    这个本地源码目录里导入（说明开发者手动 clone 了 transformers 主分支代码用于抢先适配新模型）。

    返回：
        (Qwen3_5TextConfig, Qwen3_5ForCausalLM) 二元组；如果两种方式都失败，则把原始异常
        继续向外抛出，交由调用方（模块底部的 try/except）兜底为 None。
    """
    try:
        # 路线一：直接从已安装的 transformers 包导入（要求 transformers 版本足够新）
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        return Qwen3_5TextConfig, Qwen3_5ForCausalLM
    except Exception:
        # 路线二：已安装版本没有 Qwen3.5，尝试使用仓库根目录下的本地 transformers 源码树
        # parents[3]：从当前文件 .../ch05/16_qwen3.5/tests/qwen3_5_layer_debugger.py 往上数 3 层，
        # 得到仓库根目录（Build_an_LLM_from_Scratch）
        repo_root = Path(__file__).resolve().parents[3]
        local_src = repo_root / "transformers-main" / "src"
        if not local_src.exists():
            # 本地源码树也不存在，说明确实没有可用的 Qwen3.5 实现，原样抛出方案一的异常
            raise

        # 先把已经导入过的 transformers 相关模块从缓存中清空，
        # 否则即使把本地源码路径插到 sys.path 最前面，Python 也会继续复用之前已缓存的旧模块，
        # 导致仍然导入不到 Qwen3.5（这是切换同名包源码路径时的常见陷阱）
        for name in list(sys.modules):
            if name == "transformers" or name.startswith("transformers."):
                del sys.modules[name]
        sys.path.insert(0, str(local_src))

        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        return Qwen3_5TextConfig, Qwen3_5ForCausalLM


try:
    # 模块加载时立即尝试导入一次；导入失败（两条路线都失败）则把两个类都设为 None，
    # 后续代码通过判断是否为 None 来决定是否具备运行调试脚本的条件
    Qwen3_5TextConfig, Qwen3_5ForCausalLM = _import_qwen3_5_classes()
except Exception:
    Qwen3_5TextConfig = None
    Qwen3_5ForCausalLM = None


def tiny_debug_config():
    """构造一份"迷你版" Qwen3.5 配置，仅用于调试对比，不追求真实模型规模。

    刻意把 vocab_size、emb_dim、n_layers 等都设得很小，是为了让两边模型的前向计算
    足够快，便于反复跑对比脚本；同时 layer_types 里混合了 "linear_attention"（线性注意力，
    Qwen3.5 引入的门控线性注意力/类 Mamba 结构）和 "full_attention"（标准的分组查询注意力），
    这样一次调试就能同时覆盖两种 block 类型的实现是否正确。
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
        "partial_rotary_factor": 1.0,
        "rms_norm_eps": 1e-6,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        # 两层分别用线性注意力和全量注意力，覆盖 Qwen3.5 的两种 block 类型
        "layer_types": ["linear_attention", "full_attention"],
        "dtype": torch.float32,
    }


def _hf_config_from_dict(cfg):
    """把本仓库自定义的（notebook 风格）配置字典，翻译成 HuggingFace ``Qwen3_5TextConfig``。

    两边配置字典的字段命名不同（例如本仓库用 ``emb_dim``，HF 用 ``hidden_size``），
    这个函数负责做字段名的映射，确保用来初始化 HF 官方模型的配置，和用来初始化本仓库
    手写模型的配置，在语义上完全对应，这样两个模型才具备可比性。
    """
    if Qwen3_5TextConfig is None:
        raise ImportError("Qwen3.5 classes are required for the layer debugger.")

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
        use_cache=False,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": cfg["rope_base"],
            "partial_rotary_factor": cfg.get("partial_rotary_factor", 1.0),
            # 注：mrope_interleaved / mrope_section 是多模态 RoPE（M-RoPE）相关参数，
            # 其取值约定和字段名会随 transformers 版本演进而变化，属于跨版本风险点，
            # 这里按【风险/跨版本项，仅标注不改动】处理，不在本次注释任务中修改。
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        torch_dtype=cfg.get("dtype", torch.float32),
    )
    # 强制使用 "eager"（朴素矩阵运算）注意力实现，而不是 flash-attention 等融合内核，
    # 这样才能保证和本仓库的手写 PyTorch 实现在数值上尽量可比（避免不同注意力内核
    # 之间的微小数值误差干扰调试结论）
    hf_cfg._attn_implementation = "eager"
    return hf_cfg


def load_notebook_defs(nb_name="qwen3.5.ipynb"):
    """从相邻目录的 Qwen3.5 notebook 中，动态导入其中定义的类/函数（如 ``Qwen3_5Model``）。

    ``import_definitions_from_notebook`` 是仓库工具函数，用于把 .ipynb 文件当作可导入的
    Python 模块来使用，从而无需把 notebook 手动导出为 .py 文件即可在测试/调试脚本中复用其定义。
    """
    # parents[1]：从当前文件（.../16_qwen3.5/tests/xxx.py）往上数 1 层，得到 16_qwen3.5 目录，
    # 也就是 qwen3.5.ipynb 所在的目录
    nb_dir = Path(__file__).resolve().parents[1]
    if str(nb_dir) not in sys.path:
        sys.path.insert(0, str(nb_dir))
    return import_definitions_from_notebook(nb_dir, nb_name)


def build_qwen3_5_pair(import_notebook_defs, cfg, hf_checkpoint=None):
    """构建一对可直接比较的模型：notebook 手写实现 ``ours`` 与 HF 官方实现 ``hf_model``。

    关键点在于"权重对齐"：先按 ``hf_checkpoint``（若提供）或随机初始化的配置构造出 HF 模型，
    再调用 notebook 里的 ``load_weights_into_qwen3_5``，把 HF 模型的 ``state_dict()`` 权重
    一一拷贝进本仓库手写模型里。这样两个模型在参数层面完全一致，后续如果前向输出不同，
    差异原因就必定出在"计算逻辑实现"上，而不是"权重初始化不同"，便于隔离问题根因。
    """
    if Qwen3_5ForCausalLM is None:
        raise ImportError("Qwen3.5 classes are required for the layer debugger.")

    # 先实例化本仓库手写的 Qwen3.5 模型（来自 notebook 定义）
    ours = import_notebook_defs.Qwen3_5Model(cfg)

    if hf_checkpoint:
        # 如果指定了真实的 HF checkpoint 路径/名称，直接加载预训练权重
        hf_model = Qwen3_5ForCausalLM.from_pretrained(
            hf_checkpoint,
            torch_dtype=cfg.get("dtype", torch.float32),
            attn_implementation="eager",
        )
    else:
        # 否则用调试配置构造一个随机初始化的 HF 模型，仅用于验证"计算逻辑"是否等价
        hf_cfg = _hf_config_from_dict(cfg)
        hf_model = Qwen3_5ForCausalLM(hf_cfg)

    # 把 HF 模型的随机/预训练权重，按照 notebook 里定义的映射规则，拷贝进手写模型
    import_notebook_defs.load_weights_into_qwen3_5(
        ours,
        {"n_layers": cfg["n_layers"], "layer_types": cfg["layer_types"]},
        hf_model.state_dict(),
    )
    hf_model.config.use_cache = False

    # 切换到 eval 模式，关闭 dropout 等训练态行为，确保两边前向计算是完全确定性的
    ours.eval()
    hf_model.eval()
    return ours, hf_model


def _attach_debug_hooks(model, is_hf):
    """给模型的关键子模块注册 forward hook，抓取每一层的输出张量，供后续逐层比对。

    参数：
        model: 待挂钩子的模型实例（可能是本仓库手写模型，也可能是 HF 官方模型）。
        is_hf: 是否为 HF 官方模型；因为两边模型的子模块命名/结构不同
               （例如 HF 用 ``model.model.embed_tokens``，本仓库用 ``model.tok_emb``），
               需要分支处理才能找到对应的层。

    返回：
        (traces, handles)：
            traces  —— dict，键为层名（如 "embedding"、"block_0"、"final_norm"、"logits"），
                        值为该层输出张量（已 detach、转为 float32、搬到 CPU，避免占用显存/
                        影响梯度图，同时统一 dtype 方便后续做数值比较）。
            handles —— hook 句柄列表，调用方用完之后需要逐个 remove()，防止 hook 常驻。
    """
    traces = {}
    handles = []

    def hook(name):
        # 闭包工厂：为每个层名生成一个专属的 hook 回调，回调里通过闭包捕获 name，
        # 这样就能把不同层的输出分别存到 traces 字典的不同 key 下
        def _record(_, __, output):
            if isinstance(output, tuple):
                # 部分模块（尤其是 HF 的 Transformer block）forward 返回值是元组
                # （如 (hidden_states, ...)），真正关心的隐藏状态通常是第一个元素
                output = output[0]
            traces[name] = output.detach().to(torch.float32).cpu()

        return _record

    if is_hf:
        # HF 官方模型的层级结构：model.model 是 backbone，包含 embed_tokens / layers / norm，
        # model.lm_head 是顶层的语言模型输出头
        core = model.model
        handles.append(core.embed_tokens.register_forward_hook(hook("embedding")))
        for idx, layer in enumerate(core.layers):
            handles.append(layer.register_forward_hook(hook(f"block_{idx}")))
        handles.append(core.norm.register_forward_hook(hook("final_norm")))
        handles.append(model.lm_head.register_forward_hook(hook("logits")))
    else:
        # 本仓库手写模型（Qwen3_5Model）的属性命名：tok_emb / trf_blocks（或 blocks）/
        # final_norm / out_head
        handles.append(model.tok_emb.register_forward_hook(hook("embedding")))
        blocks = getattr(model, "trf_blocks", None)
        if blocks is None:
            blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise AttributeError("Could not locate Qwen3.5 blocks on the local model.")
        for idx, block in enumerate(blocks):
            handles.append(block.register_forward_hook(hook(f"block_{idx}")))
        handles.append(model.final_norm.register_forward_hook(hook("final_norm")))
        handles.append(model.out_head.register_forward_hook(hook("logits")))

    return traces, handles


def _layer_sort_key(name):
    """把层名映射为可排序的 (阶段序号, 层内序号) 元组，用于按"embedding → block_0..N →
    final_norm → logits"的自然顺序输出调试报告，而不是按字符串默认排序（那样会把
    "block_10" 排在 "block_2" 前面）。
    """
    if name == "embedding":
        return (0, 0)
    if name.startswith("block_"):
        # "block_3" -> 拆分成 ["block", "3"]，取第二段转成整数，作为 block 序号排序
        idx = int(name.split("_")[1])
        return (1, idx)
    if name == "final_norm":
        return (2, 0)
    if name == "logits":
        return (3, 0)
    # 兜底：未识别的层名统一放最后，并按名字本身排序
    return (4, name)


def layerwise_differences(ours, hf_model, input_ids, rtol=1e-5, atol=1e-5):
    """核心比对函数：对同一份输入分别跑两遍前向，收集逐层输出并计算数值差异。

    参数：
        ours: 本仓库手写的 Qwen3.5 模型实例。
        hf_model: HuggingFace 官方 Qwen3.5 模型实例（权重已与 ours 对齐）。
        input_ids: 输入 token id 张量，形状 (batch, seq_len)。
        rtol/atol: 传给 ``torch.allclose`` 的相对/绝对误差容忍度，用于判定某层是否"通过"。

    返回：
        list[dict]，每个 dict 描述一层的比对结果，字段包括 name/status/ours_shape/
        hf_shape/max_diff/mean_abs_diff；status 取值：
            "ok"             —— 两边张量形状一致且数值在容忍度内
            "mismatch"       —— 形状一致但数值超出容忍度（说明该层计算逻辑有偏差）
            "shape_mismatch" —— 形状都不一致（说明该层的实现结构就有问题）
            "missing"        —— 某一边没有捕获到该层输出（说明 hook 挂载或层名对不上）
    """
    # 分别给两个模型挂上调试 hook，拿到各自的 traces 字典（层名 -> 输出张量）和 hook 句柄
    ours_traces, ours_handles = _attach_debug_hooks(ours, is_hf=False)
    hf_traces, hf_handles = _attach_debug_hooks(hf_model, is_hf=True)

    try:
        # inference_mode 下跑前向：既能加速（不建计算图），也符合"只做数值比对、不需要反传"
        # 的调试场景需求；hf_model 显式传 use_cache=False，避免 KV cache 分支影响输出形状/内容
        with torch.inference_mode():
            ours(input_ids)
            hf_model(input_ids, use_cache=False)
    finally:
        # 无论前向是否报错，都要把 hook 摘掉，避免钩子残留导致后续调用该模型时重复记录/内存泄漏
        for h in ours_handles + hf_handles:
            h.remove()

    # 取两边层名的并集，按自然顺序排序，保证报告里既不会漏掉"只在一边出现的层"（missing 情况），
    # 又能按 embedding -> block_0..N -> final_norm -> logits 的语义顺序展示
    layer_names = sorted(set(ours_traces) | set(hf_traces), key=_layer_sort_key)
    results = []
    for name in layer_names:
        ours_tensor = ours_traces.get(name)
        hf_tensor = hf_traces.get(name)

        if ours_tensor is None or hf_tensor is None:
            # 某一侧没抓到这一层的输出，通常意味着 hook 挂载点命名/结构对不上，
            # 而不是数值计算问题，单独归类为 "missing" 以便和真正的数值偏差区分开
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
            # 形状都对不上，说明该层的实现结构本身有问题（比如维度拼接顺序、reshape 参数错误），
            # 直接判定为 shape_mismatch，不再尝试做逐元素数值比较（形状不同也无法比较）
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

        # 形状一致时才有意义比较数值：计算逐元素绝对差，取最大值和均值，
        # 并用 torch.allclose 给出一个"是否在容忍度内"的布尔判定，
        # 这样报告里既有定性结论（ok/mismatch）又有定量数据（便于判断偏差是"轻微浮点误差"
        # 还是"实现逻辑性错误"）
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
    """把 ``layerwise_differences`` 返回的结构化结果，渲染成一份便于人眼阅读的文本报告。

    每一行对应一层，按状态使用不同前缀标签：
        [OK]      —— 该层数值一致，说明目前为止两边实现等价
        [DIFF]    —— 该层数值出现偏差，是需要重点排查的可疑层
        [SHAPE]   —— 该层输出形状就不一致，说明实现结构有问题
        [MISSING] —— 某一边没有捕获到该层输出，说明 hook 挂载点或层名不匹配
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
    # 命令行直接运行本脚本时的入口：构造一对权重对齐的迷你模型，跑一次随机输入，
    # 打印逐层数值比对报告，方便开发者快速判断 notebook 手写实现是否与官方实现等价
    if Qwen3_5ForCausalLM is None:
        # 两条导入路线（pip 版 transformers / 本地 transformers-main 源码）都失败，
        # 没有可比较的官方实现，直接终止并给出修复建议
        raise SystemExit(
            "Qwen3.5 classes are unavailable. Install a recent transformers version or use local transformers-main."
        )

    import_notebook_defs = load_notebook_defs()
    cfg = tiny_debug_config()

    ours_model, hf_model = build_qwen3_5_pair(import_notebook_defs, cfg)
    # 固定随机种子，保证每次运行生成的 input_ids 相同，调试结果可复现
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), dtype=torch.long)
    diffs = layerwise_differences(ours_model, hf_model, input_ids)
    print(format_report(diffs))
