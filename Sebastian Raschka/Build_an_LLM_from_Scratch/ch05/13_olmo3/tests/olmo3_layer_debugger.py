# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""Olmo3 逐层调试对比工具（Olmo3 layer-by-layer debugging utility）。

本模块的目的：把「本仓库手写实现的 Olmo3Model」（下文简称 ours / 自研模型）
与「HuggingFace transformers 官方实现的 Olmo3ForCausalLM」（下文简称
hf_model / 官方模型）在**相同权重、相同输入**下跑一遍前向传播，然后逐层
（embedding → 每个 Transformer block → final_norm → lm_head/logits）
比较两边的中间激活值是否一致，从而快速定位「自研实现到底是哪一层写错了」。

核心思路：
1. 用官方模型的随机初始化权重反向"拷贝"进自研模型（load_weights_into_olmo），
   保证两边权重完全一致，这样如果输出不同，差异一定来自实现逻辑而非权重不同。
2. 用 PyTorch 的 forward hook（前向钩子）在两个模型的对应子模块上分别挂钩子，
   在前向传播时把每一层的输出张量记录下来（_attach_debug_hooks）。
3. 跑完一次前向后，把两边同名层的输出张量做逐元素比较（layerwise_differences），
   计算最大绝对误差 / 平均绝对误差，并用 torch.allclose 判定是否"足够接近"。
4. 把结果格式化成可读的报告（format_report），或者只取第一个出现分歧的层
   （first_mismatch），方便调试时"从前往后找第一个出问题的层"。

注意：本文件依赖可选的 `transformers` 库（用于构造/加载官方 Olmo3 模型）。
若未安装该库，多数函数会在被调用时显式抛出 ImportError 提示安装。
"""

import importlib
import importlib.util  # 【bug修复】显式导入 importlib.util 子模块：仅 `import importlib` 不保证 importlib.util 可用（能否访问取决于其它库是否已间接导入过它），跨环境/启动方式下可能抛 AttributeError
from pathlib import Path

import torch

from llms_from_scratch.utils import import_definitions_from_notebook

# transformers 是可选依赖：只有在做"官方实现 vs 自研实现"对比调试时才需要。
# 用 try/except 包裹，允许在没有安装 transformers 的环境下也能 import 本模块
# （只是调用相关函数时会报错），避免因为一个可选依赖缺失就让整个测试文件无法导入。
try:
    from transformers import Olmo3Config, Olmo3ForCausalLM
except ImportError:
    Olmo3Config = None
    Olmo3ForCausalLM = None


def tiny_debug_config():
    """返回一个"极小号"的 Olmo3 配置字典，用于快速做单元/调试测试。

    刻意把 vocab_size、emb_dim、n_layers 等都设得很小，这样构造模型、跑前向
    传播的耗时可以忽略不计，非常适合在 CPU 上反复跑来定位 bug。
    这里使用的是默认（非 YaRN）RoPE 缩放方式：rope_type="default"。

    Returns:
        dict: 描述模型结构和超参数的配置字典。
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
        "n_kv_heads": 2,
        "sliding_window": 4,
        # 两层都用全量（非滑动窗口）注意力，简化调试时的 mask 逻辑
        "layer_types": ["full_attention", "full_attention"],
        "dtype": torch.float32,
        "query_pre_attn_scalar": 256,
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
        "rope_base": 1_000_000.0,
        "rope_attention_factor": 1.0,
        "rope_type": "default",
        "rope_factor": 1.0,
        "rope_orig_max": 8,
        "rope_local_base": 10_000.0,
    }


def yarn_debug_config():
    """返回一个启用了 YaRN（Yet another RoPE extensioN）长上下文缩放的调试配置。

    与 tiny_debug_config 的结构参数（层数、维度等）基本相同，主要差异在于
    RoPE 相关字段：rope_type="yarn"，并额外携带 beta_fast / beta_slow /
    rope_factor / rope_orig_max 等 YaRN 专用超参数，用来验证自研实现里的
    YaRN 缩放公式是否与官方实现一致。

    注意：这里 rope_orig_max（原始训练长度，8192）远大于本配置实际使用的
    context_length（8），这是调试配置的常见做法（只关心数值公式是否对得上，
    不追求真实的长文本场景），HF 端在构造 Olmo3Config 时可能会因此打印一条
    "explicit factor 与 implicit factor 不一致"的 UserWarning，这是预期内的
    良性提示，不代表配置错误。

    Returns:
        dict: 描述模型结构、超参数及 YaRN RoPE 缩放参数的配置字典。
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
        "n_kv_heads": 2,
        "sliding_window": 4,
        "layer_types": ["full_attention", "full_attention"],
        "dtype": torch.float32,
        "query_pre_attn_scalar": 256,
        "attention_bias": False,
        "rms_norm_eps": 1e-6,
        "rope_base": 500_000.0,
        "rope_attention_factor": 1.2079441541679836,
        "rope_type": "yarn",
        "rope_factor": 8.0,
        "rope_orig_max": 8192,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "rope_local_base": 10_000.0,
    }


def _hf_config_from_dict(cfg):
    """把本仓库自定义的配置字典（dict）转换成 HuggingFace 的 Olmo3Config 对象。

    这是让"自研模型"和"官方模型"能够用同一份配置字典驱动的关键适配层：
    自研代码里字段名是 emb_dim/n_heads/hidden_dim 这种简短命名，而
    transformers 官方用的是 hidden_size/num_attention_heads/intermediate_size
    这类命名，此函数负责逐字段做名字翻译和默认值兜底。

    Args:
        cfg (dict): 由 tiny_debug_config / yarn_debug_config 等函数生成的配置字典。

    Returns:
        Olmo3Config: 可直接传给 Olmo3ForCausalLM(...) 构造官方模型的配置对象。

    Raises:
        ImportError: 当前环境未安装 transformers 时抛出。
    """
    if Olmo3Config is None:
        raise ImportError("transformers is required for the Olmo-3 debugger.")

    # rope_scaling 字典是 HF 侧描述 RoPE 缩放策略的统一入口：
    # rope_type="default" 时只需要类型标记；rope_type="yarn" 时还需要补充
    # attention_factor / beta_fast / beta_slow / factor / original_max_position_embeddings
    # 等 YaRN 专属超参数，才能让 HF 内部按 YaRN 公式正确缩放 RoPE 频率。
    rope_type = cfg.get("rope_type", "default")
    rope_scaling = {"rope_type": rope_type}
    if rope_type == "yarn":
        rope_scaling.update(
            {
                "attention_factor": cfg.get("rope_attention_factor", 1.0),
                "beta_fast": cfg.get("beta_fast", 32.0),
                "beta_slow": cfg.get("beta_slow", 1.0),
                "factor": cfg.get("rope_factor", 1.0),
                "original_max_position_embeddings": cfg.get("rope_orig_max", 8192),
            }
        )

    # 风险/跨版本提示（仅标注，不改代码）：
    # 这里把 rope_theta / rope_scaling / attn_implementation / torch_dtype /
    # qk_norm / query_pre_attn_scalar / head_dim 等作为关键字参数直接传给
    # Olmo3Config(...)。这些参数之所以能生效，是依赖 HF 的 PretrainedConfig
    # 基类支持任意 **kwargs 并原样存成配置属性这一行为；在本次核对所用的
    # transformers==5.5.0 中，Olmo3Config 内部已经把 rope_theta/rope_scaling
    # 归一化整合进了新的 rope_parameters 字段（可从下面构造出的对象上验证），
    # 兼容层仍然认识旧字段名。但不同大版本的 transformers 之间，Olmo3Config
    # 的具体字段名/校验逻辑可能发生变化（例如未来某天不再接受 rope_theta 这种
    # 旧写法），因此这段"字段名翻译"逻辑存在跨版本兼容性风险，建议在升级
    # transformers 后重新跑一遍本调试脚本确认。
    return Olmo3Config(
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["context_length"],
        hidden_size=cfg["emb_dim"],
        num_attention_heads=cfg["n_heads"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["hidden_dim"],
        head_dim=cfg["head_dim"],
        num_key_value_heads=cfg["n_kv_heads"],
        rope_theta=cfg["rope_base"],
        rope_local_base_freq=cfg.get("rope_local_base", 10_000.0),
        layer_types=cfg["layer_types"],
        sliding_window=cfg["sliding_window"],
        tie_word_embeddings=False,
        # eager 实现会显式计算注意力矩阵（而不是走 flash-attention 等融合核），
        # 这样才能保证和自研的手写注意力实现在数值路径上尽量对齐，便于逐层比较。
        attn_implementation="eager",
        torch_dtype=cfg.get("dtype", torch.float32),
        query_pre_attn_scalar=cfg.get("query_pre_attn_scalar", 256),
        rope_scaling=rope_scaling,
        qk_norm=cfg.get("qk_norm", False),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
    )


def load_notebook_defs(nb_name="standalone-olmo3.ipynb"):
    """从上一级目录的 Jupyter Notebook 中动态加载自研 Olmo3 实现的类/函数定义。

    本仓库把"从零手写"的模型代码放在教学用的 .ipynb 笔记本里而不是普通 .py
    文件中，因此这里借助 llms_from_scratch.utils.import_definitions_from_notebook
    这个工具函数，把 notebook 里定义的 Olmo3Model / load_weights_into_olmo 等
    符号提取出来，当作一个"伪模块"返回，供本文件其余部分像正常模块一样使用。

    Args:
        nb_name (str): 相对于本文件父目录的父目录（即仓库中 13_olmo3/ 目录）
            下的 notebook 文件名，默认是 "standalone-olmo3.ipynb"。

    Returns:
        ModuleType 或等价对象: 携带 notebook 中所有顶层定义（类、函数等）的
        命名空间对象，可以像 `import_notebook_defs.Olmo3Model` 这样访问。
    """
    # __file__ 是 .../ch05/13_olmo3/tests/olmo3_layer_debugger.py
    # parents[1] 是 tests/ 的上一级，即 .../ch05/13_olmo3/，notebook 就放在这里。
    nb_dir = Path(__file__).resolve().parents[1]
    return import_definitions_from_notebook(nb_dir, nb_name)


def build_olmo3_pair(import_notebook_defs, cfg, hf_checkpoint=None):
    """同时构建"自研 Olmo3Model"和"官方 Olmo3ForCausalLM"，并让两者权重对齐。

    调试对比的前提是两个模型权重完全相同，否则任何输出差异都无法区分是
    "权重不同"还是"实现逻辑写错了"。本函数的做法是：
      1. 用 cfg 构造自研模型 ours（此时权重是随机初始化的）。
      2. 构造/加载官方模型 hf_model（可以是随机初始化，也可以从
         hf_checkpoint 指定的预训练权重加载）。
      3. 用 load_weights_into_olmo 把 hf_model 的权重"灌入" ours，
         使得两者权重完全一致（哪怕 hf_model 本身是随机初始化的）。

    Args:
        import_notebook_defs: load_notebook_defs() 返回的命名空间对象，
            提供 Olmo3Model 类和 load_weights_into_olmo 函数。
        cfg (dict): tiny_debug_config / yarn_debug_config 等生成的配置字典。
        hf_checkpoint (str | None): 若提供，则通过 from_pretrained 加载该
            HuggingFace Hub / 本地路径下的真实预训练权重；否则使用随机初始化
            的官方模型（仅用于验证实现逻辑是否一致，不关心权重是否有意义）。

    Returns:
        tuple[nn.Module, nn.Module]: (ours, hf_model) 均已切换到 eval() 模式，
        且权重彼此对齐。

    Raises:
        ImportError: 当前环境未安装 transformers 时抛出。
    """
    if Olmo3ForCausalLM is None:
        raise ImportError("transformers is required for the Olmo-3 debugger.")

    ours = import_notebook_defs.Olmo3Model(cfg)
    hf_cfg = _hf_config_from_dict(cfg)

    if hf_checkpoint:
        hf_model = Olmo3ForCausalLM.from_pretrained(
            hf_checkpoint,
            torch_dtype=cfg.get("dtype", torch.float32),
            attn_implementation="eager",
        )
    else:
        # 不传 checkpoint 时，官方模型也是随机初始化——但没关系，反正接下来会
        # 把它的权重拷贝进自研模型，两边最终会保持一致，重点是"实现是否等价"。
        hf_model = Olmo3ForCausalLM(hf_cfg)

    # load_weights_into_olmo 只需要知道层数和 FFN 隐藏维度就能正确地把
    # hf_model.state_dict() 里的各个张量按名字映射拷贝到 ours 的对应子模块。
    param_config = {"n_layers": cfg["n_layers"], "hidden_dim": cfg["hidden_dim"]}
    import_notebook_defs.load_weights_into_olmo(ours, param_config, hf_model.state_dict())

    # 调试对比只关心前向数值，切到 eval 模式关闭 dropout 等训练态随机性，
    # 保证两次前向传播是确定性的、可比较的。
    ours.eval()
    hf_model.eval()
    return ours, hf_model


def _attach_debug_hooks(model, is_hf):
    """在模型的关键子模块上挂 forward hook，记录每一层输出的激活值。

    PyTorch 的 register_forward_hook 允许在不修改模型代码的前提下，
    "旁路监听"某个子模块每次前向传播的输出。这里分别处理两种模型结构：
      - is_hf=True（官方 HF 实现）：模块路径是 model.model.embed_tokens /
        model.model.layers[i] / model.model.norm / model.lm_head。
      - is_hf=False（自研实现）：模块路径是 model.tok_emb / model.blocks[i] /
        model.final_norm / model.out_head。
    两边分别在结构上对应的位置挂钩子，钩子记录时使用统一的层名
    （"embedding" / "block_i" / "final_norm" / "logits"），这样后续
    layerwise_differences 才能按同名层一一比对。

    Args:
        model (nn.Module): 待挂钩子的模型（自研或官方均可）。
        is_hf (bool): True 表示 model 是 HuggingFace 官方实现，False 表示自研实现。

    Returns:
        tuple[dict, list]:
            traces: {层名: 该层输出张量(已 detach、转 float32、搬到 CPU)} 的字典，
                在调用方跑完一次前向传播之后才会被填充。
            handles: 所有已注册 hook 的句柄列表，调用方用完后需要依次调用
                handle.remove() 来卸载钩子，避免影响模型之后的正常使用。
    """
    traces = {}
    handles = []

    def hook(name):
        """构造一个绑定了具体层名 name 的 forward hook 回调函数（闭包）。"""
        def _record(_, __, output):
            # forward hook 签名固定为 (module, input, output)，这里用 _/__
            # 占位表示不关心 module 本身和输入，只关心输出。
            # detach() 断开计算图（调试只看数值，不需要反向传播）；
            # 统一转成 float32 再搬到 CPU，避免不同 dtype/设备导致比较时出错，
            # 也方便后续用 torch.allclose 之类的函数在 CPU 上做数值比较。
            traces[name] = output.detach().to(torch.float32).cpu()
        return _record

    if is_hf:
        # HF 的 Olmo3ForCausalLM 结构：model.model 是骨干（embed+layers+norm），
        # model.lm_head 是语言模型头（把最终隐藏状态映射到词表 logits）。
        core = model.model
        handles.append(core.embed_tokens.register_forward_hook(hook("embedding")))
        for idx, layer in enumerate(core.layers):
            handles.append(layer.register_forward_hook(hook(f"block_{idx}")))
        handles.append(core.norm.register_forward_hook(hook("final_norm")))
        handles.append(model.lm_head.register_forward_hook(hook("logits")))
        # 风险/跨版本提示（仅标注，不改代码）：上面对 HF 每个 decoder layer
        # （layer.register_forward_hook）挂钩子后，hook 回调里直接对 output
        # 调用 output.detach(...)，这隐含假设 Olmo3DecoderLayer.forward 的
        # 返回值是"单个张量"。经核对当前安装的 transformers==5.5.0 中
        # Olmo3DecoderLayer.forward 确实返回单个 hidden_states 张量，
        # 因此本环境下可以正常工作；但历史上/其他版本的 transformers 中，
        # decoder layer 的 forward 有时会返回 (hidden_states, ...) 形式的
        # tuple（例如开启 output_attentions/use_cache 等场景），一旦如此，
        # output.detach() 会因为 tuple 没有 detach 方法而报 AttributeError。
        # 这属于跨版本兼容性风险，不在本次改动范围内修复，升级 transformers
        # 后建议重新验证。
    else:
        # 自研 Olmo3Model 结构：tok_emb 是词嵌入层，blocks 是
        # nn.ModuleList[TransformerBlock]，final_norm 是最终的 RMSNorm，
        # out_head 是输出投影（等价于 HF 的 lm_head）。
        handles.append(model.tok_emb.register_forward_hook(hook("embedding")))
        for idx, block in enumerate(model.blocks):
            handles.append(block.register_forward_hook(hook(f"block_{idx}")))
        handles.append(model.final_norm.register_forward_hook(hook("final_norm")))
        handles.append(model.out_head.register_forward_hook(hook("logits")))

    return traces, handles


def _layer_sort_key(name):
    """把层名字符串映射成一个可排序的元组键，用于让报告按"网络前向顺序"展示。

    仅按字符串默认排序会把 "block_10" 排在 "block_2" 前面（字典序问题），
    也无法保证 embedding 排在最前、logits 排在最后。这里显式定义顺序：
    embedding(0) < block_i(1, i) < final_norm(2) < logits(3) < 其他未知层名(4)。

    Args:
        name (str): 层名，例如 "embedding" / "block_3" / "final_norm" / "logits"。

    Returns:
        tuple: 可直接用作 sorted(..., key=...) 的排序键。
    """
    if name == "embedding":
        return (0, 0)
    if name.startswith("block_"):
        # "block_3".split("_") -> ["block", "3"]，取下标 1 拿到层号并转成 int，
        # 这样才能按数值大小排序而不是按字符串排序（避免 "10" 排在 "2" 前面）。
        idx = int(name.split("_")[1])
        return (1, idx)
    if name == "final_norm":
        return (2, 0)
    if name == "logits":
        return (3, 0)
    # 兜底分支：理论上不会走到这里（当前只会产生上面四类层名），
    # 保留是为了让函数对未知层名也不至于抛异常。
    return (4, name)


def layerwise_differences(ours, hf_model, input_ids, rtol=1e-5, atol=1e-5):
    """对自研模型与官方模型跑同一批输入，逐层比较中间激活值的数值差异。

    整体流程：
      1. 分别给 ours 和 hf_model 挂上 debug hook（_attach_debug_hooks）。
      2. 在 torch.inference_mode() 下（关闭梯度追踪、加速推理）对两个模型
         用同一个 input_ids 各跑一次前向传播，hook 会把每一层的输出记录到
         对应的 traces 字典中。
      3. 无论比较是否成功，finally 块都会确保移除所有 hook 句柄，
         避免钩子常驻在模型上影响后续正常使用（例如重复调用本函数）。
      4. 按 _layer_sort_key 定义的网络前向顺序，逐层比较两边的输出张量：
         形状不一致记为 "shape_mismatch"；某一边缺失该层记为 "missing"；
         形状一致则计算最大绝对误差/平均绝对误差，并用 torch.allclose
         判定是否在给定的 rtol/atol 容差范围内一致（"ok" 或 "mismatch"）。

    Args:
        ours (nn.Module): 自研 Olmo3Model 实例。
        hf_model (nn.Module): 官方 Olmo3ForCausalLM 实例。
        input_ids (torch.LongTensor): 形状 (batch, seq_len) 的 token id 输入，
            两个模型会用完全相同的输入跑前向传播。
        rtol (float): torch.allclose 的相对容差。
        atol (float): torch.allclose 的绝对容差。

    Returns:
        list[dict]: 每个元素描述一层的比较结果，包含
            name / status（ok|mismatch|shape_mismatch|missing）/
            ours_shape / hf_shape / max_diff / mean_abs_diff。
    """
    ours_traces, ours_handles = _attach_debug_hooks(ours, is_hf=False)
    hf_traces, hf_handles = _attach_debug_hooks(hf_model, is_hf=True)

    try:
        # inference_mode 比 no_grad 更激进地关闭自动求导相关的簿记开销，
        # 适合纯前向推理调试的场景（不需要反向传播）。
        with torch.inference_mode():
            ours(input_ids)
            hf_model(input_ids)
    finally:
        # 无论上面前向传播是否抛异常，都必须卸载 hook，防止句柄泄漏
        # （常驻的 hook 会在后续每次调用模型时持续占用内存/计算开销）。
        for h in ours_handles + hf_handles:
            h.remove()

    # 取两边出现过的所有层名的并集（正常情况下应该完全相同），
    # 再按照网络的前向顺序排序，方便按"从前到后"的顺序阅读报告，
    # 第一处出现差异的层往往就是 bug 的源头。
    layer_names = sorted(set(ours_traces) | set(hf_traces), key=_layer_sort_key)
    results = []
    for name in layer_names:
        ours_tensor = ours_traces.get(name)
        hf_tensor = hf_traces.get(name)

        if ours_tensor is None or hf_tensor is None:
            # 只有一边记录到了这一层的输出——通常意味着两个模型的子模块
            # 命名/结构对不上（例如自研模型少了某个模块，或挂钩子时写错了名字）。
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

        shapes_match = ours_tensor.shape == hf_tensor.shape
        if not shapes_match:
            # 形状都对不上就没必要（也没办法）逐元素相减，直接报告形状不匹配，
            # 这通常指向配置字段翻译错误（如 head_dim/n_heads 弄反）等结构性问题。
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

        # 形状一致时才有意义做逐元素数值比较：
        # max_diff 反映"最坏情况"下两者差多少，mean_diff 反映整体平均偏差，
        # allclose 则给出一个基于 rtol/atol 的"是否可以判定为相等"的布尔结论。
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


def first_mismatch(differences):
    """从逐层比较结果中找出第一个"不是 ok"的层。

    调试时通常按前向顺序看：只要前面的层都对得上，第一个出问题的层
    大概率就是 bug 真正所在的位置（后面层的差异往往只是被放大的"连锁反应"）。

    Args:
        differences (list[dict]): layerwise_differences 的返回值。

    Returns:
        dict | None: 第一个 status != "ok" 的记录；如果所有层都一致则返回 None。
    """
    for diff in differences:
        if diff["status"] != "ok":
            return diff
    return None


def format_report(differences):
    """把逐层比较结果格式化成一份人类可读的多行文本报告。

    每一行对应一层，按 _layer_sort_key（即网络前向顺序）排序后输出：
      [OK]      形状一致且数值在容差范围内，附带最大/平均绝对误差；
      [DIFF]    形状一致但数值超出容差，附带最大/平均绝对误差；
      [SHAPE]   两边形状不一致，附带各自的形状；
      [MISSING] 只有一边记录到了该层输出，附带各自的形状（可能为 None）。

    Args:
        differences (list[dict]): layerwise_differences 的返回值。

    Returns:
        str: 多行文本报告，行与行之间用换行符连接。
    """
    lines = []
    for diff in sorted(differences, key=lambda d: _layer_sort_key(d["name"])):
        if diff["status"] == "ok":
            lines.append(f"[OK] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}")
        elif diff["status"] == "mismatch":
            lines.append(
                f"[DIFF] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}"
            )
        elif diff["status"] == "shape_mismatch":
            lines.append(
                f"[SHAPE] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}"
            )
        else:
            # 走到这里说明 status == "missing"。
            lines.append(f"[MISSING] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 命令行入口：python olmo3_layer_debugger.py
    # 用 YaRN 调试配置构建一对模型，跑一次随机输入，打印逐层比较报告。
    # 风险提示（仅标注，不改代码）：下面用 `import importlib` 后直接访问
    # `importlib.util`，而没有显式 `import importlib.util`。在常规
    # `python3 xxx.py` 启动方式下，CPython 解释器在完成自身启动（初始化
    # site 模块等）时就已经把 importlib.util 作为子模块导入并挂到了
    # importlib 包的命名空间上，因此这里能正常工作（已在本机验证）；
    # 但如果脚本被以 `python3 -S ...`（跳过 site 初始化）等非常规方式启动，
    # 或者在某些高度精简的嵌入式 Python 环境中，importlib.util 可能尚未被
    # 隐式导入，届时会抛出 AttributeError。这是环境相关的边界情况，
    # 不属于本次要求修复的"确定性 bug"，故仅在此标注，不改动代码。
    transformers_available = importlib.util.find_spec("transformers") is not None
    if not transformers_available:
        raise SystemExit("transformers is not installed; install it to run the debugger.")

    import_notebook_defs = load_notebook_defs()
    cfg = yarn_debug_config()

    ours_model, hf_model = build_olmo3_pair(import_notebook_defs, cfg)
    torch.manual_seed(0)
    # 生成 (1, context_length) 的随机 token id 序列作为两个模型共同的输入。
    input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), dtype=torch.long)
    diffs = layerwise_differences(ours_model, hf_model, input_ids)
    print(format_report(diffs))
