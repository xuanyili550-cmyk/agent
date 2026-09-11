# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Gemma 4 (E4B 变体) 逐层调试对比脚本 / Gemma 4 (E4B variant) layer-by-layer debugging comparator.

【模块作用】
本脚本用于对比「本仓库 notebook 中手写实现的 Gemma4DenseModel」（下称 "ours"，即"我们自己的实现"）
与「HuggingFace transformers 官方 Gemma4ForCausalLM」（下称 "hf"）在同一份输入、同一份权重下，
逐层（embedding -> 每个 transformer block -> 最终归一化层 -> 输出 logits）的前向传播中间结果是否一致。

核心调试思路：
1. 用同一套随机初始化的 HF 权重，分别构建 "ours" 模型和 "hf" 模型（并把 HF 的权重加载进 "ours" 模型），
   确保两边参数完全相同——这样如果输出不同，说明是「实现逻辑」出了问题，而不是「权重不同」导致的。
2. 给两边模型的关键子模块分别注册 forward hook（前向钩子），在真正跑一次 forward 时，
   把每一层的输出张量"拦截"下来存到字典里（这就是所谓的"逐层调试"/layer-wise debugging）。
3. 按层名对齐两边的 trace（跟踪记录），逐层比较形状是否一致、数值是否在容差范围内接近，
   一旦某一层开始出现较大差异，就能快速定位到"实现分叉"发生在哪一层，从而缩小 bug 排查范围。

（本文件顶部的英文版权头与下方原有的英文行内注释均予以保留，未删除任何原始内容；
 已有的【bug修复】标注同样原样保留，见下方 import 行。）
"""

import sys
import types
from pathlib import Path

import torch

from llms_from_scratch.utils import import_definitions_from_notebook  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)


def _import_gemma4_classes():
    """
    导入 Gemma4 相关类（配置类 Gemma4TextConfig 与模型类 Gemma4ForCausalLM）。

    优先尝试从当前已安装的 `transformers` 库直接导入；如果失败（例如安装的 transformers
    版本还没有收录 Gemma4，这是一个较新/实验性的模型），则回退到仓库内 `temp/gemma-4/transformers-main/src`
    路径下的"源码级" transformers（相当于手动 vendored 进来的开发分支代码），把它插入到
    sys.path 最前面，从这个本地路径重新导入。

    返回：(Gemma4TextConfig, Gemma4ForCausalLM) 二元组。
    """
    try:
        # 优先走"标准安装"路径：如果当前环境的 transformers 已经内置 Gemma4，直接用它最省心。
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        return Gemma4TextConfig, Gemma4ForCausalLM
    except Exception:
        # 走到这里说明标准安装的 transformers 里没有 Gemma4，尝试回退到仓库自带的本地源码副本。
        # parents[4]：从当前文件 tests/test_e4b/gemma4_e4b_layer_debugger.py 往上数 4 级目录，
        # 得到仓库根目录（Build_an_LLM_from_Scratch/）。
        repo_root = Path(__file__).resolve().parents[4]
        local_src = repo_root / "temp" / "gemma-4" / "transformers-main" / "src"
        if not local_src.exists():
            # 本地也没有源码副本，说明确实没法用 Gemma4，把原始异常重新抛出，不要吞掉报错。
            raise

        # 清理掉 sys.modules 里已经导入过的（不含 Gemma4 的）transformers 相关模块缓存，
        # 否则 Python 会因为模块已被缓存而不会重新从新路径加载，导致下面的 import 仍然拿到旧版本。
        for name in list(sys.modules):
            if name == "transformers" or name.startswith("transformers."):
                del sys.modules[name]

        # 把本地源码路径插到 sys.path 最前面，保证后续 `import transformers` 优先命中这里。
        sys.path.insert(0, str(local_src))
        # transformers 内部有一个依赖版本检查模块，在这种"手动 vendor 源码"的场景下容易因为
        # 找不到对应的已安装 pip 包元数据而报错，这里用一个"啥都不做"的假模块把它替换掉，
        # 相当于跳过版本校验，避免不必要的 ImportError。
        dummy_dep_module = types.ModuleType("transformers.dependency_versions_check")
        dummy_dep_module.dep_version_check = lambda *args, **kwargs: None
        sys.modules["transformers.dependency_versions_check"] = dummy_dep_module

        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        return Gemma4TextConfig, Gemma4ForCausalLM


try:
    # 模块加载时就尝试一次性把 Gemma4 类导入好，供全局使用。
    Gemma4TextConfig, Gemma4ForCausalLM = _import_gemma4_classes()
except Exception:
    # 如果两条路径（标准安装 / 本地源码）都失败，不让整个模块导入直接崩溃，
    # 而是把这两个符号置为 None，把"能不能用 Gemma4"这个判断推迟到实际使用它们的地方
    # （见下方 `_hf_config_from_dict` / `build_gemma4_pair` / `__main__`），
    # 那样即使 Gemma4 不可用，本文件其余内容仍然可以被安全 import（例如被测试收集/引用）。
    Gemma4TextConfig = None
    Gemma4ForCausalLM = None


def tiny_debug_config():
    """
    构造一份"迷你版" Gemma4（E4B 变体）配置字典，专门用于调试对比，而非真实训练/推理规模。

    刻意把 vocab_size、emb_dim、n_layers 等都设得很小，这样：
    - 模型前向传播速度快，调试迭代周期短；
    - 张量维度小，打印/对比中间结果时更直观；
    - 但仍然覆盖了 Gemma4 的关键结构特性（滑动窗口注意力 + 全局注意力混合、
      KV 头共享、双倍宽度 MLP、部分旋转位置编码 partial rotary 等），
      所以只要迷你配置下逐层比对通过，通常就能说明核心实现逻辑是对的。
    """
    return {
        "vocab_size": 257,
        "vocab_size_per_layer_input": 257,
        "emb_dim": 32,
        "hidden_dim": 64,
        "n_layers": 3,
        "n_heads": 4,
        "head_dim": 8,
        "n_kv_heads": 2,
        "num_global_kv_heads": None,
        "global_head_dim": 12,
        "context_length": 8,
        "sliding_window": 4,
        # 三层的层类型模式：第 0 层用滑动窗口局部注意力，第 1、2 层用全局注意力。
        # Gemma4 官方配置要求最后一层必须是 full_attention，这里已满足。
        "layer_types": ["sliding_attention", "full_attention", "full_attention"],
        "hidden_size_per_layer_input": 8,
        "num_kv_shared_layers": 1,
        "use_double_wide_mlp": True,
        "attention_k_eq_v": False,
        "rope_local_base": 10_000.0,
        "rope_global_base": 1_000_000.0,
        "rope_global_type": "proportional",
        "rope_global_partial_rotary_factor": 0.25,
        "layer_norm_eps": 1e-6,
        "final_logit_softcap": 30.0,
        "tie_word_embeddings": False,
        "dtype": torch.float32,
    }


def _hf_config_from_dict(cfg):
    """
    把本仓库风格的"扁平配置字典"（cfg，见 tiny_debug_config）翻译/映射成
    HuggingFace 官方 `Gemma4TextConfig` 所需要的构造参数。

    这一步是两边能否对齐比较的关键：字段名、字段含义必须严格一一对应，
    任何一个映射错了，都会导致两个模型结构不一致，从而没法做"同权重下逐层比对"。
    """
    if Gemma4TextConfig is None:
        # 如果模块顶部导入 Gemma4 失败，这里给出更友好的报错信息，而不是让调用方看到
        # 一个 NoneType 不能被调用的晦涩报错。
        raise ImportError("Gemma 4 classes are required for the layer debugger.")

    return Gemma4TextConfig(
        vocab_size=cfg["vocab_size"],
        vocab_size_per_layer_input=cfg["vocab_size_per_layer_input"],
        hidden_size=cfg["emb_dim"],
        intermediate_size=cfg["hidden_dim"],
        num_hidden_layers=cfg["n_layers"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg["n_kv_heads"],
        num_global_key_value_heads=cfg["num_global_kv_heads"],
        head_dim=cfg["head_dim"],
        global_head_dim=cfg["global_head_dim"],
        max_position_embeddings=cfg["context_length"],
        sliding_window=cfg["sliding_window"],
        layer_types=cfg["layer_types"],
        hidden_size_per_layer_input=cfg["hidden_size_per_layer_input"],
        num_kv_shared_layers=cfg["num_kv_shared_layers"],
        use_double_wide_mlp=cfg["use_double_wide_mlp"],
        attention_k_eq_v=cfg["attention_k_eq_v"],
        final_logit_softcapping=cfg["final_logit_softcap"],
        hidden_activation="gelu_pytorch_tanh",
        tie_word_embeddings=cfg["tie_word_embeddings"],
        rms_norm_eps=cfg["layer_norm_eps"],
        attention_bias=False,
        attention_dropout=0.0,
        # 局部滑动窗口注意力层与全局注意力层使用两套不同的 RoPE（旋转位置编码）参数：
        # - sliding_attention：标准 RoPE，base=10000（本仓库风格里叫 rope_local_base）；
        # - full_attention：partial rotary（只对一部分维度做旋转，partial_rotary_factor=0.25），
        #   base=1_000_000（rope_global_base），且 rope_type 为 "proportional"。
        # 这两套参数必须和 "ours" 实现中 compute_rope_params 的调用方式完全对应，
        # 否则位置编码不一致会导致注意力层输出从很靠前的位置就开始出现数值偏差。
        rope_parameters={
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": cfg["rope_local_base"],
            },
            "full_attention": {
                "rope_type": cfg["rope_global_type"],
                "rope_theta": cfg["rope_global_base"],
                "partial_rotary_factor": cfg["rope_global_partial_rotary_factor"],
            },
        },
        # 强制用 "eager"（最朴素的逐元素实现）注意力实现，而不是 flash-attention / sdpa 等融合核，
        # 这样数值路径更接近手写实现，便于做逐层数值比对（融合实现可能因算子融合顺序不同
        # 产生极小的浮点误差，干扰调试判断）。
        attn_implementation="eager",
        # 注：torch_dtype 是 transformers 较新版本中已被 dtype 字段取代的旧参数名；
        # 当前安装的 transformers 版本仍兼容接收 torch_dtype 并在内部转存为 dtype，
        # 所以此处不会报错，但属于「跨版本可能失效」的写法——如果未来 transformers
        # 彻底移除对 torch_dtype 关键字的兼容处理，这里会报 TypeError。
        # 按用户要求：此类跨版本兼容性风险仅标注，不在本次改动中修改为 dtype=。
        torch_dtype=cfg.get("dtype", torch.float32),
    )


def load_notebook_defs(nb_name="standalone-gemma4.ipynb"):
    """
    从相邻目录下的 Jupyter Notebook（默认是 `standalone-gemma4.ipynb`，即 E4B 稠密模型的
    无 KV 缓存版参考实现）中动态导入其中定义的类/函数（例如 Gemma4DenseModel、
    load_weights_into_gemma4_dense 等），这样就不必把 notebook 里的代码手动复制一份到
    .py 文件里维护两份。

    parents[2]：从当前文件 tests/test_e4b/xxx.py 往上数 2 级目录，得到
    ch05/17_gemma4/ 目录，即 notebook 所在目录。
    """
    nb_dir = Path(__file__).resolve().parents[2]
    return import_definitions_from_notebook(nb_dir, nb_name)


def build_gemma4_pair(import_notebook_defs, cfg, hf_checkpoint=None):
    """
    构建一对"权重完全相同"的模型：
      - ours：用 notebook 中定义的 Gemma4DenseModel（手写实现）实例化；
      - hf_model：官方 Gemma4ForCausalLM，要么从 `hf_checkpoint` 加载真实预训练权重，
        要么按 cfg 随机初始化一个同结构的模型。

    然后把 hf_model 的 state_dict（权重字典）通过 `load_weights_into_gemma4_dense`
    灌入 ours，确保两边权重逐一对应、完全一致——这是后续"逐层比对"能够成立的前提：
    如果权重不同，即使实现完全正确，输出也必然不同，调试就失去了意义。

    最后把两个模型都切换到 eval() 模式（关闭 dropout 等训练态行为），
    并关闭 HF 模型的 KV 缓存（use_cache=False），因为这里只做一次性前向比对，不需要缓存。
    """
    if Gemma4ForCausalLM is None:
        raise ImportError("Gemma 4 classes are required for the layer debugger.")

    ours = import_notebook_defs.Gemma4DenseModel(cfg)

    if hf_checkpoint:
        # 传入了具体的 checkpoint 路径/名称时，从 HuggingFace Hub 或本地路径加载真实预训练权重。
        hf_model = Gemma4ForCausalLM.from_pretrained(
            hf_checkpoint,
            torch_dtype=cfg.get("dtype", torch.float32),
            attn_implementation="eager",
        )
    else:
        # 未提供 checkpoint 时，按 cfg 构造一个同结构但随机初始化权重的 HF 模型，
        # 随后会把这份随机权重"抄"到 ours 里，两边就有了完全相同的（虽然是随机的）参数。
        hf_cfg = _hf_config_from_dict(cfg)
        hf_model = Gemma4ForCausalLM(hf_cfg)

    # 把 hf_model 的权重字典搬运进 ours（按名称/前缀匹配张量并逐一 copy_），
    # 确保两个模型在数值上共享同一套参数。
    import_notebook_defs.load_weights_into_gemma4_dense(ours, cfg, hf_model.state_dict())
    hf_model.config.use_cache = False

    ours.eval()
    hf_model.eval()
    return ours, hf_model


def _attach_debug_hooks(model, is_hf):
    """
    给传入的模型（"ours" 或 "hf"）的关键子模块挂上 forward hook，
    在一次前向传播过程中把每一层的输出张量记录到 `traces` 字典里，
    key 为统一的层名（如 "embedding"、"block_0"、"final_norm"、"logits"），
    方便后续按同一层名把两边模型的输出拿出来做数值比较。

    返回 (traces, handles)：
      - traces：{层名: 张量} 的字典，前向传播结束后才会被填充；
      - handles：所有已注册 hook 的句柄列表，调用方用完之后需要逐一 remove()，
        否则 hook 会一直挂在模型上，多次调用会重复记录/累积副作用。
    """
    traces = {}
    handles = []

    def hook(name):
        """
        通用的"记录输出"hook 工厂函数：返回一个绑定了具体层名 `name` 的 hook 函数。
        PyTorch 的 forward hook 签名固定为 (module, input, output)，这里用 `_`、`__`
        占位表示不关心 module 本身和输入，只关心输出。
        """
        def _record(_, __, output):
            # 有些子模块的 forward 返回的是元组（比如 (hidden_states, next_kv_cache)），
            # 这里统一只取第一个元素，即真正的隐藏状态张量。
            if isinstance(output, tuple):
                output = output[0]
            # 统一转成 float32 + 挪到 CPU 再 detach，一是避免不同 dtype（如 bfloat16）
            # 精度差异干扰数值比较，二是避免张量还挂在计算图上占用显存/内存。
            traces[name] = output.detach().to(torch.float32).cpu()

        return _record

    # 【bug修复】原代码：is_hf=False 分支直接对 model.tok_emb 挂通用 hook（等价于 hook("embedding")），
    # 未做任何缩放。
    # 为什么是 bug：Gemma 系列模型的约定是"词嵌入输出 = nn.Embedding(input_ids) * sqrt(hidden_size)"。
    # HF 官方实现里，这个缩放写在 embed_tokens 子模块自己的 forward 内部
    # （Gemma4TextScaledWordEmbedding.forward 返回 `super().forward(input_ids) * self.embed_scale`），
    # 所以对 core.embed_tokens 挂普通 hook 拿到的已经是"缩放后"的值。
    # 但本仓库 "ours" 的 Gemma4DenseModel.forward 里，缩放是在 tok_emb 调用之后另起一行做的
    # （`x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)`），并不在 tok_emb 子模块内部完成。
    # 因此原代码对 self.tok_emb 挂的普通 hook 拿到的是"未缩放"的原始嵌入，与 HF 侧"已缩放"的
    # embed_tokens 输出在同一个 "embedding" 层名下比较，比较的其实是两个不同物理量，
    # 必然产生一个恒定的、与内容无关的虚假差异（用本文件默认迷你配置实测：
    # 修复前 embedding 一行报 [DIFF]，但下游 block_0/block_1/block_2/final_norm/logits 全部
    # max_diff=0.0——这恰恰说明模型实际计算完全一致，只是调试脚本这里比错了对象，
    # 是一个确定性的比较逻辑 bug，而非模型实现问题）。
    # 修复方式：仿照同目录 test_e2b/gemma4_e2b_layer_debugger.py 里已经验证过的做法，
    # 单独定义一个 scaled_embedding_hook，在记录 "ours" 的 embedding 输出时手动补上
    # 同样的 sqrt(emb_dim) 缩放，使两边 "embedding" trace 的含义对齐、可直接比较。
    def scaled_embedding_hook(name, scale):
        """
        专门给"ours"（本仓库手写实现）词嵌入层用的 hook：额外乘以一个缩放因子 `scale`。

        原因见上方【bug修复】注释：本仓库实现里 sqrt(emb_dim) 缩放不在 tok_emb 子模块内部，
        普通 hook 拿不到；这里在记录时手动补上，让 "ours" 与 HF 侧的 "embedding" trace
        对应同一个物理量（均为缩放后的词嵌入），才能公平比较。
        """
        def _record(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            traces[name] = (output * scale).detach().to(torch.float32).cpu()

        return _record

    if is_hf:
        # HF 官方模型结构：Gemma4ForCausalLM -> .model（Gemma4TextModel 主干）
        #   -> .embed_tokens / .layers[i] / .norm，再加上顶层的 .lm_head。
        core = model.model
        handles.append(core.embed_tokens.register_forward_hook(hook("embedding")))
        for idx, layer in enumerate(core.layers):
            handles.append(layer.register_forward_hook(hook(f"block_{idx}")))
        handles.append(core.norm.register_forward_hook(hook("final_norm")))
        # 注意：lm_head 这里挂的是"线性层输出的原始 logits"，Gemma 的 final_logit_softcapping
        # （tanh 软截断）是在 lm_head 调用之后、在 Gemma4ForCausalLM.forward 里额外做的，
        # 并不在 lm_head 子模块自己的 forward 内，所以这里 hook 到的是"软截断之前"的 logits。
        # 这一点在下面 "ours" 分支的 out_head hook 上是完全对称的（同样是软截断前），
        # 因此两边这条 "logits" trace 仍然是可比较的同一含义的量。
        handles.append(model.lm_head.register_forward_hook(hook("logits")))
    else:
        # "ours"（本仓库手写实现）结构：Gemma4DenseModel -> .tok_emb / .blocks[i] / .final_norm / .out_head。
        handles.append(
            model.tok_emb.register_forward_hook(
                # emb_dim ** 0.5：与 Gemma 官方"嵌入乘以 sqrt(hidden_size)"的缩放约定保持一致。
                scaled_embedding_hook("embedding", model.cfg["emb_dim"] ** 0.5)
            )
        )
        # 兼容不同版本 notebook 实现中 transformer block 列表的属性命名
        # （有的版本叫 `blocks`，有的历史版本可能叫 `trf_blocks`）。
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            blocks = getattr(model, "trf_blocks", None)
        if blocks is None:
            raise AttributeError("Could not locate Gemma 4 blocks on the local model.")
        for idx, block in enumerate(blocks):
            handles.append(block.register_forward_hook(hook(f"block_{idx}")))
        handles.append(model.final_norm.register_forward_hook(hook("final_norm")))
        # 同上，out_head 是线性层，得到的同样是"软截断（softcap）之前"的原始 logits。
        handles.append(model.out_head.register_forward_hook(hook("logits")))

    return traces, handles


def _layer_sort_key(name):
    """
    给层名生成一个用于排序的 key，使得最终打印报告时能按照
    "embedding -> block_0 -> block_1 -> ... -> final_norm -> logits" 这种
    符合模型实际前向顺序的逻辑顺序输出，而不是按字符串默认字典序
    （字典序会把 "block_10" 排在 "block_2" 前面，且 embedding/final_norm/logits
    这些非 "block_" 开头的名字顺序也会被打乱）。
    """
    if name == "embedding":
        return (0, 0)
    if name.startswith("block_"):
        # "block_0" -> split("_") -> ["block", "0"] -> 取索引 1 转成 int，按层号数值排序
        # （而不是按字符串排序），避免出现 "block_10" 排在 "block_2" 前面的问题。
        idx = int(name.split("_")[1])
        return (1, idx)
    if name == "final_norm":
        return (2, 0)
    if name == "logits":
        return (3, 0)
    # 兜底：不认识的层名统一排到最后，按名字本身排序（不应该在正常流程中出现）。
    return (4, name)


def layerwise_differences(ours, hf_model, input_ids, rtol=1e-5, atol=1e-5):
    """
    对 "ours" 与 "hf_model" 两个模型跑同一份 input_ids 的前向传播，
    并逐层比较两边记录下来的中间输出张量，返回一份逐层对比结果列表。

    参数：
        ours / hf_model：结构对应、权重相同的两个模型实例。
        input_ids：输入的 token id 张量，形状 (batch_size, seq_len)。
        rtol / atol：torch.allclose 判断"数值是否足够接近"时使用的相对/绝对容差。

    返回：
        一个 list，每个元素是一条 dict，记录某一层的比对状态（"ok" / "mismatch" /
        "shape_mismatch" / "missing"）以及形状、最大绝对误差、平均绝对误差等信息。
    """
    # 分别给两个模型挂上调试 hook，拿到各自的 traces 字典与 handle 列表。
    ours_traces, ours_handles = _attach_debug_hooks(ours, is_hf=False)
    hf_traces, hf_handles = _attach_debug_hooks(hf_model, is_hf=True)

    try:
        # inference_mode：比 no_grad 更严格的"纯推理"上下文，禁止构建反向图，
        # 既能省内存/加速，又能确保这里只是在做前向对比，不涉及任何梯度相关副作用。
        with torch.inference_mode():
            ours(input_ids)
            hf_model(input_ids, use_cache=False)
    finally:
        # 无论前向是否抛异常，都要把注册过的 hook 清理掉，避免 hook 常驻在模型上
        # （比如后续代码里再次调用这两个模型做别的用途时，会意外触发这些调试 hook）。
        for h in ours_handles + hf_handles:
            h.remove()

    # 取两边 trace 字典 key 的并集，保证即使某一层只在一边被记录到（理论上不应发生，
    # 但作为防御性设计），也能在报告里体现为 "missing" 状态而不是被静默忽略。
    layer_names = sorted(set(ours_traces) | set(hf_traces), key=_layer_sort_key)
    results = []
    for name in layer_names:
        ours_tensor = ours_traces.get(name)
        hf_tensor = hf_traces.get(name)

        if ours_tensor is None or hf_tensor is None:
            # 某一层只有一边记录到了输出，说明两个模型的子模块结构对不上（例如某一边
            # 少了某个子模块，或者 hook 没挂上），这种情况下没法比较数值，直接标记为 "missing"。
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
            # 形状都对不上，数值比较没有意义（甚至无法相减），先标记出来，方便优先排查
            # 是不是配置映射（如 head_dim、hidden_size 等）出现了不一致。
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

        # 形状一致时才计算逐元素绝对误差，用最大值和均值两个指标共同刻画差异程度：
        # max_diff 反映"最坏情况"下的偏差，mean_abs_diff 反映"整体平均"偏差。
        diff = (ours_tensor - hf_tensor).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        # torch.allclose 用 |a - b| <= atol + rtol * |b| 的标准判据判断是否"足够接近"，
        # 这是比"完全相等"更合理的浮点数比较方式。
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
    """
    把 `layerwise_differences` 返回的逐层比对结果列表，格式化成人类可读的多行文本报告，
    每一行对应一层，用 [OK] / [DIFF] / [SHAPE] / [MISSING] 前缀直观标出该层的比对状态，
    方便在终端里一眼扫出"第一层出问题的位置"。
    """
    lines = []
    # 再次按逻辑层序排序（而不是假设传入列表已经有序），保证报告输出顺序稳定、符合前向顺序。
    for diff in sorted(differences, key=lambda d: _layer_sort_key(d["name"])):
        if diff["status"] == "ok":
            # 数值在容差范围内一致：打印最大/平均绝对误差（应该是很小的数量级，如 1e-6 ~ 1e-8）。
            lines.append(f"[OK] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}")
        elif diff["status"] == "mismatch":
            # 形状一致但数值超出容差：说明这一层（或更早的上游层）实现有偏差，是重点排查对象。
            lines.append(f"[DIFF] {diff['name']}: max={diff['max_diff']:.2e}, mean={diff['mean_abs_diff']:.2e}")
        elif diff["status"] == "shape_mismatch":
            # 形状都不一致：通常是配置参数映射错误（如 head_dim / n_heads 等）。
            lines.append(f"[SHAPE] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}")
        else:
            # "missing"：某一层只在一边被记录到，说明模型结构对不上或 hook 未正确挂载。
            lines.append(f"[MISSING] {diff['name']}: ours={diff['ours_shape']}, hf={diff['hf_shape']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 作为独立脚本运行时的入口：先检查 Gemma4 相关类是否可用，不可用则直接给出明确的报错提示，
    # 而不是让后续代码在某个深层调用处才因为 None 而报出难以理解的异常。
    if Gemma4ForCausalLM is None or Gemma4TextConfig is None:
        raise SystemExit("Gemma 4 classes are unavailable; install a compatible transformers build or use temp/gemma-4.")

    # 从相邻 notebook 动态加载 "ours" 实现所需的类/函数定义。
    import_notebook_defs = load_notebook_defs()
    # 构造迷你调试配置。
    cfg = tiny_debug_config()

    # 构建权重相同的一对模型（ours 手写实现 + hf 官方实现）。
    ours_model, hf_model = build_gemma4_pair(import_notebook_defs, cfg)
    # 固定随机种子，保证每次运行生成的随机输入 input_ids 是可复现的
    # （注意：此时两个模型的权重已经在 build_gemma4_pair 中确定好了，
    #  这里的 seed 只影响下面这一行随机输入的生成，不影响权重初始化）。
    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), dtype=torch.long)
    # 跑一次前向并逐层比较。
    diffs = layerwise_differences(ours_model, hf_model, input_ids)
    # 打印可读报告，供人工检查每一层是否对齐。
    print(format_report(diffs))
