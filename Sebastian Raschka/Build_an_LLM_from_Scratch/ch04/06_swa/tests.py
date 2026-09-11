"""
本文件的中文说明（模块级 docstring，新增）
=====================================

文件用途：
    这是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
    第 4 章配套代码 `ch04/06_swa`（Sliding Window Attention，滑动窗口注意力）
    目录下的**单元测试文件**。

    它使用 `pytest` 风格的裸函数测试（函数名以 `test_` 开头即可被 pytest 自动发现），
    用来验证两件事：

    1. 带 KV 缓存（KV cache）且支持滑动窗口注意力（SWA）的多头注意力模块
       `MultiHeadAttentionWithSWA`（定义在同目录的 `gpt_with_kv_swa.py` 中），
       在“一次性喂入全部 token（不用缓存）”和“逐步喂入并使用 KV 缓存”两种
       前向传播方式下，最终计算结果在数值上是**完全等价**的（这是 KV 缓存
       正确性的核心验证：缓存只是计算上的加速/复用手段，不应该改变数学结果）。

    2. 当滑动窗口大小 `sliding_window_size` 设置为等于模型的完整上下文长度
       `context_length` 时，SWA 版本的 GPT 模型（`GPTModelSWA`）应该退化为
       普通的、没有窗口限制的因果自注意力 GPT 模型（`GPTModelBase`），也就是
       “窗口足够大 = 不加限制”。测试通过让两者共享同一份权重（state_dict），
       比较在相同输入下的 logits 输出和自回归生成的 token 序列是否完全一致。

    该文件在全书中的角色：
        它属于“工程实践/正确性回归测试”的辅助代码，帮助读者确认自己
        （或书中示例）实现的滑动窗口注意力 + KV 缓存机制在数学上是正确的，
        而不是仅仅“跑起来不报错”。对于正在学习 Transformer 注意力机制、
        KV 缓存和滑动窗口注意力（例如 Mistral、Gemma 等模型中使用的技术）
        的读者来说，通读这些测试断言，能帮助理解“正确实现”应满足的
        数值等价关系。
"""

import copy  # 用于对注意力模块做深拷贝，得到一个“无缓存版”的参照对象，避免两者共享同一份可变缓存状态

import torch  # PyTorch 主库，张量运算、随机种子控制、数值比较工具（allclose / testing.assert_close）等都来自这里

# 从书中第 4 章的基础实现里导入“无滑动窗口”的标准 GPT 模型和贪心自回归生成函数，作为“标准答案”参照
from llms_from_scratch.ch04 import GPTModel as GPTModelBase
from llms_from_scratch.ch04 import generate_text_simple
# 从本目录下的 gpt_with_kv_swa.py 导入“带 KV 缓存 + 滑动窗口注意力”的实现，作为被测对象
from gpt_with_kv_swa import GPTModel as GPTModelSWA
from gpt_with_kv_swa import MultiHeadAttentionWithSWA
from gpt_with_kv_swa import generate_text_simple_cached


def test_cached_prefill_matches_uncached_swa():
    """
    测试目标：
        验证 `MultiHeadAttentionWithSWA` 在“不使用 KV 缓存、一次性处理全部
        token”（use_cache=False）与“使用 KV 缓存、也是一次性喂入但走缓存
        分支”（use_cache=True）两种模式下，输出的注意力结果在数值上完全一致
        （在浮点误差范围内），并且缓存张量的长度被正确裁剪到滑动窗口大小。

    背景知识：
        - KV 缓存（KV cache）是自回归生成中的标准加速技巧：把历史 token 的
          Key/Value 张量缓存下来，新 token 到来时只需计算它自己的 Q/K/V，
          再与缓存拼接，从而避免每一步都重新计算全部历史的 K/V。
        - 滑动窗口注意力（Sliding Window Attention, SWA）进一步限制：
          每个 query 只关注最近 `sliding_window_size` 个 token 的 K/V，
          缓存也只需要保留最近的窗口大小那么多，从而节省显存。
        - 本测试中的“prefill”指的是把一个完整序列一次性喂给模型（而不是
          逐 token 生成），验证在 prefill 阶段，走缓存分支和不走缓存分支
          结果应当相同。

    参数：无（pytest 无参测试函数）
    返回值：无（通过 assert 断言表达测试是否通过；断言失败会抛出 AssertionError，
             pytest 会将其判定为测试失败）
    """
    torch.manual_seed(123)  # 固定随机种子，保证权重初始化和输入张量可复现，测试结果具有确定性

    att = MultiHeadAttentionWithSWA(
        d_in=8,               # 输入特征维度（embedding 维度）
        d_out=8,               # 输出特征维度（多头注意力总输出维度，需能被 num_heads 整除）
        dropout=0.0,           # 测试中关闭 dropout，避免随机性干扰数值比较
        num_heads=2,           # 注意力头数 -> 每个头的维度 head_dim = d_out / num_heads = 4
        sliding_window_size=4,  # 滑动窗口大小：每个 query 最多只能看到最近 4 个 token 的 K/V
    )
    att.eval()  # 切换到评估模式（主要是让 dropout 等层行为确定化，虽然这里 dropout=0）
    ref_att = copy.deepcopy(att)  # 深拷贝一份完全相同权重的模块，作为“不使用缓存”的参照对象
    # 之所以要 deepcopy 而不是直接复用同一个 att 对象，是因为 att 内部有可变的
    # cache_k / cache_v 缓冲区和 ptr_current_pos 指针，两种前向模式（有无缓存）
    # 会各自读写这些状态，必须隔离开，否则两次前向会互相污染对方的缓存状态。

    x = torch.randn(1, 6, 8)  # 构造随机输入，形状为 (batch=1, seq_len=6, d_in=8)

    # 参照结果：不使用缓存，一次性把全部 6 个 token 喂进去做标准的（带滑动窗口的）自注意力计算
    expected = ref_att(x, use_cache=False)  # 形状：(1, 6, 8) = (batch, seq_len, d_out)
    att.reset_cache()  # 显式重置 att 的 KV 缓存，确保下面用缓存模式跑的时候是从空缓存开始（防御性写法）
    # 被测结果：使用缓存模式，同样一次性喂入全部 6 个 token（相当于“prefill”场景）
    actual = att(x, use_cache=True)  # 形状：(1, 6, 8)

    assert not torch.isnan(actual).any()  # 数值健全性检查：确保没有出现 NaN（常见于掩码/除零等实现错误）
    # 核心正确性断言：两种前向路径算出的注意力输出应在数值上几乎完全相等（allclose 允许极小的浮点误差）
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)
    # 缓存长度检查：由于滑动窗口大小为 4，且本次一次性喂入长度为 6 的序列，
    # 缓存中保留的 K/V 长度应该被裁剪为窗口大小 4（只保留最近 4 个 token 的 K/V），
    # 而不是保留全部 6 个 token 的 K/V —— 这正是滑动窗口注意力节省显存的关键机制。
    assert att.cache_k.size(1) == 4  # cache_k 形状：(batch, cached_len, num_heads, head_dim)，这里校验 cached_len == 4
    assert att.cache_v.size(1) == 4  # 同理校验 cache_v 的缓存长度


def test_swa_matches_base_model_when_window_equals_context():
    """
    测试目标：
        验证当滑动窗口大小 `sliding_window_size` 等于模型的完整上下文长度
        `context_length` 时（即窗口足够大，覆盖了整个可能的序列长度），
        带滑动窗口注意力的 GPT 模型（GPTModelSWA）在行为上应当与不带滑动窗口
        限制的标准 GPT 模型（GPTModelBase）完全等价：
            1）加载同一份权重后，对同一批输入算出的 logits 应数值一致；
            2）用相同的贪心解码策略做自回归文本生成，得到的 token 序列也应完全一致。

    这本质上是在验证“滑动窗口注意力是标准因果注意力的一般化”：
        当窗口宽度 >= 序列长度时，SWA 应退化为普通的因果自注意力，
        不应引入任何额外的行为差异。

    参数：无
    返回值：无（通过一系列 assert 表达测试通过与否）
    """
    cfg = {
        "vocab_size": 257,        # 词表大小（决定 token embedding 和输出投影层的维度）
        "context_length": 8,      # 模型支持的最大上下文长度（序列长度上限）
        "emb_dim": 32,            # token/位置 embedding 维度，也是 Transformer block 的隐藏维度
        "n_heads": 4,             # 多头注意力的头数
        "n_layers": 2,            # Transformer block（含注意力+前馈网络）堆叠层数
        "drop_rate": 0.0,         # 关闭 dropout，保证两个模型比较时结果具有确定性、可比性
        "qkv_bias": False,        # Q/K/V 线性层不使用偏置项，与书中默认设置一致
    }
    swa_cfg = {
        **cfg,  # 复制基础配置的所有字段，保证除滑动窗口相关参数外，两个模型的结构完全一致
        "sliding_window_size": cfg["context_length"],  # 关键设置：窗口大小 == 完整上下文长度，使滑动窗口“形同虚设”
        "sliding_window_stride": -1,  # stride=-1 通常表示“不做窗口滑动/分块”的特殊标记，具体语义参见 gpt_with_kv_swa.py 实现
    }

    torch.manual_seed(123)  # 固定随机种子，保证两个模型初始化时使用相同的随机数流（虽然后面会用 state_dict 对齐权重，种子仍有助于可复现）
    base_model = GPTModelBase(cfg)      # 标准（无 SWA）GPT 模型，作为“标准答案”
    swa_model = GPTModelSWA(swa_cfg)    # 待测的 SWA 版 GPT 模型

    # 从标准模型的权重字典中筛掉每层注意力模块里的因果掩码 buffer（".att.mask"），
    # 因为掩码通常是 register_buffer 注册的非训练参数（例如上三角布尔/-inf 掩码矩阵），
    # 两个模型的掩码实现方式可能不同（例如 SWA 模型用滑动窗口逻辑动态生成掩码而非固定 buffer），
    # 所以不能、也不需要把这个 buffer 强行加载到 SWA 模型里。
    base_state = {
        key: value
        for key, value in base_model.state_dict().items()
        if not key.endswith(".att.mask")
    }
    # 将标准模型的权重（不含掩码 buffer）严格加载到 SWA 模型中；
    # strict=True 意味着除了我们主动过滤掉的 key 之外，其余所有参数名必须一一对应，
    # 否则会抛出异常，这样能尽早发现“两个模型结构其实并不一致”的问题。
    load_result = swa_model.load_state_dict(base_state, strict=True)
    assert not load_result.missing_keys      # 校验：SWA 模型中没有缺失应被赋值的参数
    assert not load_result.unexpected_keys    # 校验：传入的权重字典中没有 SWA 模型无法识别的多余 key

    base_model.eval()  # 评估模式：关闭 dropout 等训练态行为，保证前向结果确定
    swa_model.eval()

    # 构造一个长度正好等于 context_length（8）的随机 token 序列，形状：(batch=1, seq_len=8)
    full_input = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]))
    expected_logits = base_model(full_input)  # 标准模型前向，输出形状：(1, 8, vocab_size)
    # SWA 模型同样一次性喂入完整序列，且不使用 KV 缓存（相当于普通的整体前向），
    # 由于窗口大小 == 序列长度，理论上每个 query 依然能看到它左边的全部历史 token，
    # 因此结果应与标准因果自注意力完全一致。
    actual_logits = swa_model(full_input, use_cache=False)  # 形状同样为 (1, 8, vocab_size)
    # 使用更严格的比较工具 torch.testing.assert_close（相比 allclose，报错信息更详细，
    # 便于定位不一致的具体位置），这里要求几乎完全相等（atol=1e-6, rtol=0）。
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=1e-6)

    # 接下来验证自回归生成场景：取前 6 个 token 作为 prompt（留出 2 个 token 的生成空间，
    # 因为 context_length=8，生成 2 个新 token 后总长度正好达到 8，不会超出上下文长度）。
    prompt = full_input[:, :6]  # 形状：(1, 6)

    # 标准模型：使用书中第 4 章的朴素贪心解码函数逐 token 生成（每步都重新计算全部历史的前向，不使用缓存）
    expected_ids = generate_text_simple(
        model=base_model,
        idx=prompt.clone(),           # clone 一份，避免生成函数原地修改影响到 prompt 后续被复用
        max_new_tokens=2,             # 生成 2 个新 token
        context_size=cfg["context_length"],  # 生成时用于截断上下文的窗口大小
    )
    # SWA 模型：使用支持 KV 缓存的生成函数，逐 token 走“缓存 + 滑动窗口”路径生成
    actual_ids = generate_text_simple_cached(
        model=swa_model,
        idx=prompt.clone(),
        max_new_tokens=2,
        context_size=cfg["context_length"],
        use_cache=True,               # 开启 KV 缓存，验证“缓存 + 窗口==上下文长度”场景下生成结果依旧与标准实现一致
    )
    # 最终断言：两种完全不同的实现路径（无缓存的朴素解码 vs. 有缓存+滑动窗口的解码）
    # 在“窗口足够大”的前提下，应该生成完全相同的 token 序列，这是对整个生成管线端到端正确性的验证。
    assert torch.equal(actual_ids, expected_ids)
