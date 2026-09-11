# 中文说明（模块级 docstring）：
# 本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
# 第 4 章「10_kv-sharing」示例代码的单元测试文件。
#
# 它属于「跨层 KV 共享（cross-layer KV sharing）」这一节的配套内容，用来验证同目录下
# 两个核心实现的正确性：
#   1. gpt_with_kv_mha.py         —— 标准的多头注意力（MHA）+ 逐层独立 KV cache 的 GPT 实现，
#                                     作为“正确答案”的基准（baseline）。
#   2. gpt_with_kv_sharing.py     —— 支持「跨层共享 KV」的 GPT 实现：只有前
#                                     n_kv_producing_layers 层会真正计算并缓存自己的 K、V，
#                                     其余层直接复用（共享）这些已经算好的 K、V，从而减少
#                                     KV cache 占用的显存/内存（这正是本节要讲的优化技巧，
#                                     类似 Gemma3/Gemma4 等模型中使用的做法）。
#   3. memory_estimator_kv_sharing.py —— 用来估算 MHA / GQA / KV 共享在不同配置下
#                                     KV cache 所占字节数的小工具函数。
#
# 测试要点：
#   - 当「产生 KV 的层数」等于「总层数」时（即没有真正做任何共享），KV 共享版模型的行为
#     必须和标准 MHA 模型完全一致（数值上、生成结果上都要一致），这是验证“共享机制不引入
#     额外副作用”的关键回归测试（regression test）。
#   - 当只有部分层产生 KV 时，那些不产生 KV 的层不应该有自己独立的 cache（cache_k/cache_v
#     应为 None），从而验证显存节省确实发生了。
#   - 内存估算函数的计算公式是否与手写的公式一致。
#
# 本文件不包含可执行的 __main__ 逻辑，是给 pytest（或类似测试框架）直接发现并运行的
# 测试用例集合：函数名以 test_ 开头即会被自动识别为一个测试用例。

import torch

# 从「标准 MHA + KV cache」实现中导入模型类和带缓存的文本生成函数，作为基准（baseline）。
from gpt_with_kv_mha import GPTModel as GPTModelMHA
from gpt_with_kv_mha import generate_text_simple_cached as generate_text_simple_cached_mha
# 从「支持跨层 KV 共享」的实现中导入同名功能，用于和基准做对比。
from gpt_with_kv_sharing import GPTModel as GPTModelKVSharing
from gpt_with_kv_sharing import generate_text_simple_cached as generate_text_simple_cached_sharing
# 导入 KV cache 显存占用估算函数，用于验证公式实现是否正确。
from memory_estimator_kv_sharing import calc_kv_bytes_total


def test_kv_sharing_matches_mha_when_all_layers_produce_kv():
    """
    回归测试：当 KV 共享模型的「产生 KV 的层数」(n_kv_producing_layers) 等于总层数时，
    模型中的每一层都会像标准 MHA 一样独立计算并缓存自己的 K、V，不发生任何跨层共享。

    这种极端配置下，KV 共享版模型（GPTModelKVSharing）在数学上应当与标准 MHA 模型
    （GPTModelMHA）完全等价——只要把同一份权重（state_dict）加载进去，两者在：
      1) 一次性前向传播（不使用 cache）得到的 logits；
      2) 使用 KV cache 增量式生成 token 序列的结果；
    这两方面都应该完全一致（或在数值精度范围内一致）。

    这个测试本身没有参数也没有返回值（pytest 风格的测试函数），通过内部的 assert
    语句来判断测试是否通过；任意一个 assert 失败都会导致测试失败。
    """
    # 定义一个很小的 GPT 配置，方便测试快速运行：
    # vocab_size=257  词表大小（不含特殊含义，纯粹为了测试选的数字）
    # context_length=8  最大上下文长度（序列长度上限）
    # emb_dim=32     词嵌入/隐藏层维度
    # n_heads=4      注意力头数
    # n_layers=3     Transformer block 层数
    # drop_rate=0.0  测试时关闭 dropout，保证结果可复现、可精确比较
    # qkv_bias=False 关闭 QKV 线性层的偏置，简化对比
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 3,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    # 在基础配置之上，为 KV 共享模型追加 n_kv_producing_layers 参数，并把它设为
    # 等于总层数 n_layers —— 也就是说“每一层都产生自己的 KV”，等价于没有共享。
    # 这是本测试的核心构造：用“退化到无共享”的场景来验证共享机制本身没有引入 bug。
    sharing_cfg = {**cfg, "n_kv_producing_layers": cfg["n_layers"]}

    # 固定随机种子，保证两个模型在“权重初始化的调用顺序/随机数消耗方式相同”的前提下
    # 具有可比较性（虽然下面很快会用 load_state_dict 把权重强制对齐，但设种子是好习惯）。
    torch.manual_seed(123)
    mha_model = GPTModelMHA(cfg)
    sharing_model = GPTModelKVSharing(sharing_cfg)
    # 关键步骤：把标准 MHA 模型的权重原样加载进 KV 共享模型。
    # strict=True 要求两边参数名/形状必须完全匹配（一一对应），
    # 这本身就验证了：当 n_kv_producing_layers == n_layers 时，
    # KV 共享模型的参数结构与标准 MHA 模型是完全相同的（没有多余或缺失的参数）。
    load_result = sharing_model.load_state_dict(mha_model.state_dict(), strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    # 切换到 eval 模式：关闭 dropout 等训练专用行为，保证两次前向传播结果确定、可比较。
    mha_model.eval()
    sharing_model.eval()

    # 构造一个形状为 (batch=1, seq_len=context_length) 的随机 token 序列，
    # 作为完整输入序列（既用于一次性前向，也用于后续截取出生成用的 prompt）。
    full_input = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]))
    # 分别对两个模型做“不使用 KV cache”的完整前向传播，得到 logits，
    # 形状均为 (batch=1, seq_len=context_length, vocab_size=257)。
    # 由于权重相同、结构等价（n_kv_producing_layers == n_layers），两者的输出应严格一致。
    expected_logits = mha_model(full_input, use_cache=False)
    actual_logits = sharing_model(full_input, use_cache=False)
    # 用非常严格的容差（atol=1e-6, rtol=0）比较两个 logits 张量，
    # 确认「无共享」配置下 KV 共享实现与标准实现在数值上等价。
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=1e-6)

    # 取完整序列的前 6 个 token 作为生成任务的 prompt，形状 (1, 6)，
    # 分别喂给两个模型的「带 KV cache 的增量式生成」函数，各自生成 2 个新 token。
    prompt = full_input[:, :6]
    expected_ids = generate_text_simple_cached_mha(
        model=mha_model,
        # 使用 .clone() 避免两次生成调用共享/修改同一份底层张量数据，
        # 保证两次调用互不干扰、结果可独立比较。
        idx=prompt.clone(),
        max_new_tokens=2,
        context_size=cfg["context_length"],
        use_cache=True,  # 开启 KV cache：每步只对新 token 做前向，复用历史 K/V，加速自回归生成
    )
    actual_ids = generate_text_simple_cached_sharing(
        model=sharing_model,
        idx=prompt.clone(),
        max_new_tokens=2,
        context_size=cfg["context_length"],
        use_cache=True,
    )
    # 由于两个模型在数学上完全等价，使用相同 prompt、相同贪心/采样策略（由被测函数内部决定）
    # 生成出来的 token id 序列也必须完全相同。
    assert torch.equal(actual_ids, expected_ids)


def test_only_producer_layers_store_kv_cache():
    """
    验证「跨层 KV 共享」的核心机制：只有前 n_kv_producing_layers 层会真正维护自己的
    KV cache（cache_k / cache_v 不为 None），其余层不产生、也不存储自己的 KV cache
    （cache_k / cache_v 应保持为 None），因为它们在前向传播中直接复用前面层产生的 KV。

    这正是本节「10_kv-sharing」相对于标准 MHA 能够节省显存的关键所在：
    层数越多、共享的层越多，需要缓存的 K/V 张量就越少，显存占用也就越低。
    """
    # 配置一个 4 层的模型，但只有前 2 层（n_kv_producing_layers=2）产生并缓存 KV，
    # 后 2 层将直接复用前 2 层算出来的 KV（具体共享方式由 GPTModelKVSharing 内部实现决定）。
    cfg = {
        "vocab_size": 257,
        "context_length": 8,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 4,
        "n_kv_producing_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }

    torch.manual_seed(123)
    model = GPTModelKVSharing(cfg)
    model.eval()

    # 构造一个 (batch=1, seq_len=6) 的随机输入序列。
    idx = torch.randint(0, cfg["vocab_size"], (1, 6))
    # 显式重置 KV cache：确保测试开始前所有层的 cache_k/cache_v 都是 None，
    # 避免受到模型创建时可能残留状态的影响，保证测试的可重复性。
    model.reset_kv_cache()
    # 开启 use_cache=True 做一次前向传播：这会让「产生 KV 的层」把本次计算出的 K、V
    # 写入各自的 cache_k / cache_v 缓冲区（首次调用时是整段 prompt 的 K、V）。
    # logits 形状：(batch=1, seq_len=6, vocab_size=257)。
    logits = model(idx, use_cache=True)

    # 基本健全性检查：确保前向传播过程数值稳定，没有出现 NaN（比如共享逻辑写错导致
    # 某些位置读取到未初始化/形状不匹配的张量时，很容易产生 NaN）。
    assert not torch.isnan(logits).any()
    # 前两层（下标 0、1）是「产生 KV 的层」，它们的 cache_k / cache_v 应该已经被填充，
    # 且序列长度维度（dim=1）应等于输入序列长度 idx.size(1)（这里是 6），
    # 即形状大致为 (batch, seq_len, n_heads, head_dim) 或等价的展开形式
    # （具体维度顺序以 gpt_with_kv_sharing.py 中的实现为准，这里只关心 seq_len 维度）。
    for block in model.trf_blocks[:2]:
        assert block.att.cache_k.size(1) == idx.size(1)
        assert block.att.cache_v.size(1) == idx.size(1)
    # 后两层（下标 2、3）不产生自己的 KV，只是复用前面层的结果，
    # 因此它们自身的 cache_k / cache_v 必须始终保持 None——
    # 这正是「共享」而非「各自缓存一份」所带来的显存节省的直接体现。
    for block in model.trf_blocks[2:]:
        assert block.att.cache_k is None
        assert block.att.cache_v is None


def test_memory_estimator_counts_cached_layers():
    """
    验证 memory_estimator_kv_sharing.calc_kv_bytes_total 函数计算出的
    KV cache 总字节数，与手动按公式推导出的期望值是否一致。

    KV cache 显存占用的基本公式（针对 n_cached_layers 个「产生 KV 的层」）：
        total_bytes = batch_size * context_length * head_dim * n_kv_heads
                      * 2 (K 和 V 各一份) * bytes_per_elem * n_cached_layers
    其中 head_dim = ceil(emb_dim / n_heads)。

    本测试用的参数里 emb_dim=32, n_heads=4，因此 head_dim = 32/4 = 8，
    这也是下面期望值公式中直接写死系数 8 的由来（8 == head_dim）。
    """
    # batch_size=1            单条序列
    # context_length=128      序列长度（决定要缓存多少个位置的 K/V）
    # emb_dim=32, n_heads=4   用于推出 head_dim = 32/4 = 8
    # n_kv_heads=1            假设采用类似 GQA/MQA 的设置，只有 1 个 KV 头
    #                         （远小于 n_heads=4，体现分组查询注意力节省 KV 显存的效果）
    # n_cached_layers=2       只有 2 层真正产生并缓存 KV（对应跨层共享场景）
    # bytes_per_elem=2        每个元素占用的字节数（例如 fp16/bf16 均为 2 字节）
    batch_size = 1
    context_length = 128
    emb_dim = 32
    n_heads = 4
    n_kv_heads = 1
    n_cached_layers = 2
    bytes_per_elem = 2

    # 调用被测函数，得到实际计算出的 KV cache 总字节数。
    actual = calc_kv_bytes_total(
        batch_size,
        context_length,
        emb_dim,
        n_heads,
        n_kv_heads,
        n_cached_layers,
        bytes_per_elem,
    )
    # 手动按公式计算期望值：
    # 8 这里就是 head_dim（= emb_dim // n_heads = 32 // 4），直接写成字面量 8
    # 是为了让测试断言独立于被测函数内部对 head_dim 的计算方式（避免测试和实现
    # 用同一段代码算 head_dim，从而“自己验证自己”失去测试意义）。
    # 乘以 2 表示 K 和 V 各占一份；乘以 n_cached_layers 表示只统计真正缓存 KV 的层数。
    expected = batch_size * context_length * 8 * n_kv_heads * 2 * bytes_per_elem * n_cached_layers
    assert actual == expected
