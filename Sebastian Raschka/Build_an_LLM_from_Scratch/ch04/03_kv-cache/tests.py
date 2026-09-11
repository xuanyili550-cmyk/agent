# Code to test the GPT model implementation against the KV cache variants

"""
【中文说明】本文件是《从零构建大语言模型》(Build a Large Language Model from Scratch)
一书第 4 章「KV Cache（键值缓存）」扩展示例中的测试文件。

它的作用是验证三种 GPT 模型实现在数值上是否完全等价（即给定相同的随机种子和输入，
生成的 token 序列必须完全一致）：
    1. gpt_ch04.GPTModel               —— 第 4 章的“基础版” GPT 模型，不使用 KV 缓存，
                                           每一步都对完整的历史序列重新做前向传播。
    2. gpt_with_kv_cache.GPTModel      —— 引入 KV 缓存的“朴素版”实现（KV1）。
    3. gpt_with_kv_cache_optimized.GPTModel —— 在朴素版基础上做了工程优化的实现（KV2），
                                           例如支持“预填充分块（prefill chunking）”和
                                           固定大小的 KV 缓存窗口（kv_window_size），
                                           以避免缓存无限增长占用显存。

KV 缓存（KV Cache）是自回归生成（autoregressive generation）中的核心加速技巧：
在解码阶段，每生成一个新 token，我们只需要计算这个新 token 对应的 Key/Value，
并把它追加到之前所有 token 的 Key/Value 缓存中，而不需要对整个历史序列重新计算
Key 和 Value——这样可以把每一步的计算复杂度从 O(n^2) 降到 O(n)。

本文件通过 pytest 测试用例来保证：
    - 是否使用缓存、缓存实现是否做了优化，都不应该改变模型的“生成结果”，
      即模型的数学等价性必须成立（否则说明缓存实现存在 bug）。
    - 一些边界情况（如上下文长度溢出、预填充分块）在优化版实现中被正确处理。

这是一个纯粹的“回归测试/等价性测试”脚本，不包含任何训练逻辑。
"""

# --- 依赖导入 ---
import pytest          # 测试框架，用于组织和参数化测试用例
import torch            # PyTorch，深度学习张量计算库
import tiktoken          # OpenAI 开源的 GPT-2 分词器（tokenizer）库

# 第 4 章的基础 GPT 模型（无 KV 缓存）及其贪心解码函数
from gpt_ch04 import GPTModel as GPTModelBase
from gpt_ch04 import generate_text_simple

# 朴素 KV 缓存版本的 GPT 模型（KV1）
from gpt_with_kv_cache import GPTModel as GPTModelKV1
# 优化过的 KV 缓存版本的 GPT 模型（KV2，支持预填充分块 / 缓存窗口等特性）
from gpt_with_kv_cache_optimized import GPTModel as GPTModelKV2
# 对应的“使用缓存”的生成函数
from gpt_with_kv_cache import generate_text_simple_cached as generate_text_simple_cachedKV1
from gpt_with_kv_cache_optimized import generate_text_simple_cached as generate_text_simple_cachedKV2


# GPT-124M（约1.24亿参数）的标准超参数配置，与原书第4章保持一致
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # 词表大小，对应 GPT-2 BPE 分词器的词表规模
    "context_length": 1024,   # 模型支持的最大上下文长度（位置编码的最大长度）
    "emb_dim": 768,           # 词嵌入/隐藏层维度
    "n_heads": 12,            # 多头注意力的头数
    "n_layers": 12,           # Transformer block 堆叠的层数
    "drop_rate": 0.1,         # Dropout 比例，用于正则化
    "qkv_bias": False,        # 计算 Q/K/V 的线性层是否使用偏置项(bias)
    "kv_window_size": 1024  # NEW: KV cache window size
    # 中文：KV 缓存窗口大小（新增配置项）。它限制了 KV 缓存最多能保留多少个
    # 历史 token 的 Key/Value，超出窗口的旧缓存会被丢弃，从而控制显存占用，
    # 这是优化版（KV2）实现滑动窗口式 KV 缓存的关键参数。
}


# 自动检测运行设备：优先使用 GPU（cuda），否则退回 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("ModelClass", [GPTModelBase, GPTModelKV1, GPTModelKV2])
def test_gpt_model_equivalence_not_cached(ModelClass):
    """
    测试目的：在“不使用缓存”的生成路径下（即始终调用 generate_text_simple，
    每一步都对完整序列重新做前向传播），验证三种模型实现（基础版、KV1、KV2）
    在相同随机种子下初始化参数后，生成的 token 序列是否完全一致。

    这个测试的意义在于：即使 KV1/KV2 引入了 KV 缓存的代码路径，
    但当它们“不启用”缓存、走普通前向传播时，其网络结构和数学计算
    应该与基础版 GPTModelBase 完全等价（参数形状、初始化方式相同），
    从而生成完全相同的输出。

    参数:
        ModelClass: 通过 pytest.mark.parametrize 注入的模型类，
                    依次为 GPTModelBase / GPTModelKV1 / GPTModelKV2。

    返回值: 无（pytest 测试函数通过 assert 断言判定成功/失败）。
    """
    # 固定随机种子，保证不同模型实例的权重初始化完全一致，
    # 这样才能公平比较三种实现是否数值等价
    torch.manual_seed(123)

    model = ModelClass(GPT_CONFIG_124M).to(device)
    model.eval()  # 切换到评估模式：关闭 dropout，保证生成结果确定性可复现

    tokenizer = tiktoken.get_encoding("gpt2")  # 加载 GPT-2 的 BPE 分词器
    prompt = "Hello, I am"
    encoded = tokenizer.encode(prompt)  # 将文本编码为 token id 列表
    # 转成张量并增加 batch 维度：形状从 (seq_len,) 变为 (1, seq_len)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    # 用「模块名.类名」作为可读的模型标识，方便断言失败时定位是哪个实现出了问题
    model_name = ModelClass.__module__ + "." + ModelClass.__name__

    # 调用“无缓存”的贪心解码函数：每生成一个新 token，
    # 都会把完整序列（含新 token）重新喂给模型做一次完整前向传播，
    # 这是最朴素、最慢，但逻辑最简单、最不容易出错的基准实现
    token_ids = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=30,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    # 利用函数属性(results)在多次参数化调用之间“攒”结果，
    # 因为 pytest.mark.parametrize 会对同一个函数分别调用三次（每个 ModelClass 一次）
    if not hasattr(test_gpt_model_equivalence_not_cached, "results"):
        test_gpt_model_equivalence_not_cached.results = []

    test_gpt_model_equivalence_not_cached.results.append((model_name, token_ids))

    # 当三个模型的结果都收集齐后，才做最终的两两等价性比较
    if len(test_gpt_model_equivalence_not_cached.results) == 3:
        base_name, base_output = test_gpt_model_equivalence_not_cached.results[0]
        for other_name, other_output in test_gpt_model_equivalence_not_cached.results[1:]:
            # torch.equal 要求两个张量形状和每个元素值都完全相同，
            # 这里用来验证“无缓存路径”下三种实现是否生成了完全相同的 token 序列
            assert torch.equal(base_output, other_output), (
                f"Mismatch between {base_name} and {other_name}"
            )


@pytest.mark.parametrize("ModelClass", [GPTModelBase, GPTModelKV1, GPTModelKV2])
def test_gpt_model_equivalence_cached(ModelClass):
    """
    测试目的：验证每种模型在其「推荐/默认」的生成方式下（基础版走无缓存的
    generate_text_simple；KV1/KV2 走各自的“启用 KV 缓存”生成函数）
    所产生的 token 序列是否与其它实现完全一致。

    这是本文件中最核心的等价性测试：它证明了“使用 KV 缓存进行推理加速”
    这一工程优化，不会改变模型的生成结果（即缓存实现在数学上是正确的），
    这也是所有 KV 缓存实现都必须满足的正确性标准。

    参数:
        ModelClass: 通过 pytest.mark.parametrize 注入的模型类，
                    依次为 GPTModelBase / GPTModelKV1 / GPTModelKV2。

    返回值: 无（通过 assert 断言判定测试成功/失败）。
    """
    torch.manual_seed(123)

    model = ModelClass(GPT_CONFIG_124M).to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello, I am"
    encoded_tensor = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    model_name = ModelClass.__module__ + "." + ModelClass.__name__

    if ModelClass is GPTModelBase:
        # 基础版模型没有实现 KV 缓存，因此依然使用“无缓存”的生成函数，
        # 作为与 KV1/KV2“启用缓存”生成结果比较的基准（ground truth）
        token_ids = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=30,
            context_size=GPT_CONFIG_124M["context_length"]
        )
    elif ModelClass is GPTModelKV1:
        # KV1：朴素 KV 缓存实现。生成时会先对 prompt 做一次“预填充（prefill）”
        # 前向传播，得到并缓存所有历史 token 的 K/V；之后每生成一个新 token，
        # 只需计算该 token 的 Q/K/V，并把新的 K/V 追加到缓存中参与注意力计算，
        # 避免了对已处理过的 token 重复计算 K/V。
        token_ids = generate_text_simple_cachedKV1(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=30,
            context_size=GPT_CONFIG_124M["context_length"]
        )
    else:
        # KV2：在 KV1 的基础上做了工程优化（例如支持 kv_window_size 限制的
        # 滑动窗口缓存、以及输入过长时的“预填充分块”处理，见下面两个测试）。
        token_ids = generate_text_simple_cachedKV2(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=30,
            context_size=GPT_CONFIG_124M["context_length"]
        )

    # 同样利用函数属性收集三次参数化调用的结果，凑齐后再统一比较
    if not hasattr(test_gpt_model_equivalence_cached, "results"):
        test_gpt_model_equivalence_cached.results = []

    test_gpt_model_equivalence_cached.results.append((model_name, token_ids))

    if len(test_gpt_model_equivalence_cached.results) == 3:
        base_name, base_output = test_gpt_model_equivalence_cached.results[0]
        for other_name, other_output in test_gpt_model_equivalence_cached.results[1:]:
            # 核心断言：无缓存的基础版 与 使用 KV 缓存的 KV1/KV2 版本
            # 生成的 token 序列必须完全一致，证明 KV 缓存的引入是“数值无损”的加速。
            assert torch.equal(base_output, other_output), (
                f"Mismatch between {base_name} and {other_name}"
            )


def test_context_overflow_bug():
    """
    Test that demonstrates the ptr_current_pos overflow bug.

    In old implementation:
    - context_length = 10 (positions 0-9 available)
    - We try to generate 15 tokens total (5 input + 10 generated)
    - At token 11 (position 10), it crashes trying to access pos_emb[10]

    中文说明：
    这个测试用于验证“上下文长度溢出”这一历史 bug 是否已被修复。

    背景：模型的位置编码（positional embedding）表只有 context_length 行
    （这里是 10 行，对应位置索引 0~9）。旧版实现中，用于记录 KV 缓存当前
    写入位置的指针 ptr_current_pos 在生成过程中会不断自增；如果输入 token
    数 + 要生成的新 token 数超过了 context_length，指针最终会试图访问
    pos_emb[10]（下标越界），从而导致 IndexError 崩溃。

    这里构造的场景是：
        - context_length = 10  （位置编码只支持 0~9 共 10 个位置）
        - 输入 5 个 token + 要生成 10 个新 token = 总共 15 个 token
        - 超过 context_length（10），旧实现会在第 11 个 token（位置索引 10）崩溃

    该测试没有显式的 assert，其“通过”标准是：调用
    generate_text_simple_cachedKV2 时不抛出异常——如果优化版实现正确处理了
    这种溢出场景（例如通过滑动窗口截断历史，而不是无限增长位置索引），
    这里应当能顺利跑完而不报错。
    """
    GPT_CONFIG_SMALL = {
        "vocab_size": 50257,
        "context_length": 10,  # Very small context
        # 中文：故意设置一个非常小的上下文长度(10)，方便快速触发溢出场景
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "kv_window_size": 20  # Larger than context_length
        # 中文：KV 缓存窗口(20)故意设置得比 context_length(10)更大，
        # 用来测试当缓存窗口不构成限制时，模型自身对位置编码越界的处理是否正确
    }

    torch.manual_seed(123)

    model = GPTModelKV2(GPT_CONFIG_SMALL).to(device)
    model.eval()

    # 5 input tokens
    # 中文：随机生成 5 个输入 token，形状为 (batch=1, seq_len=5)
    input_tokens = torch.randint(0, 50257, (1, 5), device=device)

    generate_text_simple_cachedKV2(
        model=model,
        idx=input_tokens,
        max_new_tokens=10,  # 5 + 10 = 15 > 10 context_length
        # 中文：5(输入) + 10(新生成) = 15，超过了 context_length=10，
        # 用来触发/验证是否还存在“位置编码下标越界”的历史 bug
        context_size=GPT_CONFIG_SMALL["context_length"],
        use_cache=True  # 中文：显式开启 KV 缓存路径，这是本测试要验证的核心逻辑
    )


def test_prefill_chunking_basic():
    """
    Test that prefill correctly chunks input when input_length > kv_window_size.

    Setup:
    - kv_window_size = 4
    - input_length = 10
    - Should process in 3 chunks: [0:4], [4:8], [8:10]

    中文说明：
    这个测试验证“预填充分块（prefill chunking）”这一优化特性是否正确工作。

    背景：在启用 KV 缓存生成之前，需要先对输入的 prompt 做一次“预填充”
    （prefill）前向传播，把 prompt 中每个 token 的 Key/Value 计算出来并
    写入 KV 缓存。如果 prompt 长度超过了 kv_window_size（KV 缓存窗口大小），
    KV2（优化版）实现会把过长的输入拆分成多个小块（chunk），逐块喂给模型
    做前向传播、逐块更新 KV 缓存，而不是一次性把超长序列塞进去
    （这样可以让显存占用与窗口大小挂钩，而不是与输入长度挂钩）。

    本测试构造的场景：
        - kv_window_size = 4    （每个缓存窗口最多容纳 4 个 token 的 K/V）
        - input_length = 10     （输入 prompt 长度为 10，超过窗口大小）
        - 预期会被自动拆分成 3 个块处理：[0:4]、[4:8]、[8:10]

    断言的验证点：
        1. 分块预填充 + 后续生成之后，输出序列总长度应为
           输入长度(10) + 新生成长度(2) = 12。
        2. 输出序列的前 10 个 token 必须与原始输入 token 完全一致
           （即分块处理不应该篡改或丢失原始输入内容）。
    """
    config = {
        "vocab_size": 50257,
        "context_length": 20,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "kv_window_size": 4  # Small window to force chunking
        # 中文：故意设置一个很小的 KV 缓存窗口(4)，从而强制触发预填充分块逻辑
    }

    torch.manual_seed(123)
    model = GPTModelKV2(config).to(device)
    model.eval()

    # 10 input tokens (> kv_window_size of 4)
    # 中文：输入长度为 10，大于 kv_window_size(4)，因此预填充阶段
    # 必须被拆分成多个块依次处理；形状为 (batch=1, seq_len=10)
    input_tokens = torch.randint(0, 50257, (1, 10), device=device)

    # Should successfully process all input in chunks
    # 中文：调用启用缓存的生成函数，验证分块预填充逻辑能否顺利跑通并正确生成后续 token
    token_ids = generate_text_simple_cachedKV2(
        model=model,
        idx=input_tokens,
        max_new_tokens=2,
        use_cache=True
    )

    # Should have 10 input + 2 generated = 12 total
    # 中文：验证输出总长度 = 输入长度(10) + 新生成 token 数(2) = 12
    assert token_ids.shape[1] == 12, f"Expected 12 tokens, got {token_ids.shape[1]}"

    # First 10 tokens should match input
    # 中文：验证输出序列的前 10 个 token 与原始输入完全一致，
    # 证明“预填充分块”处理没有破坏或改变原始输入内容
    assert torch.equal(token_ids[:, :10], input_tokens), "Input tokens should be preserved"
