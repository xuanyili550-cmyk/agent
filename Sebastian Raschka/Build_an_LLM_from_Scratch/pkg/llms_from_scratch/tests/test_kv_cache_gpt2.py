# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).

"""
本模块是针对“带 KV 缓存（KV Cache）版本”GPT-2 实现的 pytest 单元测试。

背景说明：
- 在自回归（autoregressive）文本生成中，每生成一个新 token，都需要用到之前所有
  token 在注意力（attention）计算中产生的 key/value 张量。如果不做缓存，每一步都要
  把“已生成的全部序列”重新喂给模型做一次完整前向计算，代价很高。
- KV 缓存的思路是：把历史 token 的 key/value 张量缓存起来，后续每步只需要输入
  “新的 1 个 token”，模型内部把新算出的 key/value 追加到缓存里，再和缓存中历史的
  key/value 一起做注意力计算，从而避免重复计算，显著提速。

本文件要验证的核心内容：
1. 模型对外暴露的“缓存生成”相关接口（如 current_pos、reset_kv_cache）行为是否正确；
2. 显式传入 KVCache 对象时，缓存路径是否被正确激活，且支持多步（step）连续调用；
3. 使用缓存（use_cache=True）与不使用缓存（use_cache=False）两种方式生成的结果
   在数值上应完全一致（这是最关键的正确性保证：缓存只是加速手段，不能改变输出）；
4. 非缓存生成不应“继承”上一次缓存生成遗留下来的位置状态（position state），
   即两种生成方式互不污染；
5. 连续两次使用缓存生成同一个 prompt，应当得到相同结果（说明每次生成前
   内部状态会被正确重置，不会累积上一次的缓存位置）；
6. 流式生成（generate_text_simple_stream，逐 token 产出）与非流式的
   缓存生成（generate_text_simple，一次性返回全部结果）在结果上应当一致。
"""

import pytest
import torch

from llms_from_scratch.kv_cache.generate import generate_text_simple, generate_text_simple_stream
from llms_from_scratch.kv_cache.gpt2 import GPTModel
from llms_from_scratch.kv_cache.utils import KVCache


# 测试专用的小型 GPT 配置，刻意把各维度设得很小（词表 96、上下文长度 16、
# 嵌入维度 32、4 个注意力头、2 层），这样测试运行快，也便于人工核对张量形状。
TEST_CONFIG = {
    "vocab_size": 96,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,  # 测试中关闭 dropout，保证前向计算是确定性的（可复现）
    "qkv_bias": False,
}


def make_model(seed=123):
    """
    构造一个用于测试的 GPTModel 实例。

    通过固定随机种子（torch.manual_seed）保证每次调用生成的模型权重是可复现的，
    并调用 .eval() 切换到评估模式（关闭 dropout 等训练专用行为），
    以便不同测试用例之间、以及“缓存版”与“非缓存版”生成结果可以做逐元素比较。
    """
    torch.manual_seed(seed)
    return GPTModel(TEST_CONFIG).eval()


def test_model_exposes_cached_generator_contract():
    """
    测试意图：
    验证 GPTModel 对外暴露的、与“缓存生成”相关的基本契约（接口约定）：
    - 模型应保存并暴露其配置字典 cfg，且应与传入的 TEST_CONFIG 一致；
    - 模型应有 current_pos 属性，用于记录当前已处理到序列的第几个位置
      （这是支持增量式/单步生成的关键状态）；
    - 调用 reset_kv_cache() 应当把 current_pos 重置为 0，
      即清空之前的生成进度，为一次全新的生成做准备。
    """
    model = make_model()

    # 断言：模型保存的配置应与我们传入的测试配置完全一致
    assert model.cfg == TEST_CONFIG
    model.current_pos = 7  # 人为设置一个非零的位置状态，模拟“已经生成过一些 token”
    model.reset_kv_cache()  # 调用重置方法
    assert model.current_pos == 0  # 断言：重置后位置状态应归零


def test_supplied_cache_activates_cache_path_and_supports_a_second_step():
    """
    测试意图：
    验证当外部显式传入一个 KVCache 对象给模型的 forward 调用时：
    1. 模型会走“启用缓存”的前向计算路径，返回的 logits 形状正确；
    2. 模型内部的 current_pos 会随着已处理的 token 数量正确递增；
    3. 缓存对象中，每一层保存的 key/value 张量在序列长度维度上
       会随着已处理的 token 数增长而同步增长；
    4. 该缓存可以在“第二步”（即只喂入新生成的 1 个 token）时继续被正确使用，
       验证多步连续调用（增量解码）的正确性。
    """
    model = make_model()
    # 创建一个与模型层数匹配的 KV 缓存对象，用于在多次 forward 调用间保存历史 key/value
    cache = KVCache(n_layers=TEST_CONFIG["n_layers"])
    # 构造一个长度为 4 的输入 prompt（batch_size=1）
    prompt = torch.tensor([[4, 8, 15, 16]])

    # 第一步：把整个 prompt 一次性喂给模型，同时传入缓存对象
    logits = model(prompt, cache=cache)

    # 断言：输出 logits 形状应为 (batch_size=1, seq_len=4, vocab_size)
    assert logits.shape == (1, 4, TEST_CONFIG["vocab_size"])
    # 断言：处理完 prompt 后，current_pos 应等于 prompt 的长度
    assert model.current_pos == prompt.shape[1]
    # 遍历缓存中每一层保存的 (keys, values)，检查其序列长度维度（第 2 维，索引为 2）
    for keys, values in cache.get_all():
        assert keys.shape[2] == prompt.shape[1]
        assert values.shape[2] == prompt.shape[1]

    # 从第一步的 logits 中取最后一个位置，贪心（argmax）选出下一个 token
    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    # 第二步：只把这 1 个新 token 喂给模型，复用同一个缓存对象（增量解码的典型用法）
    next_logits = model(next_token, cache=cache)

    # 断言：第二步输出形状应为 (batch_size=1, seq_len=1, vocab_size)，因为只输入了 1 个新 token
    assert next_logits.shape == (1, 1, TEST_CONFIG["vocab_size"])
    # 断言：current_pos 应在原有基础上再加 1
    assert model.current_pos == prompt.shape[1] + 1
    # 断言：缓存中每层的 key/value 序列长度应相应增加 1（新 token 的 key/value 被追加进去了）
    for keys, values in cache.get_all():
        assert keys.shape[2] == prompt.shape[1] + 1
        assert values.shape[2] == prompt.shape[1] + 1


# 使用 pytest 参数化，对不同的 batch_size 与 prompt_length 组合分别运行同一测试逻辑
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("prompt_length", [1, 4, 8])
def test_cached_generation_matches_uncached(batch_size, prompt_length):
    """
    测试意图（核心正确性测试）：
    对不同的 batch_size（1、2）和不同的 prompt_length（1、4、8）组合，
    验证“使用 KV 缓存生成”（use_cache=True）与“不使用缓存生成”（use_cache=False）
    在完全相同的模型和输入下，生成出的完整 token 序列必须逐元素完全相等。

    这是保证 KV 缓存实现正确性的核心测试：缓存机制只应带来速度上的优化，
    绝不能改变模型的数值输出结果。
    """
    model = make_model()
    # 随机生成一个形状为 (batch_size, prompt_length) 的整数 token 序列作为 prompt
    prompt = torch.randint(0, TEST_CONFIG["vocab_size"], (batch_size, prompt_length))

    # 不使用缓存的生成方式：每一步都对完整已生成序列重新做前向计算
    uncached = generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=False
    )
    # 使用缓存的生成方式：每一步只对新 token 做前向计算，复用历史 key/value
    cached = generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=True
    )

    # 断言：两种方式生成的完整序列（prompt + 新生成的 token）必须完全一致
    assert torch.equal(cached, uncached)


def test_uncached_generation_does_not_inherit_cache_position():
    """
    测试意图：
    验证“先做一次带缓存的生成，再做一次不带缓存的生成”时，
    不带缓存的这次生成不会受到前一次缓存生成所遗留的内部状态（如 current_pos）影响，
    即两种生成模式之间的状态是相互隔离的。

    验证方法：
    - 先对 model 做一次 use_cache=True 的生成（这会改变 model 内部的 current_pos 等状态）；
    - 紧接着对同一个 model 做一次 use_cache=False 的生成，得到 observed 结果；
    - 另外用一个全新的、状态干净的 fresh_model（加载相同权重）做一次 use_cache=False
      的生成，得到 expected 结果；
    - 如果 observed 与 expected 一致，说明第一次的缓存生成没有污染后续的非缓存生成。
    """
    model = make_model(seed=321)
    prompt = torch.tensor([[4, 8, 15, 16]])

    # 第一次：带缓存生成，此过程会修改 model 内部状态（如 current_pos）
    generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=True
    )
    # 第二次：对同一个 model 做不带缓存的生成，观察其结果是否受到第一次调用的影响
    observed = generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=False
    )

    # 构造一个全新的、状态未被污染的模型，并加载与 model 完全相同的权重
    fresh_model = GPTModel(TEST_CONFIG).eval()
    fresh_model.load_state_dict(model.state_dict())
    # 用这个“干净”的模型做同样的不带缓存生成，作为期望的基准结果
    expected = generate_text_simple(
        fresh_model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=False
    )

    # 断言：污染前后的非缓存生成结果应完全一致，证明状态没有被跨模式污染
    assert torch.equal(observed, expected)


def test_repeated_cached_generation_resets_position_state():
    """
    测试意图：
    验证对同一个模型、同一个 prompt，连续两次调用“带缓存生成”（use_cache=True）
    应当得到完全相同的结果。

    这说明每次调用 generate_text_simple(..., use_cache=True) 之前/之中，
    内部会正确地重置（reset）上一次生成遗留的缓存与位置状态（current_pos），
    不会出现“第二次生成时位置状态从上次结束处继续累加”从而导致结果偏差的问题。
    """
    model = make_model()
    prompt = torch.tensor([[4, 8, 15, 16]])

    # 第一次带缓存生成
    first = generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=True
    )
    # 对同一个模型、同一个 prompt 再次带缓存生成
    second = generate_text_simple(
        model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=True
    )

    # 断言：两次生成结果应完全一致，证明内部状态在每次生成前被正确重置
    assert torch.equal(first, second)


def test_streaming_generation_matches_cached_generation():
    """
    测试意图：
    验证“流式生成”接口 generate_text_simple_stream（逐个 token 以生成器/迭代器
    形式产出）与“非流式的缓存生成”接口 generate_text_simple(..., use_cache=True)
    （一次性返回完整生成序列）在结果上是等价的。

    验证方法：
    - 使用两个权重相同（reference_model 与 streaming_model 加载同一份 state_dict）
      但对象不同的模型实例，分别避免相互间的状态干扰；
    - reference_model 走非流式缓存生成得到 expected；
    - streaming_model 走流式生成，将逐个产出的 token 收集起来，
      与原始 prompt 拼接（torch.cat）后得到 observed；
    - 二者应完全一致。
    """
    reference_model = make_model()
    streaming_model = make_model()
    # 让两个模型加载完全相同的权重，确保后续比较的唯一变量是“流式 vs 非流式”这一实现差异
    streaming_model.load_state_dict(reference_model.state_dict())
    prompt = torch.tensor([[4, 8, 15, 16]])

    # 基准结果：使用非流式的、带缓存的生成接口，一次性拿到完整序列
    expected = generate_text_simple(
        reference_model, prompt.clone(), max_new_tokens=3, context_size=16, use_cache=True
    )
    # 使用流式生成接口，逐个收集生成器产出的新 token（每次产出应为新增的 1 个 token）
    streamed_tokens = list(
        generate_text_simple_stream(
            streaming_model,
            prompt.clone(),
            max_new_tokens=3,
            context_size=16,
        )
    )
    # 将原始 prompt 与流式产出的各个新 token 沿序列维度（dim=1）拼接，还原出完整序列
    observed = torch.cat([prompt, *streamed_tokens], dim=1)

    # 断言：流式生成拼接后的完整序列应与非流式缓存生成的结果完全一致
    assert torch.equal(observed, expected)
