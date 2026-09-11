# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是 pytest 单元测试文件，用于验证第 4 章（ch04）中实现的多种 GPT 模型变体
（标准版 GPTModel、加速版 GPTModelFast、以及带 KV 缓存的 GPTModelKV）
在配合不同的文本生成函数（普通逐步生成 generate_text_simple、
使用 KV 缓存加速生成 generate_text_simple_cached）时，
都能够对同一个输入产生完全一致（数值上可复现）的输出序列。

测试思路：
1. 使用固定的随机种子初始化模型，保证权重可复现；
2. 对每种「模型类 x 生成函数」的组合进行参数化测试；
3. 跳过不兼容的组合（例如不支持 KV 缓存的模型配合缓存版生成函数）；
4. 对给定的起始文本进行编码、生成，并与预先计算好的期望 token 序列做逐元素比对。
"""

# 导入标准版 GPT 模型（GPTModel）和加速版 GPT 模型（GPTModelFast）
from llms_from_scratch.ch04 import GPTModel, GPTModelFast
# 导入带 KV 缓存（Key-Value Cache）支持的 GPT 模型实现，并重命名为 GPTModelKV 以便区分
from llms_from_scratch.kv_cache.gpt2 import GPTModel as GPTModelKV
# 导入不使用缓存的、逐步生成文本的简单实现
from llms_from_scratch.ch04 import generate_text_simple
# 导入使用 KV 缓存加速的文本生成实现，并重命名以区分
from llms_from_scratch.kv_cache.generate import generate_text_simple as generate_text_simple_cached

import pytest
import torch
import tiktoken


# GPT-124M 规模模型的超参数配置字典，供下面所有被测模型共用
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size  词表大小
    "context_length": 1024,  # Context length  上下文（最大序列）长度
    "emb_dim": 768,          # Embedding dimension  词嵌入维度
    "n_heads": 12,           # Number of attention heads  注意力头数量
    "n_layers": 12,          # Number of layers  Transformer 层数
    "drop_rate": 0.1,        # Dropout rate  Dropout 比例
    "qkv_bias": False        # Query-Key-Value bias  QKV 线性层是否使用偏置
}


# 对模型类和生成函数分别做参数化：
# ModelClass 会依次取 GPTModel、GPTModelFast、GPTModelKV 三种模型实现
@pytest.mark.parametrize("ModelClass", [GPTModel, GPTModelFast, GPTModelKV])
# generate_fn 会依次取普通生成函数和带 KV 缓存的生成函数
@pytest.mark.parametrize("generate_fn", [generate_text_simple, generate_text_simple_cached])
def test_gpt_model_variants(ModelClass, generate_fn):
    """
    参数化测试：验证「模型类 x 生成函数」的各种组合都能生成与期望一致的 token 序列。

    参数：
        ModelClass: 被测的 GPT 模型类（标准版 / 加速版 / 支持 KV 缓存版）。
        generate_fn: 被测的文本生成函数（普通版 / KV 缓存加速版）。

    测试意图：
        - 只对「模型是否支持 KV 缓存」与「生成函数是否使用 KV 缓存」相匹配的组合进行实际测试；
        - 不匹配的组合直接跳过（视为不适用，而非失败）；
        - 对匹配的组合，固定随机种子构造模型，输入相同的起始文本进行生成，
          并断言生成结果与预先计算好的期望 token 序列完全一致，
          从而保证不同实现路径在数值上是等价、可复现的。
    """

    # Skip incompatible combinations
    # 跳过不兼容的组合：
    # 情况一：使用不带缓存的生成函数，但模型是支持 KV 缓存的模型（reset_kv_cache 属性为真）——不测试
    if generate_fn is generate_text_simple and getattr(ModelClass, "reset_kv_cache", False):
        return
    # 情况二：使用带缓存的生成函数，但模型并不支持 KV 缓存（没有 reset_kv_cache 属性或为假）——不测试
    if generate_fn is generate_text_simple_cached and not getattr(ModelClass, "reset_kv_cache", False):
        return

    # 固定随机种子，保证模型权重初始化结果可复现，从而生成结果也可复现
    torch.manual_seed(123)
    # 使用当前参数化得到的模型类，基于 GPT_CONFIG_124M 配置实例化模型
    model = ModelClass(GPT_CONFIG_124M)
    model.eval()  # disable dropout  切换为评估模式，关闭 dropout，保证结果确定性

    # 用于生成的起始文本（prompt）
    start_context = "Hello, I am"

    # 使用 GPT-2 的 BPE 分词器对起始文本进行编码
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    # 将编码后的 token id 列表转换为张量，并增加一个 batch 维度（unsqueeze(0)）
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # 打印分隔线及标题，标示以下是「输入」相关的调试信息
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # 调用当前参数化得到的生成函数，基于模型和输入 token，自回归生成 10 个新 token
    out = generate_fn(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    # 预先计算好的期望输出 token 序列（在固定随机种子下应当得到的确定性结果）
    expect = torch.tensor([
        [15496,   11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267,
         49706, 43231, 47062, 34657]
    ])
    # 核心断言：生成的输出必须与期望的 token 序列完全一致（逐元素相等），
    # 用于验证不同模型实现与生成函数组合在数值上的一致性/可复现性
    assert torch.equal(expect, out), "Generated output does not match expected output"
