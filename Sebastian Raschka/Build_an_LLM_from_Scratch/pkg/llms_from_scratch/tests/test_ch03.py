# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是第 3 章（多头自注意力机制，Multi-Head Attention）对应的 pytest 单元测试文件。

测试目的：
    验证 `llms_from_scratch.ch03` 模块中实现的两个多头注意力类
    （手写实现 `MultiHeadAttention` 与基于 PyTorch 原生 API 的
    `PyTorchMultiHeadAttention`）能够正常进行前向传播，
    并且输出张量的形状（shape）符合预期，即
    (batch_size, seq_len, d_out)。

运行方式：
    在项目根目录下执行 `pytest tests/test_ch03.py` 即可运行本文件中的测试。
"""

# 从第 3 章模块中导入两个多头注意力实现：
#   - MultiHeadAttention：从零手写实现的多头自注意力
#   - PyTorchMultiHeadAttention：调用 PyTorch 原生 API 实现的"加分项"版本
from llms_from_scratch.ch03 import MultiHeadAttention, PyTorchMultiHeadAttention
import torch  # PyTorch 深度学习框架，用于构造张量与调用模型


def test_mha():
    """
    测试意图：
        验证 MultiHeadAttention（手写实现）与 PyTorchMultiHeadAttention
        （基于 PyTorch 内置算子的实现）在给定随机输入下均能正确完成
        前向传播，且输出的上下文向量（context vector）张量形状符合预期。

        注意：本测试中使用的是 `==` 比较表达式而非 `assert` 语句，
        因此该比较结果实际上不会导致测试失败（即使形状不匹配，
        pytest 也不会报错）。这是原始代码本身的写法，本次仅添加注释，
        不修改任何可执行逻辑。
    """

    context_length = 100  # 支持的最大上下文长度（序列长度上限），用于因果掩码等
    d_in = 256  # 输入向量的维度（embedding 维度）
    d_out = 16  # 输出向量的维度（多头注意力输出的总维度）

    # 实例化手写版多头注意力：num_heads=2 表示切分为 2 个注意力头，dropout=0.0 表示不使用随机丢弃
    mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)

    # 构造随机输入张量，形状为 (batch_size=8, seq_len=6, d_in=256)
    batch = torch.rand(8, 6, d_in)
    # 执行前向传播，得到上下文向量
    context_vecs = mha(batch)

    # 期望输出形状为 (8, 6, d_out)，即每个 batch、每个位置输出 d_out 维向量
    # 【bug 修复】原代码缺少 assert 关键字，比较结果被丢弃，测试形同虚设；此处补上 assert。
    assert context_vecs.shape == torch.Size([8, 6, d_out])

    # Test bonus class
    # 测试加分项类：基于 PyTorch 原生多头注意力实现（例如内部使用 nn.MultiheadAttention 或
    # scaled_dot_product_attention 等高效算子）
    mha = PyTorchMultiHeadAttention(d_in, d_out, num_heads=2)

    # 重新生成一批随机输入张量，形状同样为 (8, 6, d_in)
    batch = torch.rand(8, 6, d_in)
    # 执行前向传播
    context_vecs = mha(batch)

    # 再次校验输出形状是否为 (8, 6, d_out)
    # 【bug 修复】同上，原代码缺少 assert 关键字，此处补上使其成为真正的断言。
    assert context_vecs.shape == torch.Size([8, 6, d_out])
