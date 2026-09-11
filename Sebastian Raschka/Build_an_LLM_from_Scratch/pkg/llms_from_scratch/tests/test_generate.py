# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是 pytest 单元测试文件。

测试目的：验证第 2 章实现的 `create_dataloader_v1` 数据加载器能够正常工作，
即从原始文本构建 (输入, 目标) 批次数据，并能与 token 嵌入层、位置嵌入层
正确组合，生成形状符合预期的输入嵌入张量（用于后续送入 GPT 模型）。
"""

from llms_from_scratch.ch02 import create_dataloader_v1

import os
import urllib.request

import pytest
import torch


# 使用 pytest 参数化装饰器，将测试语料文件名 "the-verdict.txt" 作为参数
# 传入测试函数，便于以后扩展为多个文件名的测试用例
@pytest.mark.parametrize("file_name", ["the-verdict.txt"])
def test_dataloader(tmp_path, file_name):
    """
    测试意图：
    1. 若本地不存在示例文本 "the-verdict.txt"，则从 GitHub 仓库下载；
    2. 读取文本内容后，使用 create_dataloader_v1 构建数据加载器；
    3. 从数据加载器中取出一个批次，分别通过 token 嵌入层和位置嵌入层
       计算嵌入向量并相加，得到最终的输入嵌入；
    4. 检查输入嵌入的形状是否符合预期 (batch_size, max_length, output_dim)。

    注意：tmp_path 是 pytest 内置夹具（本测试未直接使用其路径，
    仅作为参数占位，未修改原逻辑）。
    """

    # 若当前工作目录下不存在该语料文件，则从远程 URL 下载
    if not os.path.exists("the-verdict.txt"):
        url = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
        file_path = "the-verdict.txt"
        urllib.request.urlretrieve(url, file_path)

    # 以 UTF-8 编码读取文本内容到内存中
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 词表大小（与 GPT-2 使用的 BPE 分词器词表大小一致）
    vocab_size = 50257
    # 嵌入向量的维度
    output_dim = 256
    # 模型支持的最大上下文长度
    context_length = 1024

    # token 嵌入层：将 token id 映射为 output_dim 维的向量
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # 位置嵌入层：将位置索引映射为 output_dim 维的向量，用于注入位置信息
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    # 每个批次包含的样本数量
    batch_size = 8
    # 每个样本（序列）的长度
    max_length = 4
    # 构建数据加载器：
    # stride=max_length 表示滑动窗口步长等于序列长度，即窗口之间不重叠
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length
    )

    # 遍历数据加载器，取出第一个批次进行验证
    for batch in dataloader:
        x, y = batch  # x: 输入 token id 序列；y: 目标（下一个 token）序列

        # 计算输入序列对应的 token 嵌入，形状为 (batch_size, max_length, output_dim)
        token_embeddings = token_embedding_layer(x)
        # 计算位置嵌入：torch.arange(max_length) 生成 [0, 1, ..., max_length-1] 的位置索引
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        # 将 token 嵌入与位置嵌入相加，得到最终输入嵌入（位置嵌入通过广播机制应用到每个样本）
        input_embeddings = token_embeddings + pos_embeddings

        # 只取第一个批次即可完成验证，随后跳出循环
        break

    # 校验输入嵌入的形状是否为 (batch_size=8, max_length=4, output_dim=256)
    # 注意：此处使用的是比较表达式而非 assert 语句（保留原始代码逻辑，未做修改）
    input_embeddings.shape == torch.Size([8, 4, 256])
