# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块为第 2 章（ch02）内容编写的 pytest 单元测试。

测试目标：验证 `llms_from_scratch.ch02.create_dataloader_v1` 构建的数据加载器
能够正常工作——即能够从原始文本中正确切分出输入/目标 token 序列，并且这些
token 序列能够顺利喂给词元嵌入层（token embedding）和位置嵌入层
（position embedding），最终得到形状符合预期的输入嵌入张量。

测试所需的语料文件 "the-verdict.txt" 若本地不存在，会自动从 GitHub 仓库下载。
"""

from llms_from_scratch.ch02 import create_dataloader_v1

import os

import requests
import pytest
import torch


# 使用 pytest 参数化装饰器，声明测试要用到的语料文件名（此处只有一个取值）
@pytest.mark.parametrize("file_name", ["the-verdict.txt"])
def test_dataloader(tmp_path, file_name):
    """
    测试 create_dataloader_v1 构建的数据加载器是否能正确输出批次数据，
    并验证经过词元嵌入 + 位置嵌入相加后得到的输入嵌入张量形状是否符合预期。

    参数:
        tmp_path: pytest 内置夹具，提供一个测试专用的临时目录（本测试未直接使用，
                   但通过参数化机制被注入）。
        file_name: 由 @pytest.mark.parametrize 提供的测试语料文件名。
    """

    # 若当前目录下不存在语料文件，则从 GitHub 上下载该书配套的示例文本
    if not os.path.exists("the-verdict.txt"):
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        file_path = "the-verdict.txt"

        # 发起 HTTP GET 请求下载文件，超时时间设为 30 秒
        response = requests.get(url, timeout=30)
        # 若响应状态码表示请求失败，则抛出异常，测试会因此报错
        response.raise_for_status()
        # 以二进制写模式保存下载到的文件内容
        with open(file_path, "wb") as f:
            f.write(response.content)

    # 以 UTF-8 编码读取语料文件的全部文本内容
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 词表大小（对应 GPT-2 使用的 BPE 分词器词表规模）
    vocab_size = 50257
    # 嵌入向量的维度
    output_dim = 256
    # 上下文长度（模型一次能看到的最大 token 数量）
    context_length = 1024

    # 词元嵌入层：将 token id 映射为 output_dim 维的向量
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # 位置嵌入层：将位置索引映射为 output_dim 维的向量
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    # 每个批次包含的样本数量
    batch_size = 8
    # 每个样本（序列）的最大 token 长度
    max_length = 4
    # 构建数据加载器：将原始文本切分为 (输入, 目标) 序列对，并按批次输出
    # stride=max_length 表示滑动窗口步长等于序列长度，即样本之间不重叠
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length
    )

    # 遍历数据加载器，取出第一个批次进行验证
    for batch in dataloader:
        # x 为输入 token 序列，y 为对应的目标（下一个 token）序列
        x, y = batch

        # 将输入 token id 通过词元嵌入层，得到形状为 (batch_size, max_length, output_dim) 的张量
        token_embeddings = token_embedding_layer(x)
        # 生成位置索引 [0, 1, ..., max_length-1]，并通过位置嵌入层得到位置向量
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        # 词元嵌入与位置嵌入相加（广播机制），得到最终输入嵌入
        input_embeddings = token_embeddings + pos_embeddings

        # 只取第一个批次即可完成验证，跳出循环
        break

    # 校验输入嵌入张量的形状是否为 [batch_size, max_length, output_dim]
    # 【bug 修复】原代码为 `input_embeddings.shape == torch.Size([8, 4, 256])`，
    # 这只是一个比较表达式、缺少 assert 关键字，结果被直接丢弃，
    # 导致该测试形同虚设（即使形状不符也不会失败）。此处补上 assert 使其成为真正的断言。
    # 形状是确定性的，补 assert 后测试仍应通过。
    assert input_embeddings.shape == torch.Size([8, 4, 256])
