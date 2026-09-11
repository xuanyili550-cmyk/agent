# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

"""
本文件用途（中文说明）
======================
这是《从零构建大语言模型》（Build a Large Language Model From Scratch）一书
第 5 章（ch05，预训练阶段）配套代码仓库中的**内部单元测试文件**，供作者/维护者
在提交代码前用 pytest 做回归测试，并不属于教学正文代码，读者学习时可以忽略。

文件里包含两类测试：
1. `test_main`：端到端地跑一遍 `gpt_train.py` 中的 `main` 训练流程（用很小的模型
   配置和很少的训练轮数），验证训练/验证损失、已见 token 数等返回值的长度是否
   符合预期，从而确保训练脚本本身没有被改坏。
2. `test_model_files`：联网校验 OpenAI 官方发布的 GPT-2（124M / 355M）预训练权重
   文件，以及本书作者提供的备用镜像（Backblaze B2）上的文件，是否仍然可以访问、
   文件大小是否与预期一致，用来提前发现「下载链接失效」或「文件被替换」等问题。

注意：这些测试需要联网（`test_model_files`）以及能够运行完整训练流程
（`test_main`，依赖 `gpt_train.py` 和相应数据），因此更适合在 CI 环境中运行，
而不是每次修改代码后手动逐条阅读。
"""

import pytest
from gpt_train import main  # 导入第5章训练脚本中的主训练函数，测试时直接调用它跑一遍完整训练流程
import requests  # 用于发送 HTTP HEAD 请求，检查远程模型权重文件是否可访问及文件大小是否正确

@pytest.fixture
def gpt_config():
    """
    pytest fixture：提供一份「迷你版」GPT 模型配置字典。

    与书中正式训练所用的 GPT-2 124M 配置相比，这里把 context_length、emb_dim、
    n_heads、n_layers 都调得很小，目的是让 test_main 中的端到端训练测试能在
    很短时间内跑完（提高测试效率），而不是为了得到一个真正有用的模型。

    返回值:
        dict: 包含 vocab_size / context_length / emb_dim / n_heads / n_layers /
              drop_rate / qkv_bias 等构建 GPTModel 所需的超参数。
    """
    return {
        "vocab_size": 50257,   # 词表大小，沿用 GPT-2 使用的 BPE 分词器词表大小
        "context_length": 12,  # small for testing efficiency  # 上下文长度调小，减少每个训练样本的序列长度，加快前向/反向传播
        "emb_dim": 32,         # small for testing efficiency  # 嵌入维度调小，大幅减少参数量
        "n_heads": 4,          # small for testing efficiency  # 注意力头数调小
        "n_layers": 2,         # small for testing efficiency  # Transformer 块层数调小，进一步压缩模型规模
        "drop_rate": 0.1,
        "qkv_bias": False
    }


@pytest.fixture
def other_settings():
    """
    pytest fixture：提供训练超参数（优化器学习率、训练轮数、批大小、权重衰减）。

    同样是为了让测试快速跑完，num_epochs 被设为 1（只训练一轮）。

    返回值:
        dict: 包含 learning_rate / num_epochs / batch_size / weight_decay。
    """
    return {
        "learning_rate": 5e-4,
        "num_epochs": 1,    # small for testing efficiency  # 只训练 1 个 epoch，避免单元测试耗时过长
        "batch_size": 2,
        "weight_decay": 0.1
    }


def test_main(gpt_config, other_settings):
    """
    端到端测试：调用 gpt_train.main() 跑一次完整的（迷你规模）预训练流程，
    并检查返回的训练/验证损失列表以及已处理 token 数列表的长度是否符合预期。

    该测试的意图是：一旦训练脚本 gpt_train.py 中的数据加载、批次划分、
    训练/验证循环逻辑被意外改动导致「记录评估次数」发生变化，这个测试就会失败，
    从而及时发现回归问题；而不是关心训练出来的模型效果本身。

    参数:
        gpt_config (dict): 由 gpt_config fixture 注入的迷你 GPT 模型配置。
        other_settings (dict): 由 other_settings fixture 注入的训练超参数。

    断言:
        train_losses / val_losses / tokens_seen 三个列表的长度都应为 39——
        这个具体数值取决于给定小数据集在给定 batch_size、eval_freq 下
        一共触发了多少次评估记录，属于「基线回归」式的硬编码断言。
    """
    train_losses, val_losses, tokens_seen, model = main(gpt_config, other_settings)

    assert len(train_losses) == 39, "Unexpected number of training losses"
    assert len(val_losses) == 39, "Unexpected number of validation losses"
    assert len(tokens_seen) == 39, "Unexpected number of tokens seen"


def check_file_size(url, expected_size):
    """
    向指定 URL 发送 HTTP HEAD 请求，校验远程文件是否可访问，
    以及其 Content-Length（字节数）是否与期望大小一致。

    之所以只发 HEAD 请求而不下载整个文件，是因为被测文件（GPT-2 预训练权重）
    体积可能高达几百 MB～1GB+，用 HEAD 请求既能拿到文件大小，又不必真正下载，
    大幅提升测试速度、节省带宽。

    参数:
        url (str): 待检查的文件下载地址。
        expected_size (int): 期望的文件大小（字节数）。

    返回值:
        tuple[bool, str]: (是否校验通过, 说明信息/错误信息)。
            - 状态码非 200：文件不可访问。
            - 缺少 Content-Length 响应头：无法判断大小。
            - 实际大小与期望不符：文件可能已被替换或损坏。
            - 请求过程抛异常（超时、网络错误等）：一并捕获并返回失败原因。
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)  # allow_redirects：有些托管服务会重定向到真实下载地址；timeout 防止网络异常导致测试卡死
        if response.status_code != 200:
            return False, f"{url} not accessible"

        size = response.headers.get("Content-Length")
        if size is None:
            return False, "Content-Length header is missing"

        size = int(size)
        if size != expected_size:
            return False, f"{url} file has expected size {expected_size}, but got {size}"

        return True, f"{url} file size is correct"

    except requests.exceptions.RequestException as e:  # 捕获所有 requests 相关异常（连接失败、超时、DNS 解析失败等），转成友好的失败信息而不是让测试直接崩溃
        return False, f"Failed to access {url}: {e}"


def test_model_files():
    """
    校验 GPT-2 预训练模型文件（124M 与 355M 两种规模）在两个不同数据源上
    是否都能正常访问、大小是否一致：
        1. OpenAI 官方发布地址（openaipublic.blob.core.windows.net）；
        2. 本书作者提供的备用镜像（Backblaze B2，用于官方源不可用/被墙时的替代下载）。

    该测试不训练模型、不下载完整权重，只用 HEAD 请求逐个核对每个文件
    （checkpoint、encoder.json、hparams.json、模型权重分片、词表文件等）
    的字节数，目的是尽早发现「下载链接失效」或「文件内容被替换」这类会导致
    读者在跟随本书教程时下载失败的问题。
    """
    def check_model_files(base_url):
        """
        针对给定的 base_url（模型托管根地址），依次校验 124M 和 355M 两种
        GPT-2 模型规模下，各个必需文件的可访问性与文件大小。

        参数:
            base_url (str): 模型文件托管的根路径，实际请求地址为
                             f"{base_url}/{model_size}/{file_name}"。
        """

        model_size = "124M"
        files = {
            "checkpoint": 77,
            "encoder.json": 1042301,
            "hparams.json": 90,
            "model.ckpt.data-00000-of-00001": 497759232,  # 124M 模型的权重数据分片，体积最大，用 HEAD 请求避免真正下载
            "model.ckpt.index": 5215,
            "model.ckpt.meta": 471155,
            "vocab.bpe": 456318
        }

        for file_name, expected_size in files.items():
            url = f"{base_url}/{model_size}/{file_name}"
            valid, message = check_file_size(url, expected_size)
            assert valid, message  # 任一文件校验失败就立即中断并报出具体的失败原因（哪个文件、期望/实际大小或不可访问）

        model_size = "355M"
        files = {
            "checkpoint": 77,
            "encoder.json": 1042301,
            "hparams.json": 91,
            "model.ckpt.data-00000-of-00001": 1419292672,  # 355M 模型体积明显更大，同样只做 HEAD 校验
            "model.ckpt.index": 10399,
            "model.ckpt.meta": 926519,
            "vocab.bpe": 456318
        }

        for file_name, expected_size in files.items():
            url = f"{base_url}/{model_size}/{file_name}"
            valid, message = check_file_size(url, expected_size)
            assert valid, message

    # 分别校验官方源和备用镜像源，两边都要保持文件一致、可下载
    check_model_files(base_url="https://openaipublic.blob.core.windows.net/gpt-2/models")
    check_model_files(base_url="https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2")
