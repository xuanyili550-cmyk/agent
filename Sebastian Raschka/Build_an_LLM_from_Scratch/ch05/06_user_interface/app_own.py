"""
中文模块说明：
本文件实现了一个基于 Chainlit 的聊天界面（Web 交互式对话应用），
用于加载“作者自己预训练”的 GPT 模型（即在第 5 章代码中训练/微调得到的模型），
并通过该模型对用户输入的文本进行续写（文本生成）。

运行前提：需要先运行第 5 章的代码（ch05.ipynb），
在 `01_main-chapter-code` 目录下生成 `model.pth` 权重文件，
否则本脚本在加载模型时会报错并退出。

以下保留原始英文注释：
Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
Source for "Build a Large Language Model From Scratch"
  - https://www.manning.com/books/build-a-large-language-model-from-scratch
Code: https://github.com/rasbt/LLMs-from-scratch
"""

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from pathlib import Path  # 用于处理文件路径（跨平台）
import sys  # 用于在模型文件缺失时退出程序

import tiktoken  # OpenAI 开源的分词器库，这里用于加载 GPT-2 的编码器
import torch  # PyTorch 深度学习框架
import chainlit  # Chainlit：用于快速搭建对话式 Web 界面的库

# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# 中文：从自定义包中导入第 4 章实现的 GPT 模型类
from llms_from_scratch.ch04 import GPTModel  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)
# 中文：从自定义包中导入第 5 章实现的文本生成相关工具函数
from llms_from_scratch.ch05 import (  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)
    generate,  # 文本生成函数（自回归解码）
    text_to_token_ids,  # 将文本转换为 token id 张量
    token_ids_to_text,  # 将 token id 张量转换回文本
)


# 中文：自动检测运行设备，若有可用 GPU（CUDA）则使用 GPU，否则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with pretrained weights generated in chapter 5.
    This requires that you run the code in chapter 5 first, which generates the necessary model.pth file.

    中文说明：
    该函数用于加载在第 5 章代码中训练/生成的、带有预训练权重的 GPT-2 模型。
    使用前必须先运行第 5 章的代码（ch05.ipynb），
    以生成本函数所需的 model.pth 权重文件。
    返回值为 (tokenizer分词器, model模型, GPT_CONFIG_124M模型配置字典) 三元组。
    """

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size  # 中文：词表大小（GPT-2 使用的 BPE 词表大小）
        "context_length": 256,  # Shortened context length (orig: 1024)  # 中文：上下文长度（此处缩短为 256，原始为 1024）
        "emb_dim": 768,         # Embedding dimension  # 中文：词嵌入维度
        "n_heads": 12,          # Number of attention heads  # 中文：多头注意力的头数
        "n_layers": 12,         # Number of layers  # 中文：Transformer 层数（层数越多模型越深）
        "drop_rate": 0.1,       # Dropout rate  # 中文：Dropout 比例，用于正则化防止过拟合
        "qkv_bias": False       # Query-key-value bias  # 中文：QKV 线性层是否使用偏置项
    }

    # 中文：加载 GPT-2 官方的 BPE 分词器（编码/解码文本用）
    tokenizer = tiktoken.get_encoding("gpt2")

    # 中文：拼接模型权重文件的路径，指向第 5 章主代码目录下的 model.pth
    model_path = Path("..") / "01_main-chapter-code" / "model.pth"
    if not model_path.exists():
        # 中文：若权重文件不存在，打印提示信息并直接退出程序
        print(f"Could not find the {model_path} file. Please run the chapter 5 code (ch05.ipynb) to generate the model.pth file.")
        sys.exit()

    # 中文：加载权重文件（weights_only=True 表示仅加载权重，更安全，避免执行任意代码）
    checkpoint = torch.load(model_path, weights_only=True)
    # 中文：根据配置字典实例化 GPT 模型结构
    model = GPTModel(GPT_CONFIG_124M)
    # 中文：将加载到的权重（state_dict）载入模型中
    model.load_state_dict(checkpoint)
    # 中文：将模型迁移到之前确定的设备（GPU 或 CPU）上
    model.to(device)

    return tokenizer, model, GPT_CONFIG_124M


# Obtain the necessary tokenizer and model files for the chainlit function below
# 中文：在模块加载时就获取好分词器、模型以及模型配置，供下方 chainlit 回调函数使用
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message  # 中文：注册为 Chainlit 的消息回调函数，每当用户发送一条消息时会自动触发
async def main(message: chainlit.Message):
    """
    The main Chainlit function.

    中文说明：
    Chainlit 应用的核心异步回调函数。
    当用户在聊天界面中发送消息（message）后，本函数会被自动调用，
    负责将用户输入的文本转为 token，交给模型进行自回归续写生成，
    再将生成结果转换回文本并发送回聊天界面展示给用户。
    """
    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        # 中文：generate 函数内部已经使用了 `with torch.no_grad()`，因此这里无需再手动关闭梯度计算
        model=model,  # 中文：传入之前加载好的 GPT 模型
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # The user text is provided via as `message.content`
        # 中文：将用户发送的文本（message.content）通过分词器转换为 token id 张量，并移动到指定设备上
        max_new_tokens=50,  # 中文：最多生成 50 个新 token
        context_size=model_config["context_length"],  # 中文：使用模型配置中的上下文长度作为生成时的上下文窗口大小
        top_k=1,  # 中文：top-k 采样参数，设为 1 表示每一步都只取概率最高的那个 token（贪心解码）
        temperature=0.0  # 中文：采样温度设为 0，进一步保证生成结果是确定性的（不引入随机性）
    )

    # 中文：将模型生成的 token id 序列解码回可读文本
    text = token_ids_to_text(token_ids, tokenizer)

    await chainlit.Message(
        content=f"{text}",  # This returns the model response to the interface
        # 中文：将模型生成的文本内容作为回复，返回给聊天界面
    ).send()  # 中文：异步发送该消息到前端界面显示
