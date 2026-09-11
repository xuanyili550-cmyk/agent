# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文文档字符串（模块功能说明）。

本文件是基于 Chainlit 构建的 GPT-2 文本生成聊天界面（原始版 app_orig.py）。

功能概述：
- 启动时加载 OpenAI 发布的预训练 GPT-2 权重（默认使用 "gpt2-small (124M)" 配置）；
- 使用 tiktoken 的 "gpt2" 编码器完成文本与 token id 之间的相互转换；
- 通过 Chainlit 的消息事件处理器 `main`，接收用户在网页聊天界面输入的文本，
  调用第 5 章实现的贪心解码函数 `generate` 生成后续文本，并将结果返回给前端展示。

注意：这是"原始版"实现，模型在模块导入时一次性加载（全局变量），
所有用户共享同一个模型实例，且未做会话隔离或多轮对话上下文管理。
"""

import tiktoken
import torch
import chainlit

# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# 中文说明：从本仓库自定义包中导入第4章实现的 GPTModel 模型类
from llms_from_scratch.ch04 import GPTModel  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)
# 中文说明：从第5章包中导入以下工具函数/方法：
# - download_and_load_gpt2：下载并加载 OpenAI 发布的 GPT-2 预训练权重
# - generate：基于贪心/温度采样等策略进行自回归文本生成
# - load_weights_into_gpt：将下载得到的权重加载进自定义 GPTModel 结构中
# - text_to_token_ids：将文本编码为模型输入所需的 token id 张量
# - token_ids_to_text：将模型输出的 token id 解码回可读文本
from llms_from_scratch.ch05 import (  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)
    download_and_load_gpt2,
    generate,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
)

# 中文说明：优先使用 GPU（CUDA）进行推理，如果没有可用的 CUDA 设备则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with pretrained weights from OpenAI.
    The code is similar to chapter 5.
    The model will be downloaded automatically if it doesn't exist in the current folder, yet.
    """
    """
    中文文档字符串：

    加载带有 OpenAI 预训练权重的 GPT-2 模型的代码。
    该代码与第 5 章中的实现类似。
    如果模型权重文件尚不存在于当前文件夹中，将会自动下载。

    返回值：
        tuple: (tokenizer, gpt, BASE_CONFIG)
            - tokenizer: tiktoken 的 GPT-2 编码器实例，用于文本与 token 的互相转换；
            - gpt: 已加载预训练权重、并放置到目标设备（CPU/GPU）上的 GPTModel 实例，
                   同时已经切换到 eval（推理）模式；
            - BASE_CONFIG: 合并了通用配置与所选模型规格后的最终模型配置字典。
    """

    CHOOSE_MODEL = "gpt2-small (124M)"  # Optionally replace with another model from the model_configs dir below
    # 中文说明：这里选定要使用的 GPT-2 规格，默认是最小的 124M 参数版本；
    # 可以按需替换为下面 model_configs 字典中列出的其他规格（medium/large/xl）

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    # 中文说明：基础配置字典，包含所有 GPT-2 规格共有的参数：
    # 词表大小、上下文长度、Dropout 比例（推理时设为0）、QKV 线性层是否使用偏置项

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    # 中文说明：不同规格 GPT-2 模型各自特有的结构参数：
    # 词嵌入维度 emb_dim、Transformer 层数 n_layers、注意力头数 n_heads

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    # 中文说明：从 CHOOSE_MODEL 字符串（如 "gpt2-small (124M)"）中解析出模型体积标记
    # 例如取出最后一个空格分隔的片段 "(124M)"，再去掉左右括号，得到 "124M"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    # 中文说明：将所选模型规格特有的参数（emb_dim/n_layers/n_heads）合并进基础配置中，
    # 得到该规格模型的完整配置

    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    # 中文说明：调用工具函数，按照 model_size（如 "124M"）自动下载（若本地不存在）
    # 并加载 OpenAI 官方发布的 GPT-2 权重；
    # settings 为模型的原始超参数设置，params 为权重参数（numpy 数组等形式）

    gpt = GPTModel(BASE_CONFIG)
    # 中文说明：使用完整配置实例化自定义的 GPTModel 网络结构（此时权重是随机初始化的）

    load_weights_into_gpt(gpt, params)
    # 中文说明：将下载得到的 OpenAI 预训练权重 params 逐层加载/拷贝到自定义 gpt 模型中

    gpt.to(device)
    # 中文说明：将模型迁移到之前确定的计算设备（GPU 或 CPU）上

    gpt.eval()
    # 中文说明：将模型切换为评估/推理模式，关闭 Dropout 等训练专用行为

    tokenizer = tiktoken.get_encoding("gpt2")
    # 中文说明：获取与 GPT-2 配套的 tiktoken 编码器，用于文本 <-> token id 的转换

    return tokenizer, gpt, BASE_CONFIG
    # 中文说明：返回分词器、加载好权重的模型、以及最终的模型配置字典


# Obtain the necessary tokenizer and model files for the chainlit function below
# 中文说明：在模块导入时（即应用启动时）立即执行一次模型与分词器的加载，
# 得到的 tokenizer、model、model_config 作为全局变量，供下面的 Chainlit 消息处理函数使用
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
# 中文说明：使用 Chainlit 提供的装饰器，将下面的异步函数注册为
# "收到用户消息时" 触发的事件处理器
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    """
    中文文档字符串：

    Chainlit 应用的主处理函数（消息事件回调）。

    每当用户在聊天界面发送一条消息时，Chainlit 框架会自动调用本函数，
    函数内部完成：将用户输入文本编码为 token、调用模型生成后续文本、
    再将生成结果解码为文本并发送回聊天界面。

    参数：
        message (chainlit.Message): Chainlit 封装的用户消息对象，
            其 `.content` 属性即为用户输入的原始文本字符串。
    """
    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        # 中文说明：调用生成函数，函数内部已经自带 `with torch.no_grad()` 上下文，
        # 因此这里无需再手动包裹禁用梯度计算的上下文
        model=model,
        # 中文说明：传入前面全局加载好的 GPT-2 模型
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # The user text is provided via as `message.content`
        # 中文说明：将用户输入文本（通过 `message.content` 获取）转换为 token id 张量，
        # 并搬运到目标计算设备（GPU/CPU）上，作为生成的起始上下文 idx
        max_new_tokens=50,
        # 中文说明：限定本次生成最多新增 50 个 token
        context_size=model_config["context_length"],
        # 中文说明：传入模型支持的最大上下文长度，用于生成时对输入序列做截断处理
        top_k=1,
        # 中文说明：top_k=1 表示每一步只从概率最高的1个候选 token 中选取，等价于贪心解码
        temperature=0.0
        # 中文说明：温度设为0，进一步确保采样退化为确定性的贪心解码（不引入随机性）
    )

    text = token_ids_to_text(token_ids, tokenizer)
    # 中文说明：将模型生成的 token id 序列解码回可读的文本字符串

    await chainlit.Message(
        content=f"{text}",  # This returns the model response to the interface
        # 中文说明：将生成的文本内容包装为 Chainlit 消息，f-string 格式化为字符串
    ).send()
    # 中文说明：异步发送该消息，前端聊天界面会展示这条模型回复
