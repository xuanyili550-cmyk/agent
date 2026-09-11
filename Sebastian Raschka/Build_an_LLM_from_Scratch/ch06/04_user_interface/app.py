# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""第6章「分类微调」的交互式聊天前端（基于 Chainlit）。

加载第6章微调得到的 GPT-2 垃圾/正常评论分类器（review_classifier.pth），
把用户在网页界面输入的文本实时判为某一类别并返回结果。
运行方式：先跑完 ch06.ipynb 生成权重文件，再用 `chainlit run app.py` 启动界面。
"""

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit

# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from llms_from_scratch.ch04 import GPTModel  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)
from llms_from_scratch.ch06 import classify_review  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 GPU 用 GPU，否则退回 CPU


def get_model_and_tokenizer():
    """
    Code to load finetuned GPT-2 model generated in chapter 6.
    This requires that you run the code in chapter 6 first, which generates the necessary model.pth file.

    中文说明：加载第6章微调好的 GPT-2 分类模型与分词器。
    返回 (tokenizer, model)。需先运行 ch06.ipynb 生成 review_classifier.pth 权重文件。
    """

    # GPT-2 124M 的结构配置（须与训练时完全一致，否则权重无法对上）
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")  # 使用与 GPT-2 相同的 BPE 分词器

    # 权重文件位于相邻的主章节代码目录；找不到则提示先运行 ch06 生成，再退出
    model_path = Path("..") / "01_main-chapter-code" / "review_classifier.pth"
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. Please run the chapter 6 code"
            " (ch06.ipynb) to generate the review_classifier.pth file."
        )
        sys.exit()

    # Instantiate model
    model = GPTModel(GPT_CONFIG_124M)  # 先按 GPT-2 结构实例化（此时输出头还是语言模型头）

    # Convert model to classifier as in section 6.5 in ch06.ipynb
    # 按书中 6.5 节把语言模型头替换成二分类头：把最后一层换成 emb_dim -> 2 的线性层
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)

    # Then load model weights
    # 载入微调后的分类器权重；weights_only=True 更安全（只反序列化张量，不执行任意对象）
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # 切到评估模式（关闭 dropout 等），推理时必须

    return tokenizer, model


# Obtain the necessary tokenizer and model files for the chainlit function below
# 模块导入时就把模型/分词器加载好（只加载一次），供下面每次消息回调复用
tokenizer, model = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.

    中文说明：Chainlit 的消息回调——每当用户在界面发一条消息就触发一次。
    把用户输入交给 classify_review 分类，并把类别标签返回到界面。
    """
    user_input = message.content  # 取出用户输入的文本

    # 调用第6章的分类推理函数：截断到最多 120 个 token 后前向，返回预测类别标签
    label = classify_review(user_input, model, tokenizer, device, max_length=120)

    await chainlit.Message(
        content=f"{label}",  # This returns the model response to the interface（把分类结果发回界面）
    ).send()