# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块是《从零构建大语言模型》第7章（指令微调）配套的 Chainlit 聊天界面。

整体流程：
1. 模块导入时立即调用 get_model_and_tokenizer()，一次性加载 GPT-2 355M
   结构、经过指令微调后保存的权重文件（.pth）以及 GPT-2 的 BPE 分词器，
   并把模型搬到可用的计算设备（GPU 优先，否则 CPU）上。
2. 通过 @chainlit.on_message 注册的 main() 协程会在每次用户在网页聊天
   界面发送一条消息时被调用：
   - 把用户输入的文本套进训练时使用的“指令模板”里，拼成完整 prompt；
   - 用 ch05 的 generate() 基于该 prompt 做自回归生成，得到新增的 token id；
   - 把 token id 解码回文本，并用 extract_response() 去掉 prompt 部分和
     模板中的 "### Response:" 标记，只保留模型真正生成的回答；
   - 把回答通过 chainlit.Message 发送回聊天界面。

运行方式：需先在本文件所在目录的上一级 `01_main-chapter-code` 目录下，
通过运行 ch07.ipynb 生成 `gpt2-medium355M-sft.pth` 权重文件，然后在本
目录下执行 `chainlit run app.py` 启动网页聊天界面。
"""

from pathlib import Path
import sys

import tiktoken
import torch
import chainlit


# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from llms_from_scratch.ch04 import GPTModel  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)
from llms_from_scratch.ch05 import (  # 【bug修复】原为 Build_an_LLM_from_Scratch.*，该包并不存在(实测 ModuleNotFoundError)，已改回真实包名 llms_from_scratch(与仓库其余38个文件及安装说明一致)
    generate,          # 自回归文本生成函数（内部已包含 torch.no_grad()，无需外部再包一层）
    text_to_token_ids,  # 把字符串编码成模型可用的 token id 张量
    token_ids_to_text,  # 把模型输出的 token id 解码回可读字符串
)

# 【风险标注/不改动】优先使用 GPU（cuda），否则回退到 CPU；
# 若在 Apple Silicon 上运行，这里不会使用到 "mps" 后端，只是设备选择范围较窄，
# 不属于确定性 bug，是否需要支持 mps 由使用者按自身硬件决定，故不擅自修改。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    加载第7章指令微调后得到的 GPT-2（355M）模型与对应分词器。

    Code to load a GPT-2 model with finetuned weights generated in chapter 7.
    This requires that you run the code in chapter 7 first, which generates the necessary gpt2-medium355M-sft.pth file.

    返回：
        tokenizer: GPT-2 使用的 tiktoken BPE 分词器，用于文本<->token id 互转。
        model: 已加载指令微调权重、并放到目标 device 上的 GPTModel 实例。
        GPT_CONFIG_355M: 模型结构配置字典，供 main() 中生成时读取
            context_length 等参数。

    【风险标注/不改动】函数内没有调用 model.eval()。由于此处
    drop_rate 配置为 0.0，Dropout 在训练/评估模式下行为等价，
    因此当前配置下不会产生实际影响；但如果之后有人把 drop_rate 改为
    非 0 或模型里引入了 BatchNorm 等对模式敏感的层，忘记 eval() 就会
    变成真实的推理期 bug。这不是本次改动范围内的确定性 bug，故只标注、不修改。
    """

    # GPT-2 355M（medium）版本的结构超参数，需要与训练/微调时使用的配置完全一致，
    # 否则加载 state_dict 时会因形状不匹配而报错。
    GPT_CONFIG_355M = {
        "vocab_size": 50257,     # Vocabulary size 词表大小（GPT-2 BPE 词表）
        "context_length": 1024,  # Shortened context length (orig: 1024) 模型支持的最大上下文长度
        "emb_dim": 1024,         # Embedding dimension 词嵌入/隐藏层维度
        "n_heads": 16,           # Number of attention heads 多头注意力的头数
        "n_layers": 24,          # Number of layers Transformer 块的层数
        "drop_rate": 0.0,        # Dropout rate 推理阶段通常设为 0，避免随机性影响输出
        "qkv_bias": True         # Query-key-value bias 是否给 Q/K/V 线性层加偏置（需与预训练权重一致）
    }

    # 加载与 GPT-2 兼容的 BPE 分词器，用于后面把用户消息编码成 token id，
    # 以及把模型生成的 token id 解码回文本。
    tokenizer = tiktoken.get_encoding("gpt2")

    # 【风险标注/不改动】此处使用相对路径 ".."，要求必须在本文件所在目录
    # （06_user_interface）下执行 `chainlit run app.py`，否则会因为工作目录
    # 不同而找不到权重文件。这是运行方式上的约束，不是代码逻辑错误，故不修改。
    model_path = Path("..") / "01_main-chapter-code" / "gpt2-medium355M-sft.pth"
    if not model_path.exists():
        # 找不到权重文件时给出提示并直接退出，避免后面用未初始化的权重跑出无意义的回答。
        print(
            f"Could not find the {model_path} file. Please run the chapter 7 code "
            " (ch07.ipynb) to generate the gpt2-medium355M-sft.pth file."
            # 【bug修复】原代码此处写的是 "gpt2-medium355M-sft.pt"（缺少结尾的 "h"），
            # 与上面 model_path 实际检查/使用的文件名 "gpt2-medium355M-sft.pth" 不一致，
            # 是提示信息里的确定性拼写 bug：按提示里的文件名去核对，会误以为文件名
            # 不带 "h"。因为只是打印文案、不影响程序逻辑，属于安全的确定性修复，已改为
            # 与真实文件名一致的 ".pth"。
        )
        sys.exit()

    # weights_only=True 表示这里只加载张量参数（state_dict），不会反序列化任意 Python
    # 对象，是 PyTorch 官方推荐的更安全的加载方式。
    checkpoint = torch.load(model_path, weights_only=True)
    # 【风险标注/不改动】weights_only 参数是较新版本 PyTorch（2.0+）才提供的；
    # 若运行环境使用更旧版本的 PyTorch，torch.load 可能不识别该关键字参数而报错，
    # 这是跨版本兼容性风险，不属于本文件的确定性 bug，故只标注、不修改。
    model = GPTModel(GPT_CONFIG_355M)          # 按上面的结构配置构建一个空的 GPT 模型
    model.load_state_dict(checkpoint)          # 把微调好的参数灌入模型
    model.to(device)                           # 搬到 GPU（若可用）或 CPU 上

    return tokenizer, model, GPT_CONFIG_355M


def extract_response(response_text, input_text):
    """
    从模型完整的生成文本中，截取出真正属于“回答”的那一部分。

    参数：
        response_text: token_ids_to_text() 解码出的完整文本，通常是
            "prompt原文 + 模型续写内容"（因为 generate() 是在 prompt 后面继续生成的）。
        input_text: 原始 prompt 文本（即拼接指令模板后的用户输入）。

    做法：按 input_text 的长度把 response_text 前面对应的 prompt 部分切掉，
    只保留模型新生成的续写内容；再去掉模板中残留的 "### Response:" 标记文字，
    最后 strip() 掉首尾空白，得到干净的回答文本返回给聊天界面。
    """
    return response_text[len(input_text):].replace("### Response:", "").strip()


# Obtain the necessary tokenizer and model files for the chainlit function below
# 模块导入时就立刻加载一次模型和分词器（而不是每次收到消息都重新加载），
# 作为全局变量供下面的 main() 复用，避免重复加载权重带来的延迟。
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message  # 注册为 Chainlit 的消息回调：每当用户在网页界面发送一条消息，就会触发下面的 main()
async def main(message: chainlit.Message):
    """
    Chainlit 主回调函数：接收用户在网页聊天界面发送的一条消息，
    驱动模型完成“指令 -> 回答”的推理，并把结果发回界面。

    The main Chainlit function.

    流程：
        1. 固定随机种子，保证同样的输入每次生成的结果可复现；
        2. 把 message.content（用户输入的指令文本）套进训练时使用的
           指令模板，拼成完整 prompt；
        3. 用 generate() 基于 prompt 做自回归解码，得到新增的 token id；
        4. 把 token id 解码回文本，并用 extract_response() 截取出模型
           真正的回答部分；
        5. 通过 chainlit.Message(...).send() 把回答发送回聊天界面。
    """

    # 固定随机种子：generate() 内部若采用采样策略（如 top-k/温度采样）会用到
    # 随机数，这里固定种子是为了让同一条输入每次运行结果一致，便于复现/调试。
    torch.manual_seed(123)

    # 把用户输入嵌入到与第7章指令微调数据集一致的模板中，
    # 让模型看到与训练时相同风格的“指令提示”，从而更好地遵循指令作答。
    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """

    token_ids = generate(  # function uses `with torch.no_grad()` internally already 内部已用 no_grad，无需外部再包一层
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),  # The user text is provided via as `message.content` 把 prompt 编码成 token id 张量并放到模型所在设备上
        max_new_tokens=35,  # 最多新生成 35 个 token 作为回答（超过则截断）
        context_size=model_config["context_length"],  # 限制生成时可见的上下文窗口大小，需与模型结构配置一致
        eos_id=50256  # GPT-2 的结束符 token id，生成到该 token 时提前停止
    )

    # 把模型输出的 token id（prompt + 新生成部分）解码回可读文本
    text = token_ids_to_text(token_ids, tokenizer)
    # 从完整文本中去掉 prompt 部分和模板残留标记，只保留模型真正给出的回答
    response = extract_response(text, prompt)

    await chainlit.Message(
        content=f"{response}",  # This returns the model response to the interface 把最终回答内容发送回 Chainlit 网页聊天界面
    ).send()