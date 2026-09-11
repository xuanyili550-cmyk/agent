# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明（原文件无模块级 docstring，此处为新增的中文说明，不影响任何可执行逻辑）：

本文件是基于 Chainlit 构建的 Qwen3 “多轮对话”聊天界面示例（multiturn 版本）。
整体流程：
    1. 启动时（进程级别，只执行一次）：根据顶部的配置项选择 Qwen3 的模型规模、
       是否使用“推理/思考”版聊天模型、运行设备等，下载/加载权重，构建模型与分词器
       （tokenizer），并将模型放到目标设备（CPU/CUDA/MPS）上、设为 eval 模式。
    2. 每个新的浏览器会话开始时（@chainlit.on_chat_start）：初始化一个用于保存
       多轮对话历史（history）的会话级列表，并放入一条 system 角色的系统提示词。
    3. 用户每发送一条消息（@chainlit.on_message）：将当前这条用户消息文本编码为
       token id 序列，然后调用支持 KV 缓存的流式生成函数逐个 token 生成回复，
       并通过 Chainlit 的流式接口把生成的文本片段实时“打字机”式地推送到前端。

需要特别提醒的一点（仅作代码行为说明，不修改任何逻辑）：
    on_start 中创建了 chainlit.user_session 里的 "history" 列表并写入了系统提示词，
    但在下面的 on_message 函数体中，实际编码/送入模型的输入只是
    `message.content`（即“这一轮”用户刚发送的原始文本），并没有再次读取
    `chainlit.user_session.get("history")`、把历史轮次的用户/助手发言拼接进
    Prompt，也没有把新产生的用户消息或助手回复追加回 history 列表。
    也就是说，本文件中 history 变量目前只是被“初始化”了，但没有在生成时被
    实际使用或更新；是否能获得跨轮次的上下文记忆，完全取决于
    `generate_text_simple_stream` 所依赖的 KV 缓存机制在多次调用之间
    是否于模型内部被保留（该细节在 llms_from_scratch 包内部实现，
    本文件中未展示）。这一点在对照同目录下不使用 KV 缓存持续状态、而是显式
    通过 `build_prompt_from_history` 拼接历史文本的 qwen3-chat-interface.py
    版本时尤为明显。此处仅作为阅读代码时的说明，不对原代码做任何修改。
"""

import torch
import chainlit

# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
from llms_from_scratch.kv_cache.qwen3 import (
    # Qwen3Model：支持 KV 缓存（key-value cache）的 Qwen3 模型实现，
    # KV 缓存可以让自回归生成时无需每步都重新计算全部历史 token 的注意力，从而加速逐 token 的流式生成
    Qwen3Model,
    # Qwen3Tokenizer：Qwen3 专用分词器封装，内部可选地应用“聊天模板”（chat template），
    # 即按照 <|im_start|>role ... <|im_end|> 之类的特殊标记把普通文本包装成模型期望的对话格式
    Qwen3Tokenizer,
    # 从 HuggingFace Hub 的模型快照（snapshots）中下载模型权重文件
    download_from_huggingface_from_snapshots,
    # 把下载到的权重字典（state dict 形式）加载进已构建好的 Qwen3Model 结构中
    load_weights_into_qwen
)
from llms_from_scratch.kv_cache.generate import (
    # generate_text_simple_stream：基于 KV 缓存的“流式”自回归生成函数，
    # 每次调用会以生成器（generator）的形式逐个 yield 出新生成的 token，
    # 从而支持一边生成一边把内容推送到前端界面（打字机效果），而不是等全部生成完才一次性返回
    generate_text_simple_stream
)

# ============================================================
# EDIT ME: Simple configuration
# ------------------------------------------------------------
# 以下为可编辑的简单配置项，运行前可根据需要修改
# ============================================================
MODEL = "0.6B"            # options: "0.6B","1.7B","4B","8B","14B","32B","30B-A3B"
                          # 选择 Qwen3 模型的参数规模
REASONING = True          # True = "thinking" chat model, False = Base
                          # True 表示使用带“思考/推理”能力的聊天（instruct）模型并应用聊天模板；
                          # False 表示使用未经过对话微调的 Base 基础模型
DEVICE = "auto"           # "auto" | "cuda" | "mps" | "cpu"
                          # 运行设备："auto" 会自动探测可用的加速硬件
MAX_NEW_TOKENS = 38912
                          # 单次生成允许产生的最大新 token 数量上限
LOCAL_DIR = None          # e.g., "Qwen3-0.6B-Base"; None auto-selects
                          # 本地权重缓存目录；为 None 时按模型名与是否推理版自动生成目录名
# ============================================================


def get_qwen_config(name):
    """
    根据模型规模名称（如 "0.6B"、"1.7B" 等）动态导入并返回对应的 Qwen3 结构配置字典（QWEN3_CONFIG）。
    使用惰性导入（在函数内部按需 import）是为了避免一次性加载所有规模的配置。
    若传入未识别的名称，则抛出 ValueError。
    """
    if name == "0.6B":
        from llms_from_scratch.qwen3 import QWEN_CONFIG_06_B as QWEN3_CONFIG
    elif name == "1.7B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
    elif name == "4B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
    elif name == "8B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
    elif name == "14B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
    elif name == "32B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
    elif name == "30B-A3B":
        from llms_from_scratch.qwen3 import QWEN3_CONFIG_30B_A3B as QWEN3_CONFIG
    else:
        raise ValueError(f"Invalid model name: {name}")
    return QWEN3_CONFIG


def build_repo_and_local(model_name, reasoning, local_dir_arg):
    """
    根据模型规模、是否使用推理（对话）版本、以及用户是否手动指定本地目录，
    拼装出：
        1) HuggingFace 上的仓库 ID（repo_id），例如 "Qwen/Qwen3-0.6B" 或 "Qwen/Qwen3-0.6B-Base"；
        2) 本地权重存放目录名（local_dir）。
    若 reasoning 为 False（即使用 Base 基础模型），仓库名与本地目录名都会带上 "-Base" 后缀。
    """
    base = f"Qwen3-{model_name}"
    repo_id = f"Qwen/{base}-Base" if not reasoning else f"Qwen/{base}"
    local_dir = local_dir_arg if local_dir_arg else (f"{base}-Base" if not reasoning else base)
    return repo_id, local_dir


def get_device(name):
    """
    根据配置的设备名称字符串解析出实际使用的 torch.device。
    当 name 为 "auto" 时，按优先级依次探测 CUDA（NVIDIA GPU）、MPS（Apple Silicon GPU）、
    最后回退到 CPU；否则直接按用户显式指定的名称返回对应设备。
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif name == "cuda":
        return torch.device("cuda")
    elif name == "mps":
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_model_and_tokenizer(qwen3_config, repo_id, local_dir, device, use_reasoning):
    """
    构建 Qwen3 模型实例并加载预训练权重，同时构建配套的分词器（tokenizer）。

    参数说明：
        qwen3_config: 模型结构配置字典（层数、隐藏维度等）
        repo_id:      HuggingFace 上的模型仓库 ID，用于下载权重与分词器文件
        local_dir:    权重在本地缓存的目录
        device:       模型要放置到的计算设备
        use_reasoning: 是否为“推理/对话”模型；决定分词器是否应用聊天模板、
                       是否自动添加“生成提示”（generation prompt）以及是否开启思考模式

    返回：(model, tokenizer) 二元组。
    """
    model = Qwen3Model(qwen3_config)
    weights_dict = download_from_huggingface_from_snapshots(
        repo_id=repo_id,
        local_dir=local_dir
    )
    load_weights_into_qwen(model, qwen3_config, weights_dict)
    del weights_dict

    model.to(device)  # safe for all but required by the MoE model
    model.eval()

    tok_filename = "tokenizer.json"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tok_filename,
        repo_id=repo_id,
        # apply_chat_template：是否把输入文本自动包装成 Qwen 的聊天模板格式
        # （例如加上 <|im_start|>user ... <|im_end|> 等特殊标记），
        # 这里直接用 use_reasoning 控制：使用推理/对话模型时才需要套用聊天模板
        apply_chat_template=use_reasoning,
        # add_generation_prompt：是否在编码结果末尾自动追加“assistant”角色的生成提示头，
        # 提示模型接下来应输出助手的回复
        add_generation_prompt=use_reasoning,
        # add_thinking：是否为“思考”模型开启思考模式的特殊标记（例如 Qwen3 的 <think> 机制）
        add_thinking=use_reasoning
    )
    return model, tokenizer


# 以下几行是模块加载时（进程启动阶段）立即执行的初始化代码：
# 依次完成——选取模型结构配置、拼装仓库与本地目录名、解析运行设备、
# 下载/构建模型与分词器。注意这些全局变量（QWEN3_CONFIG、MODEL、TOKENIZER 等）
# 在整个 Chainlit 应用生命周期内只会被创建一次，被所有会话共享。
QWEN3_CONFIG = get_qwen_config(MODEL)
REPO_ID, LOCAL_DIR = build_repo_and_local(MODEL, REASONING, LOCAL_DIR)
DEVICE = get_device(DEVICE)
MODEL, TOKENIZER = get_model_and_tokenizer(QWEN3_CONFIG, REPO_ID, LOCAL_DIR, DEVICE, REASONING)


@chainlit.on_chat_start
async def on_start():
    """
    Chainlit 会话开始时的回调（每当有新的前端会话/新用户打开聊天页面时触发一次）。
    在当前会话的 user_session（会话级键值存储，不同浏览器会话相互隔离）中
    初始化一个名为 "history" 的列表，用来保存这一多轮对话的历史消息，
    并先放入一条 role 为 "system" 的系统提示词，为后续对话设定助手的行为基调。

    注意：如前面模块 docstring 中所说明的，这里创建的 history 列表在下面的
    on_message 函数中并未被再次读取或写回；该函数目前只是完成了“历史容器”的
    初始化动作。
    """
    chainlit.user_session.set("history", [])
    chainlit.user_session.get("history").append(
        {"role": "system", "content": "You are a helpful assistant."}
    )


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.

    中文说明：Chainlit 收到用户发送的每一条新消息时都会触发本函数。
    主要步骤：
        1) 将本轮用户输入的文本编码为 token id 序列；
        2) 新建一条空内容的输出消息并立即发送（用于后续持续追加流式内容）；
        3) 调用基于 KV 缓存的流式生成函数，每生成一个新 token 就解码并推送到前端，
           实现“打字机”式的流式输出效果；
        4) 生成结束后，调用 update() 让 Chainlit 前端消息最终定型/刷新。
    """
    # 1) Encode input
    # 中文：仅对“当前这一条”用户消息文本进行编码（若 TOKENIZER 开启了 apply_chat_template，
    # 会在编码时自动套用聊天模板并附加生成提示/思考标记）；
    # 注意这里没有拼接 on_start 中初始化的历史对话（history）
    input_ids = TOKENIZER.encode(message.content)
    # 将 python list 形式的 token id 转成 torch 张量，放到目标计算设备上，
    # 并通过 unsqueeze(0) 增加一个 batch 维度（模型期望输入形状为 [batch, seq_len]）
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    # 2) Start an outgoing message we can stream into
    # 中文：先创建一条内容为空字符串的 Chainlit 消息对象并发送出去，
    # 之后可以通过 stream_token 持续向这条消息里追加文本片段，实现流式展示
    out_msg = chainlit.Message(content="")
    await out_msg.send()

    # 3) Stream generation
    # 中文：generate_text_simple_stream 是一个生成器（generator），
    # 依赖 KV 缓存进行自回归解码，每次循环产出“新生成的一个 token”，
    # 直到达到 max_new_tokens 上限或遇到 eos_token_id（结束符）为止
    for tok in generate_text_simple_stream(
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=TOKENIZER.eos_token_id
    ):
        # tok 的形状通常带有 batch 维度，这里去掉 batch 维（squeeze(0)）得到单个 token id
        token_id = tok.squeeze(0)
        # 将 token id 解码为可读文本片段（可能是一个字词、子词或标点等）
        piece = TOKENIZER.decode(token_id.tolist())
        # 把解码出来的文本片段实时追加/推送到前端消息中，产生流式“打字机”效果
        await out_msg.stream_token(piece)

    # 4) Finalize the streamed message
    # 中文：流式生成结束后，调用 update() 通知前端该消息已经生成完毕，完成最终渲染/持久化
    await out_msg.update()
