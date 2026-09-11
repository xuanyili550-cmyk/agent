# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
模块级中文说明文档(docstring,新增)。

Qwen3 单轮/多轮聊天界面 —— 基于 Chainlit 搭建的网页聊天应用。

功能概述:
- 加载本仓库自带的 Qwen3 模型实现(支持 KV 缓存加速推理)及其配套分词器;
- 通过 Chainlit 提供的聊天网页界面接收用户输入的消息;
- 把完整的对话历史按照 Qwen(ChatML 风格)聊天模板拼接成一个提示词(prompt)字符串;
- 使用支持 KV 缓存的流式生成函数,逐个 token 生成回复并实时推送到网页前端;
- 生成结束后把助手的回复重新加入对话历史,供下一轮对话继续使用。

聊天模板说明(重点):
Qwen 系列模型采用 ChatML 风格的对话模板,每一轮对话都用特殊标记包裹:
    <|im_start|>角色名\n消息内容<|im_end|>\n
其中角色名可以是 "system"(系统提示)、"user"(用户)或 "assistant"(助手)。
在请求模型生成回复之前,还需要在历史文本末尾额外补上
"<|im_start|>assistant\n" 作为"生成提示(generation prompt)",
用来告诉模型接下来该以助手身份继续生成文本;生成会一直持续到模型输出
"<|im_end|>"(或 "<|endoftext|>")这两个结束符之一为止。

生成流程说明(重点):
1. 将本轮用户输入追加到会话历史(history)列表中;
2. 调用 build_prompt_from_history,把全部历史拼接成符合 ChatML 格式的字符串;
3. 用分词器(Tokenizer)把该字符串编码成 token id 序列,并按模型上下文长度做裁剪;
4. 调用支持 KV 缓存的流式生成函数 generate_text_simple_stream,逐个 token
   生成,一边生成一边通过 Chainlit 的消息对象把内容流式推送到网页前端,
   直至遇到结束符(EOS token)或达到最大新 token 数(MAX_NEW_TOKENS)为止;
5. 生成结束后,把助手的完整回复也追加进历史,供下一轮对话使用。

注意:本次改动仅新增中文注释与文档字符串(docstring),
未修改原文件任何可执行代码(逻辑、变量名、签名、字符串、缩进、导入等均保持不变)。
"""

import torch  # PyTorch:用于张量运算、设备管理(CPU/GPU/MPS)以及模型推理
import chainlit  # Chainlit:用于快速搭建聊天类 Web 应用的前端/后端框架

# For llms_from_scratch installation instructions, see:
# https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
# 中文说明:从本书配套的 llms_from_scratch 包中导入 Qwen3 相关组件
# (以下均为“kv_cache”版本,即带 KV 缓存优化、推理速度更快的实现)
from llms_from_scratch.kv_cache.qwen3 import (
    Qwen3Model,  # Qwen3 模型结构本体(Transformer 解码器)
    Qwen3Tokenizer,  # Qwen3 配套分词器,负责文本与 token id 之间的编解码
    download_from_huggingface_from_snapshots,  # 从 HuggingFace 快照下载模型权重
    load_weights_into_qwen  # 将下载好的权重字典加载进 Qwen3Model 实例
)
from llms_from_scratch.kv_cache.generate import (
    generate_text_simple_stream,  # 支持 KV 缓存的流式文本生成函数,逐 token yield 结果
    trim_input_tensor  # 按上下文长度裁剪输入 token 张量,防止超出模型最大上下文
)

# ============================================================
# EDIT ME: Simple configuration
# 中文说明:以下为可编辑的简单配置区,按需修改这几个变量即可切换模型规格/推理设备等
# ============================================================
MODEL = "0.6B"            # options: "0.6B","1.7B","4B","8B","14B","32B","30B-A3B"  # 中文:选择使用的 Qwen3 模型规格
REASONING = True          # True = "thinking" chat model, False = Base  # 中文:True 用带“思考(reasoning)”能力的对话模型,False 用基础(Base)模型
DEVICE = "auto"           # "auto" | "cuda" | "mps" | "cpu"  # 中文:推理设备,"auto" 表示自动检测(优先 GPU/MPS,否则用 CPU)
MAX_NEW_TOKENS = 38912  # 中文:单次生成允许产生的最大新 token 数量上限
LOCAL_DIR = None          # e.g., "Qwen3-0.6B-Base"; None auto-selects  # 中文:模型权重的本地缓存目录,None 表示按模型名自动选择目录名
# ============================================================


def get_qwen_config(name):
    """
    中文说明:根据传入的模型规格名称(如 "0.6B"、"1.7B" 等),
    从 llms_from_scratch.qwen3 模块中导入并返回对应的配置字典(QWEN3_CONFIG)。
    若传入了未定义的名称,则抛出 ValueError。
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
    中文说明:根据模型规格(model_name)、是否使用推理/思考模型(reasoning)
    以及用户手动指定的本地目录(local_dir_arg),推导出:
    1) HuggingFace 上对应的仓库 ID(repo_id);
    2) 本地缓存权重所使用的目录名(local_dir)。
    若 reasoning 为 False,则使用 "-Base" 后缀的基础模型仓库/目录。
    """
    base = f"Qwen3-{model_name}"
    repo_id = f"Qwen/{base}-Base" if not reasoning else f"Qwen/{base}"  # 中文:非推理(Base)模型使用带 -Base 后缀的仓库
    local_dir = local_dir_arg if local_dir_arg else (f"{base}-Base" if not reasoning else base)  # 中文:未手动指定本地目录时,按仓库同名自动生成
    return repo_id, local_dir


def get_device(name):
    """
    中文说明:根据配置的设备名称(name)选择并返回 torch.device。
    当 name 为 "auto" 时,自动按优先级检测:CUDA(NVIDIA GPU) > MPS(Apple Silicon) > CPU。
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")  # 中文:优先使用 NVIDIA GPU(CUDA)
        elif torch.backends.mps.is_available():
            return torch.device("mps")  # 中文:其次使用 Apple Silicon 的 MPS 加速
        else:
            return torch.device("cpu")  # 中文:都不可用时退回 CPU
    elif name == "cuda":
        return torch.device("cuda")
    elif name == "mps":
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_model_and_tokenizer(qwen3_config, repo_id, local_dir, device, use_reasoning):
    """
    中文说明:构建并返回可用于推理的 Qwen3 模型与分词器。

    具体步骤:
    1) 依据配置字典实例化 Qwen3Model 结构;
    2) 从 HuggingFace 快照下载(或读取本地缓存)权重字典;
    3) 把权重加载进模型,随后释放权重字典以节省内存;
    4) 将模型移动到目标设备并切换为 eval(推理)模式;
    5) 构建分词器 Qwen3Tokenizer,注意这里关闭了分词器自带的聊天模板
       (apply_chat_template=False)与自动生成提示(add_generation_prompt=False),
       因为聊天模板的拼装改由 build_prompt_from_history 手动完成,
       以避免对历史记录中的每一条消息重复包裹模板。
    """
    model = Qwen3Model(qwen3_config)
    weights_dict = download_from_huggingface_from_snapshots(
        repo_id=repo_id,
        local_dir=local_dir
    )
    load_weights_into_qwen(model, qwen3_config, weights_dict)  # 中文:把下载好的权重实际写入模型各层参数
    del weights_dict  # 中文:权重已加载进模型,释放这份字典以节省内存

    model.to(device)  # safe for all but required by the MoE model  # 中文:将模型搬到指定设备(CPU/CUDA/MPS),对 MoE 模型是必需操作
    model.eval()  # 中文:切换为推理模式(关闭 dropout 等训练专用行为)

    tok_filename = "tokenizer.json"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tok_filename,
        repo_id=repo_id,
        apply_chat_template=False,    # disable to avoid double-wrapping prompts in history  # 中文:关闭分词器自带的聊天模板,避免历史记录被重复包裹
        add_generation_prompt=False,  # we add the assistant header manually  # 中文:关闭自动生成提示,因为下面手动追加 assistant 头部
        add_thinking=use_reasoning  # 中文:是否为“思考(reasoning)”模式追加思考相关的特殊标记
    )
    return model, tokenizer


def build_prompt_from_history(history, add_assistant_header=True):
    """
    history: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    """
    # 中文说明(新增,原有英文 docstring 保持不变):
    # 该函数负责把整个对话历史(history)按照 Qwen 的 ChatML 聊天模板拼接成一段
    # 纯文本提示词(prompt),供分词器编码后送入模型生成。
    # ChatML 模板格式为:<|im_start|>角色\n内容<|im_end|>\n ,对历史中的每一条
    # 消息(系统/用户/助手)都会被这样包裹一遍,然后按顺序拼接在一起。
    # 当 add_assistant_header=True 时,会在拼接结果末尾额外补上
    # "<|im_start|>assistant\n"(不带结束符),作为让模型以助手身份继续生成的
    # “生成提示”,模型会从这里开始续写,直到输出 <|im_end|> 等结束符为止。
    parts = []
    for m in history:
        role = m["role"]  # 中文:取出该条消息的角色(system/user/assistant)
        content = m["content"]  # 中文:取出该条消息的文本内容
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")  # 中文:按 ChatML 模板包裹该条消息

    if add_assistant_header:
        parts.append("<|im_start|>assistant\n")  # 中文:追加“生成提示”,提示模型接下来以助手身份续写(不加 <|im_end|>,留给模型生成后自然结束)
    return "".join(parts)  # 中文:把所有片段拼接成最终的完整提示词字符串


# 中文说明:以下为模块加载时立即执行的全局初始化代码(只会在应用启动时运行一次)。
QWEN3_CONFIG = get_qwen_config(MODEL)  # 中文:按 MODEL 名称取出对应的模型结构配置字典
REPO_ID, LOCAL_DIR = build_repo_and_local(MODEL, REASONING, LOCAL_DIR)  # 中文:推导出 HuggingFace 仓库 ID 与本地缓存目录
DEVICE = get_device(DEVICE)  # 中文:确定实际使用的推理设备(覆盖前面字符串形式的 DEVICE 配置)
MODEL, TOKENIZER = get_model_and_tokenizer(QWEN3_CONFIG, REPO_ID, LOCAL_DIR, DEVICE, REASONING)  # 中文:加载模型与分词器(覆盖前面字符串形式的 MODEL 配置)

# Even though the official TOKENIZER.eos_token_id is either <|im_end|> (reasoning)
# or <|endoftext|> (base), the reasoning model sometimes emits both.
# 中文:因此这里把两种结束符的 token id 都收集起来,生成时只要遇到其中任意一个就停止。
EOS_TOKEN_IDS = (TOKENIZER.encode("<|im_end|>")[0], TOKENIZER.encode("<|endoftext|>")[0])


@chainlit.on_chat_start  # 中文:Chainlit 装饰器,注册“新会话开始”时的回调钩子
async def on_start():
    """
    中文说明:每当用户打开一个新的聊天会话(chat session)时被 Chainlit 调用。
    负责初始化该会话独立的对话历史(history),并预置一条系统提示(system prompt),
    为后续每一轮对话提供统一的“助手人设”上下文。
    """
    chainlit.user_session.set("history", [])  # 中文:在当前用户会话中新建一个空的历史列表
    chainlit.user_session.get("history").append(
        {"role": "system", "content": "You are a helpful assistant."}  # 中文:追加系统角色消息,设定助手的行为基调
    )


@chainlit.on_message  # 中文:Chainlit 装饰器,注册“收到用户新消息”时的回调钩子
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    # 中文说明(新增,原有英文 docstring 保持不变):
    # 这是每次用户在网页上发送一条消息时被 Chainlit 调用的主处理函数,
    # 完整实现了“追加历史 -> 拼接聊天模板 -> 编码 -> 流式生成 -> 更新历史”的一轮对话流程。

    # 0) Get and track chat history
    history = chainlit.user_session.get("history")  # 中文:取出当前会话已有的对话历史列表
    history.append({"role": "user", "content": message.content})  # 中文:把本次用户输入以 "user" 角色追加进历史

    # 1) Encode input
    # 中文:调用 build_prompt_from_history,将完整历史按 ChatML 模板拼接成一段
    # 提示词字符串,并在末尾追加 "<|im_start|>assistant\n" 作为生成提示,
    # 让模型接下来以助手身份续写回复。
    prompt = build_prompt_from_history(history, add_assistant_header=True)
    input_ids = TOKENIZER.encode(prompt)  # 中文:用分词器把拼接好的提示词字符串编码成 token id 列表
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)  # 中文:转成张量并增加 batch 维度(batch_size=1),放到目标设备上
    input_ids_tensor = trim_input_tensor(
        input_ids_tensor=input_ids_tensor,
        context_len=MODEL.cfg["context_length"],
        max_new_tokens=MAX_NEW_TOKENS
    )  # 中文:按模型上下文长度与最大新增 token 数,从左侧裁剪掉过长的历史 token,防止超出模型上下文窗口

    # 2) Start an outgoing message we can stream into
    out_msg = chainlit.Message(content="")  # 中文:创建一条初始为空内容的助手消息,用于后续流式追加文本
    await out_msg.send()  # 中文:先把这条空消息发送到前端,占位以便随后逐 token 流式更新

    # 3) Stream generation
    # 中文:调用支持 KV 缓存的流式生成函数,按最大新 token 数逐个生成 token id;
    # 每生成一个 token 就立即解码并推送到前端,实现打字机式的流式输出效果。
    for tok in generate_text_simple_stream(
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        # eos_token_id=TOKENIZER.eos_token_id
    ):
        token_id = tok.squeeze(0)  # 中文:去掉 batch 维度,得到单个 token id(标量张量)
        if token_id in EOS_TOKEN_IDS:  # 中文:命中任意一个结束符(<|im_end|> 或 <|endoftext|>)则停止生成
            break
        piece = TOKENIZER.decode(token_id.tolist())  # 中文:把这个 token id 解码回文本片段
        await out_msg.stream_token(piece)  # 中文:把该文本片段流式追加到前端已发送的消息上

    # 4) Finalize the streamed message
    await out_msg.update()  # 中文:流式生成结束后,通知前端该消息已完成更新

    # 5) Update chat history
    history.append({"role": "assistant", "content": out_msg.content})  # 中文:把模型生成的完整回复以 "assistant" 角色追加进历史,供下一轮对话使用
    chainlit.user_session.set("history", history)  # 中文:把更新后的历史写回当前用户会话,持久保存在本次会话生命周期内