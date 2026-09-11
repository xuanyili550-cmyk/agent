# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
本模块对应《从零构建大语言模型》第 7 章"指令微调"（Instruction Fine-Tuning）的可复用代码。

主要内容包括：
1. `download_and_load_file`：下载并加载指令微调数据集（JSON 格式）。
2. `format_input`：把一条原始数据（instruction/input/output）格式化为
   Alpaca 风格的提示词文本。
3. `InstructionDataset`：PyTorch `Dataset`，在初始化时预先把所有样本
   分词（tokenize），避免训练时重复分词。
4. `custom_collate_draft_1` / `custom_collate_draft_2` / `custom_collate_fn`：
   三个逐步演进的 DataLoader collate 函数，依次展示：
     - 草稿1：仅做动态 padding，生成 inputs；
     - 草稿2：在草稿1基础上，额外构造"整体右移一位"的 targets，
       用于下一个 token 预测的语言模型训练目标；
     - 正式版：在草稿2基础上，引入 `ignore_index` 机制，把 targets 中
       多余的 padding token 替换为 -100，使得计算交叉熵损失时会被
       自动忽略（PyTorch `CrossEntropyLoss` 默认 `ignore_index=-100`），
       并支持可选的最大长度截断。
5. `check_if_running`：检查本机是否有指定名称的进程在运行（例如检查
   Ollama 服务是否已启动）。
6. `query_model`：向本地 Ollama 服务发送对话请求，用于调用本地 LLM
   （如 llama3）对模型回答进行打分。
7. `generate_model_scores`：批量调用 `query_model`，对一批模型生成的
   回答进行 0~100 分的评分，用于评估指令微调后模型的效果。
"""

import json
import os
import psutil
import requests

import torch
from tqdm import tqdm
from torch.utils.data import Dataset


def download_and_load_file(file_path, url):
    """
    下载指令微调数据集（JSON 文件）并加载为 Python 对象。

    如果本地 `file_path` 不存在该文件，则从 `url` 下载文本内容并写入
    本地文件；如果文件已存在，则直接跳过下载，复用本地缓存。
    最终无论哪种情况，都会重新从本地文件读取并用 `json.load` 解析。

    参数：
        file_path (str): 本地缓存文件的路径。
        url (str): 数据集的下载地址。

    返回：
        解析后的 Python 对象（通常是 list[dict]，每个 dict 是一条
        包含 "instruction"/"input"/"output" 字段的指令样本）。
    """
    if not os.path.exists(file_path):
        # 本地不存在缓存文件时，通过 HTTP GET 下载数据集原始文本
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 若请求失败（非 2xx），抛出异常
        text_data = response.text
        # 将下载到的文本原样写入本地文件，作为下次运行的缓存
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # 无论是刚下载的还是本地已有的，都统一从文件中读取并解析为 JSON 对象
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# The book originally used the following code below
# However, urllib uses older protocol settings that
# can cause problems for some readers using a VPN.
# The `requests` version above is more robust
# in that regard.

# 中文说明：以下是书中最初使用 urllib 实现的版本，因为 urllib 使用了
# 较旧的协议设置，部分读者在使用 VPN 时会遇到问题，因此上面改用了更
# 稳健的 `requests` 版本。此处保留原始注释掉的代码，仅作参考，不参与
# 实际运行。

# import urllib

# def download_and_load_file(file_path, url):

#     if not os.path.exists(file_path):
#         with urllib.request.urlopen(url) as response:
#             text_data = response.read().decode("utf-8")
#         with open(file_path, "w", encoding="utf-8") as file:
#             file.write(text_data)

#     else:
#         with open(file_path, "r", encoding="utf-8") as file:
#             text_data = file.read()

#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)

#     return data


def format_input(entry):
    """
    将一条原始指令数据格式化为 Alpaca 风格的提示词（prompt）文本。

    参数：
        entry (dict): 包含 "instruction" 和 "input" 键的字典，
            其中 "input" 可以为空字符串（表示该任务不需要额外输入）。

    返回：
        str: 拼接好的提示词文本，格式类似：
            "Below is an instruction ... \n\n### Instruction:\n{instruction}\n\n### Input:\n{input}"
            若 entry["input"] 为空，则不包含 "### Input:" 部分。
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 仅当该样本存在非空的 "input" 字段时，才拼接 "### Input:" 段落；
    # 否则 input_text 为空字符串，最终不会出现在提示词中
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class InstructionDataset(Dataset):
    """
    指令微调数据集，继承自 `torch.utils.data.Dataset`。

    在 `__init__` 中会预先把每条样本的完整文本
    （指令 + 输入 + 响应）一次性分词并缓存为 token id 列表，
    这样在训练迭代（多个 epoch）时无需重复调用分词器，
    以空间换时间，加快训练速度。

    注意：这里返回的是**未经过 padding 的变长 token 序列**，
    真正的动态 padding 由后续的 `custom_collate_fn` 在组 batch 时完成。
    """
    def __init__(self, data, tokenizer):
        """
        参数：
            data (list[dict]): 原始指令数据列表，每个元素需包含
                "instruction"、"input"、"output" 字段。
            tokenizer: 具备 `.encode(text) -> list[int]` 接口的分词器
                （例如 tiktoken 的 GPT-2 分词器）。
        """
        self.data = data

        # Pre-tokenize texts
        # 预先对所有样本分词，避免训练循环中重复分词带来的开销
        self.encoded_texts = []
        for entry in data:
            # 拼接 "指令(+输入)" 部分文本
            instruction_plus_input = format_input(entry)
            # 拼接 "响应" 部分文本，作为模型需要学习生成的目标内容
            response_text = f"\n\n### Response:\n{entry['output']}"
            # 完整训练文本 = 指令部分 + 响应部分
            full_text = instruction_plus_input + response_text
            # 将完整文本分词为 token id 列表，并缓存起来
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        """
        返回索引为 `index` 的样本的 token id 列表（一维 list[int]，长度可变）。
        """
        return self.encoded_texts[index]

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.data)


def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    """
    第一版（草稿）collate 函数：仅完成"动态 padding"，只返回 inputs。

    动态 padding 的思路：不同样本长度不一致，需要在组成一个 batch 时
    将它们填充（pad）到同一长度，才能堆叠（stack）成一个规整的张量。
    这里选择"当前 batch 内最长样本长度"作为 padding 目标长度
    （而不是数据集全局最大长度），因此称为"动态"padding。

    参数：
        batch (list[list[int]]): 一个 batch 内多条样本的 token id 列表，
            每条样本长度可能不同。
        pad_token_id (int): 用于填充的 token id，默认使用 GPT-2 的
            <|endoftext|> token（id 为 50256）。
        device (str): 结果张量要转移到的设备，如 "cpu" 或 "cuda"。

    返回：
        torch.Tensor: 形状为 (batch_size, batch_max_length) 的输入张量，
            其中 batch_max_length = batch 内最长样本长度（未 +1，因为
            末尾多出来的那个 padding token 在函数内部又被裁掉了）。
    """
    # Find the longest sequence in the batch
    # and increase the max length by +1, which will add one extra
    # padding token below
    # 中文：先找出 batch 中最长序列的长度，并 +1，
    # 这样后面每条样本追加一个 pad_token_id 后，最长的样本也会
    # 恰好比原长度多 1，从而保证至少留出 1 个 padding 位置
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        # 中文：给每条样本末尾追加一个 <|endoftext|> token，
        # 作为该样本结束的标记（同时也充当 padding 的种子）
        new_item += [pad_token_id]
        # Pad sequences to batch_max_length
        # 中文：将样本填充（pad）到 batch_max_length 长度，
        # 不足的部分用 pad_token_id 补齐
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        # Via padded[:-1], we remove the extra padded token
        # that has been added via the +1 setting in batch_max_length
        # (the extra padding token will be relevant in later codes)
        # 中文：通过 padded[:-1] 去掉最后一个多余的 padding token
        # （该 token 是因为前面 batch_max_length 计算时 +1 才多出来的，
        # 在后续版本的 collate 函数中，这个多出来的位置会派上用场，
        # 用来构造"整体右移一位"的 targets）
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # Convert list of inputs to tensor and transfer to target device
    # 中文：把 list[Tensor] 堆叠（stack）成一个二维张量
    # 形状为 (batch_size, batch_max_length)，并转移到目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor


def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    """
    第二版（草稿）collate 函数：在草稿1的基础上，新增 targets 的构造，
    用于"下一个 token 预测"（next-token prediction）语言模型训练目标。

    核心思路：
        inputs  = padded[:-1]   # 去掉最后一个 token
        targets = padded[1:]    # 去掉第一个 token（整体右移一位）
    这样 targets[i] 就是 inputs[i] 的"下一个 token"，即模型在看到
    inputs[:i+1] 后应当预测出的下一个 token 为 targets[i]。

    参数：
        batch (list[list[int]]): 一个 batch 内多条样本的 token id 列表。
        pad_token_id (int): 填充 token id，默认 50256（<|endoftext|>）。
        device (str): 结果张量要转移到的设备。

    返回：
        tuple[torch.Tensor, torch.Tensor]:
            inputs_tensor, targets_tensor，形状均为
            (batch_size, batch_max_length)，且 targets 相对 inputs
            整体右移了一位。
    """
    # Find the longest sequence in the batch
    # 中文：同草稿1，先确定 batch 内最长序列长度（+1 留出一个位置）
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        # 中文：追加一个 <|endoftext|> token 作为结束标记
        new_item += [pad_token_id]
        # Pad sequences to max_length
        # 中文：填充到 batch_max_length 长度
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        # 中文：inputs 去掉末尾一个 token；targets 去掉开头一个 token，
        # 即整体右移一位，构成"预测下一个 token"的监督信号
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs to tensor and transfer to target device
    # 中文：分别把 inputs 和 targets 的列表堆叠成张量
    # 形状均为 (batch_size, batch_max_length)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    正式版 collate 函数：在草稿2的基础上，新增两项关键能力：
      1. 损失掩码（loss masking）：将 targets 中"除第一个之外"的
         padding token 全部替换为 `ignore_index`（默认 -100），
         这样在计算交叉熵损失（`nn.CrossEntropyLoss` 默认
         `ignore_index=-100`）时，这些位置会被自动跳过，不参与
         损失计算和梯度更新，从而避免模型被大量 padding token
         "带偏"，只学习去预测无意义的填充符号。
         （注意：保留 targets 中第一个 padding token 不做屏蔽，
         是为了让模型仍然能学会在响应结束后正确生成
         <|endoftext|> 结束符。）
      2. 可选的最大长度截断：若指定了 `allowed_max_length`，
         则将 inputs 和 targets 都截断到该长度，控制显存占用。

    参数：
        batch (list[list[int]]): 一个 batch 内多条样本的 token id 列表。
        pad_token_id (int): 填充 token id，默认 50256（<|endoftext|>）。
        ignore_index (int): 用于屏蔽 targets 中多余 padding 位置的
            占位值，默认 -100（对应 PyTorch 交叉熵损失的默认忽略值）。
        allowed_max_length (int | None): 若不为 None，则将序列截断到
            该长度。
        device (str): 结果张量要转移到的设备。

    返回：
        tuple[torch.Tensor, torch.Tensor]:
            inputs_tensor, targets_tensor，形状均为
            (batch_size, batch_max_length)（若截断则为
            (batch_size, min(batch_max_length, allowed_max_length))）。
            targets 中多余的 padding 位置已被替换为 ignore_index。
    """
    # Find the longest sequence in the batch
    # 中文：确定 batch 内最长序列长度（+1 留出右移所需的位置）
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        # 中文：追加一个 <|endoftext|> token 作为结束/填充标记
        new_item += [pad_token_id]
        # Pad sequences to max_length
        # 中文：填充到 batch_max_length 长度
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        # 中文：inputs/targets 构造方式与草稿2相同：
        # targets 相对 inputs 整体右移一位，用于下一个 token 预测

        # New: Replace all but the first padding tokens in targets by ignore_index
        # 中文（关键步骤——损失掩码）：
        # 1) mask：标记 targets 中所有等于 pad_token_id 的位置（布尔张量）
        # 2) indices：取出这些位置对应的下标
        # 3) 若 padding 位置数量 > 1，则只保留第一个 padding token
        #    （即 indices[0]，代表真正的 <|endoftext|> 结束符）作为
        #    有效的监督信号，其余 indices[1:] 位置的 targets 一律替换
        #    为 ignore_index，训练时这些位置不计入损失
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        # 中文：如果指定了 allowed_max_length，则将 inputs 和 targets
        # 都截断到该长度上限，防止序列过长占用过多显存
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    # 中文：将 inputs/targets 列表分别堆叠为形状
    # (batch_size, seq_len) 的二维张量，并转移到目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def check_if_running(process_name):
    """
    检查本机是否存在名称包含 `process_name` 的进程正在运行。

    典型用途：在调用本地 Ollama 服务（`query_model`）之前，
    先检查 Ollama 进程是否已启动，避免请求失败。

    参数：
        process_name (str): 待检查的进程名称（子串匹配）。

    返回：
        bool: 若存在匹配的正在运行的进程，返回 True；否则返回 False。
    """
    running = False
    # 遍历系统当前所有进程，只获取每个进程的 "name" 信息以提升效率
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    """
    向本地运行的 Ollama 服务发送一次对话请求，返回模型生成的完整回复文本。

    该函数使用流式（stream=True）方式接收响应，Ollama 会按行返回
    JSON 格式的增量消息，这里将各行中的 "message.content" 拼接起来，
    得到完整的回复字符串。

    参数：
        prompt (str): 发送给模型的用户提问/提示词内容。
        model (str): Ollama 中使用的模型名称，默认 "llama3"。
        url (str): Ollama 服务的聊天接口地址。

    返回：
        str: 模型生成的完整回复文本。
    """
    # Create the data payload as a dictionary
    # 中文：构造发送给 Ollama /api/chat 接口的请求体
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            # 中文：以下参数是为了让模型生成结果具有确定性（可复现）
            "seed": 123,        # 固定随机种子
            "temperature": 0,   # 温度设为 0，去除采样随机性
            "num_ctx": 2048     # 上下文窗口长度
        }
    }

    # Send the POST request
    # 中文：以流式方式发送 POST 请求，逐行读取并解析 JSON 响应
    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            response_json = json.loads(line)
            if "message" in response_json:
                # 中文：将每个增量返回的内容片段拼接成完整回复
                response_data += response_json["message"]["content"]

    return response_data


def generate_model_scores(json_data, json_key, model="llama3"):
    """
    使用本地 LLM（通过 Ollama）对一批模型生成的回答进行 0~100 分打分，
    用于评估指令微调模型的输出质量。

    参数：
        json_data (list[dict]): 数据列表，每条记录需包含
            "instruction"/"input"/"output"（标准答案）以及
            `json_key` 对应的模型生成回答字段。
        json_key (str): `json_data` 中存放"待评分的模型回答"的字段名。
        model (str): 用作评分裁判（judge）的 Ollama 模型名称，默认 "llama3"。

    返回：
        list[int]: 每条样本对应的评分列表（0~100 的整数）。
            若某条评分结果无法解析为整数，则跳过该条，不计入结果列表。
    """
    scores = []
    # 中文：用 tqdm 显示评分进度条
    for entry in tqdm(json_data, desc="Scoring entries"):
        # 中文：构造评分提示词，包含"输入"、"标准答案输出"、
        # "待评分的模型回答"三部分，要求裁判模型只回复一个整数分数
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            # 中文：尝试将裁判模型返回的文本转换为整数分数
            scores.append(int(score))
        except ValueError:
            # 中文：若返回内容不是合法整数（例如模型多说了几句话），
            # 打印提示并跳过该条，不中断整体评分流程
            print(f"Could not convert score: {score}")
            continue

    return scores
