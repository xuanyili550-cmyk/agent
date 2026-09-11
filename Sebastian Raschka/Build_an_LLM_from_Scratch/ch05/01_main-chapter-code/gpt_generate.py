# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
【中文模块说明】

本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 5 章 (ch05) 的配套代码，对应主题是"在无标注文本上进行预训练"。

这个脚本的核心作用是：
1. 从 OpenAI 官方发布的 GPT-2 TensorFlow checkpoint 中下载并加载预训练权重
   （包括 124M / 355M / 774M / 1558M 四种规模）；
2. 把这些用 TensorFlow 格式保存的权重，"搬运"到我们在本书前面章节
   （见 previous_chapters.py 中的 GPTModel）用 PyTorch 手写实现的 GPT 模型上，
   使得我们自己实现的模型拥有和官方 GPT-2 完全一致的推理能力；
3. 实现一个通用的自回归文本生成函数 generate()，支持贪心解码、
   温度采样 (temperature scaling) 与 top-k 采样等常见的解码策略；
4. 提供一个命令行入口 (main)，输入一段提示词 (prompt)，
   加载 GPT-2-small (124M) 权重后生成后续文本。

对于正在学习 LLM 原理的读者，本文件是理解"预训练权重如何映射到自己实现的
Transformer 结构"以及"自回归生成的具体解码算法"的绝佳材料。
"""

import argparse
import json
import numpy as np
import os

import requests
import tensorflow as tf
import tiktoken
import torch
from tqdm import tqdm

# Import from local files
# 从同目录下的 previous_chapters.py 导入前几章已经手写实现好的 GPTModel 类
# （包含词嵌入、位置嵌入、多层 Transformer Block、最终 LayerNorm 和输出投影头）
from previous_chapters import GPTModel


def text_to_token_ids(text, tokenizer):
    """
    将一段原始文本编码为模型可以接受的 token id 张量。

    参数:
        text (str): 原始输入文本，例如提示词 "Every effort moves you"。
        tokenizer: 具备 .encode() 方法的分词器对象（这里使用 tiktoken 的 GPT-2 BPE 分词器）。

    返回:
        torch.Tensor: 形状为 (1, seq_len) 的整型张量，其中 1 是 batch 维度，
                       seq_len 是编码后 token 的数量。
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    # unsqueeze(0) 在最前面新增一个 batch 维度：(seq_len,) -> (1, seq_len)
    # 因为模型的前向传播默认期望输入形状为 (batch_size, seq_len)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    将模型生成的 token id 张量解码回可读的文本字符串。

    参数:
        token_ids (torch.Tensor): 形状为 (1, seq_len) 的 token id 张量（batch_size 固定为 1）。
        tokenizer: 具备 .decode() 方法的分词器对象。

    返回:
        str: 解码后的自然语言文本。
    """
    flat = token_ids.squeeze(0)  # remove batch dimension
    # squeeze(0) 去掉 batch 维度：(1, seq_len) -> (seq_len,)，方便转成 list 后传给 decode
    return tokenizer.decode(flat.tolist())


def download_and_load_gpt2(model_size, models_dir):
    """
    从 OpenAI 官方地址下载指定规模的 GPT-2 TensorFlow checkpoint 文件，
    并将其解析为 Python 字典形式的超参数 (settings) 与权重 (params)。

    参数:
        model_size (str): 模型规模标识，必须是 "124M"、"355M"、"774M"、"1558M" 之一。
        models_dir (str): 用于存放下载文件的本地根目录，例如 "gpt2"。

    返回:
        tuple:
            settings (dict): 从 hparams.json 中读取的模型超参数
                              （如 n_layer、n_head、n_ctx 等）。
            params (dict): 解析后的权重字典，结构为
                            {"wte": ..., "wpe": ..., "blocks": [每层一个字典], "g": ..., "b": ...}。
    """
    # Validate model size
    # 校验传入的模型规模是否是 OpenAI 官方发布过的四种规模之一，避免下载到不存在的文件
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    # 拼接出本地保存目录，以及 OpenAI 官方托管 GPT-2 checkpoint 的公共 URL 前缀
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]
    # 这些文件是 TensorFlow 1.x checkpoint 的标准组成部分：
    # - checkpoint: 记录最新 checkpoint 的元信息
    # - model.ckpt.data-00000-of-00001: 实际保存权重张量数值的数据文件
    # - model.ckpt.index / model.ckpt.meta: TensorFlow 用来索引/还原计算图和变量的辅助文件
    # - hparams.json: 模型超参数（层数、头数、embedding 维度等）
    # - encoder.json / vocab.bpe: GPT-2 使用的 BPE 分词器所需的词表和合并规则

    # Download files
    # 逐个下载上述文件到本地目录，若已存在且大小一致则跳过（见 download_file 内部逻辑）
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and params
    # 用 TensorFlow 的 API 找到最新的 checkpoint 路径，再读取超参数 JSON 文件
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    # 从 TF checkpoint 中把所有权重张量按照本书自定义的字典结构提取出来
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination):
    """
    从给定 URL 下载文件到本地路径，并在下载过程中显示进度条。
    如果本地已存在同名且同大小的文件，则跳过下载（简单的"断点续存在性检查"）。

    参数:
        url (str): 待下载文件的远程地址。
        destination (str): 保存到本地的目标路径。

    返回:
        None（副作用是在 destination 路径写入文件内容）。
    """
    # Send a GET request to download the file
    # 使用流式请求 (stream=True)，避免一次性把整个大文件（可能几百 MB 到几 GB）读入内存
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("Content-Length", 0))

    # Check if file exists and has the same size
    # 简单的缓存机制：如果本地文件已存在且字节数与远程一致，就认为已经下载完整，直接返回
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size and file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = os.path.basename(url)
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        # 以二进制方式逐块写入，边写边更新进度条，适合下载体积较大的权重文件
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))  # Update progress bar


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    从 TensorFlow checkpoint 中读取所有权重变量，并按照层级结构重新组织成
    嵌套的 Python 字典，方便后续按名字取出对应权重赋给 PyTorch 模型。

    参数:
        ckpt_path (str): TensorFlow checkpoint 的路径前缀（不含文件后缀）。
        settings (dict): 模型超参数字典，这里主要用到 settings["n_layer"] 来确定层数。

    返回:
        dict: 形如
            {
                "wte": np.ndarray,             # token embedding 权重
                "wpe": np.ndarray,             # position embedding 权重
                "g": np.ndarray, "b": np.ndarray,  # 最终 LayerNorm 的 scale/shift
                "blocks": [
                    {   # 每一层 Transformer Block 对应一个字典
                        "attn": {"c_attn": {"w": ..., "b": ...}, "c_proj": {...}},
                        "mlp":  {"c_fc": {...}, "c_proj": {...}},
                        "ln_1": {"g": ..., "b": ...},
                        "ln_2": {"g": ..., "b": ...},
                    },
                    ...
                ]
            }
    """
    # Initialize parameters dictionary with empty blocks for each layer
    # 先为每一层（共 settings["n_layer"] 层）建一个空字典占位，后面再往里面填充权重
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    # TF checkpoint 中每个变量都有一个形如 "model/h3/attn/c_attn/w" 的名字，
    # 其中 "h3" 表示第 3 层 Transformer Block，"attn/c_attn/w" 表示该层内部具体的权重矩阵
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        # np.squeeze 去掉多余的长度为 1 的维度（TF 有些变量会带一个多余的维度）
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix
        # 按 "/" 切分变量名，并跳过开头固定的 "model" 前缀，
        # 例如 "model/h3/attn/c_attn/w" -> ["h3", "attn", "c_attn", "w"]

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            # 如果变量名以 "h" 开头（如 "h0", "h1", ...），说明它属于某一层 Transformer Block
            # 提取层号后，把 target_dict 指向对应层的字典，后续的 key 都会写入这一层内部
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        # 对剩余路径中间的每一段 key（不含最后一段），逐层创建/进入嵌套字典
        # 例如 ["attn", "c_attn", "w"] 中间的 "attn" 会创建一层嵌套 {"attn": {}}
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        # 最后一段作为字典的最终 key，把权重数组真正存进去，例如 target_dict["w"] = variable_array
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def assign(left, right):
    """
    一个带形状校验的赋值辅助函数：把 right（通常是 NumPy 数组，来自 TF checkpoint）
    包装为 torch.nn.Parameter 返回，用来替换 left（PyTorch 模型中原有的参数）。

    参数:
        left (torch.nn.Parameter 或 torch.Tensor): 目标模型中原本的参数，仅用于比对形状。
        right (np.ndarray): 待赋值的新权重数据，来自 GPT-2 官方 checkpoint。

    返回:
        torch.nn.Parameter: 包装好新数值、可直接赋给模型属性的参数对象。

    说明:
        通过先比较 left.shape 与 right.shape，可以在权重"张冠李戴"（比如把
        query 权重错误地赋给了 key）时第一时间抛出异常，而不是让模型悄悄地
        加载错误的权重、之后生成出乱码文本却难以排查原因。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    """
    把从 TF checkpoint 中解析出来的 OpenAI 官方 GPT-2 权重，逐一搬运并赋值给
    我们自己用 PyTorch 手写实现的 GPTModel 实例 gpt。

    参数:
        gpt (GPTModel): 本书前几章手写实现的 GPT 模型实例（结构需要与 GPT-2 完全对齐）。
        params (dict): load_gpt2_params_from_tf_ckpt 返回的嵌套权重字典。

    返回:
        None。此函数直接原地 (in-place) 修改 gpt 内部各层的 .weight / .bias / .scale / .shift 属性。

    关键点:
        - OpenAI 官方 GPT-2 用 TensorFlow 的 Dense 层习惯，权重矩阵的形状与
          PyTorch nn.Linear 的权重形状是转置关系，因此下面大量出现 `.T`（转置）。
        - GPT-2 的注意力层把 Q、K、V 三个线性变换合并成了一个大矩阵 c_attn，
          所以这里要用 np.split 把它按最后一维切成三份，分别对应 Q/K/V。
        - GPT-2 采用"权重绑定"(weight tying)：输出层 out_head 与词嵌入 tok_emb
          共享同一份权重矩阵，因此最后 out_head.weight 直接又赋值为 wte。
    """
    # 位置嵌入 (positional embedding) 和词嵌入 (token embedding) 直接整体赋值
    # wpe: (context_length, emb_dim)；wte: (vocab_size, emb_dim)
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # ---- 注意力模块：QKV 权重 ----
        # c_attn 的权重 "w" 形状是 (emb_dim, 3 * emb_dim)，
        # 沿最后一维切成三份，分别对应 Query、Key、Value 的权重矩阵
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        # 注意 .T：TensorFlow 的 Dense 层里 y = x @ W，而 PyTorch nn.Linear 里
        # y = x @ W^T，两者权重矩阵互为转置，因此这里要转置后再赋值
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # ---- 注意力模块：QKV 偏置 ----
        # 同理，c_attn 的偏置 "b" 形状是 (3 * emb_dim,)，也按最后一维切成三份
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # ---- 注意力模块：输出投影 (out_proj / c_proj) ----
        # 多头注意力把各头拼接后的结果 (batch, seq_len, emb_dim) 再经过一次线性变换
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # ---- 前馈网络 (Feed-Forward / MLP) ----
        # layers[0] 是第一层线性变换：emb_dim -> 4*emb_dim（GPT-2 中间层通常放大 4 倍再做非线性激活）
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        # layers[2] 是第二层线性变换：4*emb_dim -> emb_dim，把维度投影回原始 embedding 维度
        # （layers[1] 是激活函数 GELU，没有可学习参数，因此索引上没有出现 layers[1]）
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # ---- 两个 LayerNorm 层 ----
        # norm1 在注意力子层之前，norm2 在前馈子层之前（Pre-LayerNorm 结构）
        # OpenAI 权重里 "g" 对应缩放系数 scale (gamma)，"b" 对应偏移 shift (beta)
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # ---- 最终的 LayerNorm 与输出投影头 ----
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 权重绑定 (weight tying)：输出层直接复用词嵌入矩阵 wte，
    # 这样可以减少参数量，也是 GPT-2 等许多语言模型的常见做法
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    以自回归 (autoregressive) 的方式，基于给定的起始 token 序列 idx，
    循环调用模型预测下一个 token，并把预测结果拼接回序列末尾，直到生成
    max_new_tokens 个新 token 或提前遇到结束符 eos_id 为止。

    参数:
        model: 训练好的（或加载了预训练权重的）GPT 模型，前向传播返回 logits。
        idx (torch.Tensor): 形状为 (batch_size, seq_len) 的初始 token id 序列。
        max_new_tokens (int): 最多生成多少个新 token。
        context_size (int): 模型支持的最大上下文长度，用于对输入做滑动截断。
        temperature (float): 温度系数。大于 0 时对 logits 做温度缩放后按概率采样；
                              等于 0（默认）时退化为贪心解码 (greedy decoding)。
        top_k (int, 可选): 若指定，则只在概率最高的 k 个 token 中采样，
                            其余全部屏蔽为 -inf，避免采样到极低概率的"离谱"词。
        eos_id (int, 可选): 结束符 (end-of-sequence) 的 token id，
                             一旦采样到该 id 就提前终止生成（仅支持 batch_size=1）。

    返回:
        torch.Tensor: 形状为 (batch_size, seq_len + 新生成的 token 数) 的完整序列，
                       包含原始输入以及新生成的所有 token。
    """

    if eos_id is not None and idx.shape[0] != 1:
        # 因为不同样本可能在不同时间步分别遇到 eos，若要支持 batch>1 的提前停止，
        # 需要为每个样本单独维护"是否已完成"的状态，这里为了简化直接不支持
        raise ValueError("EOS stopping currently supports batch size 1 only")

    # For-loop is the same as before: Get logits, and only focus on last time step
    # 逐个 token 生成：每一步都要把当前完整序列重新喂给模型做一次完整的前向传播
    # （这是最朴素的做法，没有使用 KV cache 做加速，所以每步都会有大量重复计算）
    for _ in range(max_new_tokens):
        # 由于模型只支持最长 context_size 的上下文，这里做滑动窗口截断，
        # 只保留最近的 context_size 个 token 作为本次前向传播的输入
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            # 关闭梯度计算：推理阶段不需要反向传播，可以节省显存、加快速度
            logits = model(idx_cond)  # logits 形状: (batch_size, seq_len, vocab_size)
        logits = logits[:, -1, :]
        # 自回归生成只关心"下一个 token"，所以只取序列最后一个时间步的 logits
        # 形状从 (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size)

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            # 取出概率最高的 top_k 个 logits 值（不排序索引，只要数值）
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            # min_val 是 top_k 个候选中的最小值，作为筛选阈值：
            # 所有小于该阈值的 logits 都被置为 -inf，
            # 这样后续 softmax 之后这些位置的概率会趋近于 0，相当于被排除在采样范围之外
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            # 温度缩放：logits 除以 temperature。
            # temperature < 1 时会拉大 logits 之间的差距，让分布更"尖锐"（更接近贪心）；
            # temperature > 1 时会缩小差距，让分布更"平滑"（生成结果更随机、更有多样性）
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            # 减去每一行（每个样本）的最大值，是 softmax 的常见数值稳定技巧：
            # 不会改变 softmax 结果，但可以避免 exp() 计算时数值溢出，
            # 也是为了在不同硬件后端（如 Apple 的 mps）上得到与 CPU/CUDA 一致的结果
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            # 把 logits 转换为合法的概率分布，每一行元素和为 1
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            # 按概率分布随机采样一个 token id，而不是总是选概率最大的那个，
            # 这是引入生成随机性/多样性的关键步骤
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            # 贪心解码 (greedy decoding)：直接选 logits 最大的那个 token，
            # 结果是确定性的（相同输入永远得到相同输出），但容易生成重复、乏味的文本
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            # 一旦采样/预测到结束符，就提前跳出循环，不再继续生成后续 token
            break

        # Same as before: append sampled index to the running sequence
        # 把新生成的 token 拼接到序列末尾，作为下一轮迭代的输入的一部分
        # （这就是"自回归"的含义：一步步用已生成内容预测下一步）
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def main(gpt_config, input_prompt, model_size, device):
    """
    脚本的核心业务流程：下载/加载指定规模的 GPT-2 官方权重、
    构建并初始化本书自实现的 GPTModel、将权重搬运进去，
    最后用给定的 prompt 做一次文本生成并打印结果。

    参数:
        gpt_config (dict): 传给 GPTModel 的完整超参数字典
                            (vocab_size、context_length、emb_dim、n_layers、n_heads 等)。
        input_prompt (str): 用作生成起点的提示词文本。
        model_size (str): GPT-2 权重规模标识，如 "124M"。
        device (torch.device): 推理所使用的设备 (cpu / cuda / mps)。

    返回:
        None（直接把生成的文本打印到标准输出）。
    """

    # 下载（若本地已有则复用）并解析出 GPT-2 官方超参数与权重字典
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # 用本书自实现的结构构建一个"空壳"GPT 模型（此时参数是随机初始化的）
    gpt = GPTModel(gpt_config)
    # 把从官方 checkpoint 中解析出的权重逐层搬运/赋值进这个模型
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    # 切换到 eval 模式：关闭 Dropout 等只在训练阶段生效的随机性行为，保证推理结果稳定
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)  # 固定随机种子，使得涉及采样的生成过程可复现

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=25,
        context_size=gpt_config["context_length"],
        top_k=50,
        temperature=1.0
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":

    # 命令行参数解析：允许用户自定义提示词 prompt 和推理设备 device
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Generate text with a pretrained GPT-2 model.")
    parser.add_argument(
        "--prompt",
        default="Every effort moves you",
        help="Prompt text used to seed the generation."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for running inference, e.g., cpu, cuda, mps, or auto."
    )

    args = parser.parse_args()


    torch.manual_seed(123)

    # 本示例脚本固定使用最小规模的 GPT-2（124M 参数），方便快速下载和运行演示
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = args.prompt
    DEVICE = torch.device(args.device)

    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)


    # 基础配置：与 GPT-2 官方保持一致的词表大小和上下文长度；
    # drop_rate 设为 0 是因为推理阶段不需要 dropout 正则化；
    # qkv_bias=True 是为了与 OpenAI 原始 GPT-2 的实现（Q/K/V 线性层带偏置项）保持一致
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    # 不同规模 GPT-2 模型各自的结构超参数：
    # emb_dim 是隐藏层/嵌入维度，n_layers 是 Transformer Block 的层数，n_heads 是多头注意力的头数
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # 从形如 "gpt2-small (124M)" 的字符串中提取出括号内的规模标识 "124M"
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    # 把选定模型的结构超参数合并进基础配置，得到传给 GPTModel 的完整配置字典
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    main(BASE_CONFIG, INPUT_PROMPT, model_size, DEVICE)
