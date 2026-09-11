# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


# ============================================================================
# 本文件作用：从 OpenAI 的公开 Azure Blob 存储下载官方 GPT-2 的 TensorFlow
# checkpoint（权重），并把 TF checkpoint 里的变量解析成一个嵌套的 Python
# 字典 params，供后续（04_part-1.ipynb 的 load_weights_into_gpt）搬运到我们
# 自己用 PyTorch 手写的 GPTModel 中。
# 核心三步：下载文件 -> 定位 checkpoint -> 逐变量解析成 params 字典。
# ============================================================================

import os
import urllib.request  # 用标准库发起 HTTP 下载（无需第三方 requests）

# import requests
import json
import numpy as np
# ✅ 已修复：原代码误写成 `import transformers as tf`。下方 download_and_load_gpt2 /
#   load_gpt2_params_from_tf_ckpt 里用到的 tf.train.latest_checkpoint、
#   tf.train.list_variables、tf.train.load_variable 全是 TensorFlow 的 API
#   （并非 HuggingFace transformers），现已改为正确的 import tensorflow as tf。
import tensorflow as tf
from tqdm import tqdm  # 进度条库，用于可视化下载进度


def download_and_load_gpt2(model_size, models_dir):
    # 对外主入口：给定模型规模与本地目录，下载并加载 GPT-2 权重。
    # 返回 (settings, params)：settings 是超参数字典，params 是权重字典。

    # Validate model size
    # 校验模型规模是否为 OpenAI 官方发布的四种之一，避免拼出无效下载 URL。
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    # 组装本地保存目录与远端下载基址。
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    # 一个完整的 GPT-2 checkpoint 由以下 7 个文件组成：
    #   checkpoint                     -> 文本索引文件，记录“最新 checkpoint”的名字
    #   encoder.json                   -> BPE 分词器的 token->id 词表
    #   hparams.json                   -> 模型超参数（层数、维度、头数、词表大小等）
    #   model.ckpt.data-00000-of-00001 -> 真正存放权重数值的二进制大文件
    #   model.ckpt.index               -> 变量名到 data 文件中偏移位置的索引
    #   model.ckpt.meta                -> TF 计算图的元信息
    #   vocab.bpe                      -> BPE 合并规则表
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    # 逐个下载上述文件到本地 model_dir（若已存在且大小一致则跳过）。
    os.makedirs(model_dir, exist_ok=True)  # 目录已存在也不报错
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and params
    # tf.train.latest_checkpoint 读取 "checkpoint" 索引文件，返回最新
    # checkpoint 的路径前缀（不含 .data/.index 后缀），供后续加载变量使用。
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    # 直接以 JSON 读取超参数字典。
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    # 把 TF checkpoint 里的所有变量解析成我们需要的嵌套 params 字典。
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


# 下面这段被三引号包裹的是“已弃用的旧实现”（基于第三方 requests 库），
# 作为对照保留但不会执行。当前实际使用的是其后基于 urllib 的版本。
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def download_file(url, destination):
    # 用标准库 urllib 流式下载单个文件，并带大小校验与进度条。
    # Send a GET request to download the file
    with urllib.request.urlopen(url) as response:  # 打开远端连接（上下文管理器自动关闭）
        # Get the total file size from headers, defaulting to 0 if not present
        # 从响应头 Content-Length 读取文件总字节数，用于进度条与去重校验。
        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has the same size
        # 断点/去重逻辑：若本地已有同名文件且字节数与远端完全一致，
        # 就认为已下载完成，直接返回，避免重复下载（注意：仅比大小，不校验哈希）。
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte  # 每次从网络读取 1KB，边读边写、避免占用大内存

        # Initialize the progress bar with total file size
        progress_bar_description = os.path.basename(url)  # Extract filename from URL  # 进度条标题用文件名
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:  # 以二进制写模式落盘
                # Read the file in chunks and write to destination
                # 循环按块读取，直到读到空块（下载完成）为止。
                while True:
                    chunk = response.read(block_size)
                    if not chunk:  # 读到空字节串表示流结束
                        break
                    file.write(chunk)  # 写入本地文件
                    progress_bar.update(len(chunk))  # Update progress bar  # 按实际读到的字节数推进进度条


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 把 TF checkpoint 中扁平的变量名（如 "model/h0/attn/c_attn/w"）解析成
    # 一个层次化的 params 字典，方便后续按 blocks[层号][子模块][权重名] 取用。
    # 目标结构示例：
    #   params = {
    #       "wte": ..., "wpe": ..., "g": ..., "b": ...,   # 全局权重
    #       "blocks": [ {block0 的权重}, {block1 的权重}, ... ]  # 每个 Transformer 层一个字典
    #   }

    # Initialize parameters dictionary with empty blocks for each layer
    # 预先按层数创建等量的空字典，占好 blocks 列表的每个槽位。
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    # tf.train.list_variables 列出 checkpoint 里所有变量的 (名字, 形状)。
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        # 读取该变量的实际数值张量；np.squeeze 去掉长度为 1 的多余维度
        # （例如 (1, 768) -> (768,)，方便后续形状对齐）。
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        # 变量名形如 "model/h0/attn/c_attn/w"，按 "/" 切分并去掉开头的 "model" 前缀。
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        # 判断该变量属于哪一层：以 "h" 开头（如 "h0"、"h11"）表示第 N 个 Transformer
        # 块，则定位到 params["blocks"][N]；否则是全局变量，写入 params 顶层。
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])  # 取 "h" 后面的数字作为层号
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        # 沿着中间层级（除首段层号与末段权重名外）逐级下钻，缺失则自动创建子字典。
        # 例如 attn/c_attn/w -> 依次进入 target_dict["attn"]["c_attn"]。
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        # 最后一段（如 "w" 或 "b"）作为叶子键，把权重数组挂上去。
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
