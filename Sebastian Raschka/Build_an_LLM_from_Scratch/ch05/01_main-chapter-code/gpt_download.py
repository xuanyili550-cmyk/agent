# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
【模块说明 / 中文补充说明】

本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch) 一书
第 5 章 (ch05) 配套代码的一部分。它的作用是:

1. 从 OpenAI 官方(或备用镜像)服务器下载 GPT-2 的预训练权重文件
   (包括 checkpoint、词表 encoder.json、超参数 hparams.json、
   TensorFlow 格式的模型权重 model.ckpt.* 以及 BPE 分词规则 vocab.bpe);
2. 使用 TensorFlow 的 checkpoint 读取接口,把这些以 TensorFlow 格式存储的
   权重张量(numpy 数组)提取出来,并按照"每一层 Transformer block"的结构
   整理成一个嵌套字典 `params`,方便后续在本书自己实现的 GPT 模型
   (第 4/5 章手写的 PyTorch GPTModel)中,把这些权重"搬运"(加载)进去,
   从而不用自己训练就能直接使用官方 GPT-2 的预训练参数做推理或微调。

简单来说,这个文件不涉及任何"从零手写"的模型结构或注意力机制代码,
它只是一个"下载 + 权重格式转换"的工具脚本,是连接
"OpenAI 官方 GPT-2 checkpoint" 和 "本书手写 PyTorch GPT 模型" 之间的桥梁。
"""


import os

import requests
import json
import numpy as np
import transformers as tf
# 注意:这里的别名 `tf` 容易让人误以为导入的是 TensorFlow,
# 但实际导入的是 `transformers` 库(HuggingFace)。不过下面代码中用到的
# `tf.train.latest_checkpoint` / `tf.train.list_variables` / `tf.train.load_variable`
# 都是 TensorFlow 提供的 checkpoint 读取 API。这是原始代码本身的写法(依赖
# `transformers` 库间接携带/暴露了 tensorflow 的 train 模块,或环境中另有 tensorflow),
# 我们这里只做注释说明,不改动任何代码。
from tqdm import tqdm  # tqdm 用于在下载大文件时显示进度条,提升用户体验


def download_and_load_gpt2(model_size, models_dir):
    """
    下载指定大小的 GPT-2 预训练权重,并将其加载为 Python 可用的格式。

    这是本文件对外暴露的主入口函数,典型调用方式为:
        settings, params = download_and_load_gpt2("124M", "gpt2")

    参数:
        model_size (str): GPT-2 模型规模标识,必须是
            "124M"(约1.24亿参数, 12层)、"355M"(24层)、
            "774M"(36层) 或 "1558M"(48层, 即 GPT-2 XL) 之一。
            数字越大表示参数量越多、模型越深/越宽,效果通常更好但推理更慢。
        models_dir (str): 用于存放下载文件的本地目录(会在其下
            按 model_size 再建一个子目录,例如 "gpt2/124M")。

    返回:
        settings (dict): 从 hparams.json 中读到的模型超参数,
            例如 n_layer(Transformer block 层数)、n_head(注意力头数)、
            n_ctx(上下文长度,即 max sequence length)、n_embd(词嵌入维度)等。
            这些超参数将用于构建与官方 GPT-2 结构完全一致的 PyTorch 模型。
        params (dict): 从 TensorFlow checkpoint 中提取出来的所有权重,
            以嵌套字典的形式组织,结构大致为:
                {
                    "wte": (vocab_size, n_embd)   # token embedding 矩阵
                    "wpe": (n_ctx, n_embd)         # position embedding 矩阵
                    "blocks": [ {每一层 Transformer block 的权重字典}, ... ]
                    ...
                }
            具体的键名/嵌套结构由权重文件中变量名的路径(以"/"分隔)决定,
            详见 load_gpt2_params_from_tf_ckpt 函数的注释。
    """
    # Validate model size
    # 校验传入的 model_size 是否是 GPT-2 官方发布的四种规模之一,
    # 避免用户传入拼写错误的字符串导致后面下载 URL 拼接出错却报错信息不明确。
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    # 本地保存目录,例如 models_dir="gpt2", model_size="124M" -> "gpt2/124M"
    model_dir = os.path.join(models_dir, model_size)
    # OpenAI 官方托管 GPT-2 权重的地址(可能因网络原因在国内不稳定或失效)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    # 备用镜像地址:当官方地址下载失败时,自动尝试从这里下载,提高下载成功率
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    # GPT-2 官方 checkpoint 目录下固定包含的 7 个文件:
    #   - checkpoint: TensorFlow checkpoint 的元信息文件,记录最新 checkpoint 路径
    #   - encoder.json / vocab.bpe: BPE 分词器所需的词表与合并规则
    #   - hparams.json: 模型结构超参数(层数、头数、embedding 维度等)
    #   - model.ckpt.data-00000-of-00001: 实际存储权重数值的二进制文件(体积最大)
    #   - model.ckpt.index: 权重文件的索引,记录每个变量在 data 文件中的位置
    #   - model.ckpt.meta: 记录计算图结构等元信息
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    # 若目录不存在则创建;exist_ok=True 表示目录已存在时不报错(方便重复运行脚本)
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        # 依次拼出"主地址"和"备用地址"的完整下载 URL,以及本地保存路径
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        # 逐个文件下载(内部会自动处理"已存在且大小一致则跳过"以及主/备地址切换)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    # 找到 model_dir 下最新的 TensorFlow checkpoint 前缀路径
    # (对应 model.ckpt.index / model.ckpt.data-* / model.ckpt.meta 这一组文件)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    # 读取超参数 JSON 文件,得到模型结构配置(如 n_layer、n_head、n_embd 等)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    # 从 TensorFlow checkpoint 中把所有权重张量读出来,整理成嵌套字典结构,
    # 方便后续按名字取出对应权重赋值给手写的 PyTorch GPT 模型的各层参数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024  # 1 KB
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Alternative way using `requests`
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


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params