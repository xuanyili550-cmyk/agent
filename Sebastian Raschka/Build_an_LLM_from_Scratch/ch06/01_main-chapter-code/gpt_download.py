# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
中文模块说明:

本文件用于从 OpenAI 官方（或备用镜像）下载预训练的 GPT-2 权重文件，
并把 TensorFlow 格式的 checkpoint 解析成一个便于在 PyTorch 中加载的
嵌套字典（`params`），以及模型的超参数字典（`settings`）。

在《从零构建大语言模型》（Build a Large Language Model From Scratch）一书中，
第 6 章（分类微调）会用到本文件：先下载官方发布的 GPT-2 预训练权重，
再把这些权重加载进我们自己手写实现的 GPT 模型结构中，从而在真实预训练权重
的基础上进行微调，而不需要从零开始预训练模型。

核心流程：
1. `download_and_load_gpt2`：对外的主入口函数，根据传入的模型规格
   （如 "124M"）下载所有必需的 checkpoint 相关文件，并解析出
   `settings`（超参数，如层数、隐藏维度等）与 `params`（权重张量字典）。
2. `download_file`：负责单个文件的下载，支持断点判断（文件已存在且大小一致则跳过）、
   进度条展示，以及主 URL 失败时自动切换到备用 URL。
3. `load_gpt2_params_from_tf_ckpt`：读取 TensorFlow 格式的 checkpoint 文件，
   将其中的变量按照变量名（如 "model/h0/attn/c_attn/w"）解析并重组为
   嵌套字典结构，方便后续按层、按模块取出对应的权重矩阵。
"""


import os
import json
import numpy as np
import requests
import transformers as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    """
    下载指定规格的 GPT-2 预训练权重，并加载为可用的超参数与权重字典。

    参数:
        model_size (str): 模型规格，只能是 "124M"、"355M"、"774M"、"1558M" 之一，
            对应 OpenAI 发布的四种不同大小的 GPT-2 模型。
        models_dir (str): 本地用于保存下载文件的目录（会在其下按 model_size
            再建一个子目录）。

    返回:
        tuple:
            settings (dict): 从 hparams.json 中读取的模型超参数（如层数、
                注意力头数、embedding 维度等）。
            params (dict): 由 `load_gpt2_params_from_tf_ckpt` 解析出的权重
                字典，包含每一层（block）的权重张量。
    """
    # Validate model size
    # 校验传入的模型规格是否合法，避免下载到不存在的路径
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    # 本地保存目录，以及官方主下载源与备用下载源（防止官方源不可用/被墙）
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    # GPT-2 checkpoint 所需的全部文件：
    # - checkpoint: TensorFlow checkpoint 索引文件，记录最新 checkpoint 名称
    # - encoder.json / vocab.bpe: BPE 分词器所需的词表与合并规则
    # - hparams.json: 模型超参数（层数、维度等）
    # - model.ckpt.*: TensorFlow 格式保存的实际权重数据
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    # 逐个文件下载到本地目录；download_file 内部会处理已存在文件的跳过逻辑
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    # 找到最新的 TensorFlow checkpoint 路径（本质上是读取 "checkpoint" 文件）
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    # 读取超参数 JSON 文件，得到 n_layer、n_head、n_embd 等信息
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    # 将 TensorFlow checkpoint 中的权重解析为嵌套字典结构，便于后续按名取用
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    """
    下载单个文件到指定本地路径，支持跳过已下载文件、显示下载进度条，
    以及在主 URL 下载失败时自动尝试备用 URL。

    参数:
        url (str): 主下载地址。
        destination (str): 文件保存的本地路径。
        backup_url (str, optional): 备用下载地址，当主地址下载失败时使用。

    返回:
        None。函数通过 print 输出下载状态/错误信息，不抛出未处理异常
        （网络相关异常会被内部捕获并打印提示）。
    """
    def _attempt_download(download_url):
        # 内部辅助函数：尝试从给定 URL 以流式方式下载文件
        # 使用 stream=True 避免一次性把大文件全部读入内存
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()  # 若 HTTP 状态码非 2xx，则抛出异常触发重试/报错逻辑

        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has same size
        # 如果本地已存在同名文件且大小与服务器返回的一致，则认为已是最新，跳过下载
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024  # 1 KB
        desc = os.path.basename(download_url)
        # 使用 tqdm 显示下载进度条，按块（block）读取并写入本地文件，
        # 避免一次性加载整个大文件导致内存占用过高
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        # 优先尝试主 URL 下载
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        # 主 URL 下载失败（网络错误、超时、HTTP 错误等），若提供了备用 URL 则重试
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                # 备用 URL 也失败，忽略此异常，走到下面统一的错误提示逻辑
                pass

        # If we reach here, both attempts have failed
        # 主备两个下载源均失败，打印友好的错误提示，指引用户检查网络或查看讨论区
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        # 捕获其他未预期的异常（例如文件写入失败等），避免整个下载流程崩溃
        print(f"An unexpected error occurred: {e}")


# Alternative way using `requests`
# 中文说明：以下为注释掉的另一种实现方式（未启用），逻辑与上面的 download_file 类似，
# 仅供参考对比，不会被实际调用。
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
    """
    从 TensorFlow 格式的 GPT-2 checkpoint 中读取所有权重变量，
    并按照变量名的层级结构重新组织成嵌套字典，便于后续映射到
    自定义的 PyTorch GPT 模型参数上。

    参数:
        ckpt_path (str): TensorFlow checkpoint 文件路径（不含扩展名，
            通常由 `tf.train.latest_checkpoint` 返回）。
        settings (dict): 模型超参数字典，至少需要包含 "n_layer"
            （Transformer 层数），用于预先初始化每一层的权重容器。

    返回:
        dict: 形如
            {
                "blocks": [ {层0的权重字典}, {层1的权重字典}, ... ],
                其他顶层参数（如词嵌入、位置嵌入等）...
            }
            的嵌套字典，可用于后续加载进自定义 GPT 模型。
    """
    # Initialize parameters dictionary with empty blocks for each layer
    # 先为每一层（Transformer block）建立一个空字典占位，后面按层号填充
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    # 遍历 checkpoint 中保存的所有变量名（不包含变量值本身）
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        # 加载变量的实际数值，并用 np.squeeze 去掉多余的单一维度（如 shape 中的 1）
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        # TensorFlow 中变量名形如 "model/h0/attn/c_attn/w"，
        # 去掉开头的 "model" 前缀后，剩余部分即描述了该变量在模型结构中的位置
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        # 如果变量属于某个 Transformer 层（名称以 "h" 开头，如 "h0"、"h1"），
        # 则将目标字典指向对应层号的 blocks[layer_number]；
        # 否则视为顶层参数（如词嵌入/位置嵌入），直接放入 params 顶层
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        # 根据变量名中间的各级路径（去掉首尾两段），逐层创建/获取嵌套字典，
        # 从而还原出与 TensorFlow 变量名一致的层级结构
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        # 变量名的最后一段作为最终的键，把解析出的权重数组保存进去
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
