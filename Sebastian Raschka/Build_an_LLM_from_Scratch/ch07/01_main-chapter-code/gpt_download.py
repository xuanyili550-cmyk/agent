# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
下载并加载 OpenAI 公开发布的 GPT-2 预训练权重（TensorFlow checkpoint 格式）。
Download and load the GPT-2 pretrained weights released by OpenAI
(stored as a TensorFlow checkpoint).

本模块提供三个函数：
1. download_and_load_gpt2: 对外主入口，下载指定规模的 GPT-2 模型文件，
   并解析出超参数（settings）与权重参数（params）。
2. download_file: 通用的单文件下载器，支持主/备用 URL 与断点式的
   “已存在且大小一致则跳过”逻辑，并带进度条。
3. load_gpt2_params_from_tf_ckpt: 从 TensorFlow checkpoint 中读取所有
   变量，按照 GPT-2 的命名规则（如 "model/h0/attn/c_attn/w"）重组为
   嵌套的 Python 字典，方便后续加载进自定义的 PyTorch/NumPy 模型中。
"""

import os
import json

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    """
    下载指定规模的 GPT-2 预训练权重文件，并加载为 (settings, params) 二元组。

    参数:
        model_size (str): 模型规模标识，必须是
            "124M" / "355M" / "774M" / "1558M" 之一。
        models_dir (str): 本地保存模型文件的根目录，实际文件会保存到
            models_dir/model_size 子目录下。

    返回:
        settings (dict): 从 hparams.json 加载的超参数字典
            （如 n_layer、n_head、n_embd 等）。
        params (dict): 从 TF checkpoint 解析出的权重参数字典，
            结构见 load_gpt2_params_from_tf_ckpt。

    风险/跨版本项（不改，仅标注上报）：
        - 下面用 os.path.join(base_url, model_size, filename) 拼接 URL。
          os.path.join 会使用当前操作系统的路径分隔符，在 Windows 上会
          用反斜杠 "\\" 而不是 "/"，从而拼出无效的 URL（跨平台风险）。
          目前仅在类 Unix 系统（如本机 macOS）上是正确的。
        - download_file 内部下载失败时只打印错误信息、不抛异常，若某个
          文件下载失败，本函数会继续往下执行，可能导致
          tf.train.latest_checkpoint 返回 None，进而在后面读取
          hparams.json 或解析 checkpoint 时报出不直观的错误。这是原有的
          错误处理设计，属于行为变更风险，未在本次注释中修改。
    """
    # 校验传入的模型规模是否受支持
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # 定义本地保存路径与远程下载地址
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    # 备用地址：当官方地址不可用时（例如网络受限）用于兜底下载
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    # GPT-2 官方发布的一套 checkpoint 通常包含以下 7 个文件
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载文件：逐个文件调用 download_file，内部会处理主/备 URL 切换
    os.makedirs(model_dir, exist_ok=True)  # 若目录已存在则不报错
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # 加载超参数（settings）与权重参数（params）
    # tf.train.latest_checkpoint 会在 model_dir 中查找最新的 checkpoint 前缀
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    """
    下载单个文件到本地路径，支持：
        1. 若本地文件已存在且大小与远程一致，则跳过下载；
        2. 主 URL 下载失败时，自动尝试 backup_url；
        3. 下载过程中显示 tqdm 进度条。

    参数:
        url (str): 主下载地址。
        destination (str): 本地保存的完整文件路径。
        backup_url (str, 可选): 备用下载地址，主地址失败时使用。

    返回:
        None。函数通过内部 print 报告下载结果/失败原因，不抛出异常
        （这是原有设计，见 download_and_load_gpt2 文档中的风险说明）。
    """
    def _attempt_download(download_url):
        """
        实际执行一次下载尝试（内部辅助函数，闭包引用外层的 destination）。

        原代码 / 为什么是 bug：
            原代码写的是
                response = requests.get(download_url, stream=True, timeout=60)
                ...
                return True
            即使命中“文件已存在且大小一致”分支直接 return True，
            也没有关闭这个 stream=True 的响应对象；requests 在
            stream 模式下不会自动释放底层 TCP 连接，必须显式调用
            response.close()（或用 with 语句）才会归还连接池。
            这是一个确定性的资源泄漏 bug：函数每次被调用都会新建一次
            HTTP 连接，如果命中"已存在"分支就不会被关闭，在
            download_and_load_gpt2 中会为每个模型文件都发起一次
            GET 请求，多次调用会逐渐耗尽连接池 / 文件描述符。
            修复方式：用 `with requests.get(...) as response:` 包裹，
            让 Python 在函数退出（包括 return）时自动调用
            response.close()，语义与原代码完全一致，只是修复了泄漏。
        """
        # 使用 with 语句确保无论从哪个分支返回，响应对象都会被正确关闭
        with requests.get(download_url, stream=True, timeout=60) as response:
            response.raise_for_status()  # 非 2xx 状态码会抛出 HTTPError

            # 从响应头读取文件总大小（部分服务器可能不返回该字段，此时为 0）
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查本地是否已存在同名文件，且大小与远程一致 —— 若一致则跳过下载
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size and file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True

            block_size = 1024  # 每次读取 1 KB 数据块
            desc = os.path.basename(download_url)
            # 用 tqdm 显示下载进度条，单位按字节自动换算（KB/MB/GB）
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
                with open(destination, "wb") as file:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:  # 过滤掉保活用的空 chunk
                            file.write(chunk)
                            progress_bar.update(len(chunk))
        return True

    try:
        # 先尝试主 URL
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        # 主 URL 下载失败（网络错误、超时、HTTP 错误码等），尝试备用 URL
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                # 备用 URL 也失败，静默吞掉异常，交给下面统一打印错误信息
                pass

        # 主/备 URL 均下载失败，打印提示信息（不抛出异常，见函数文档说明）
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        # 捕获其他未预期的异常（如磁盘写入失败等），避免整个下载流程中断
        print(f"An unexpected error occurred: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    从 TensorFlow checkpoint 中读取全部变量，并按照 GPT-2 的命名规则
    重组为嵌套字典结构，便于后续映射到自定义模型（如 PyTorch 实现）的
    权重上。

    参数:
        ckpt_path (str): TensorFlow checkpoint 的路径前缀
            （即 tf.train.latest_checkpoint 的返回值）。
        settings (dict): 超参数字典，必须包含 "n_layer"
            （用于预先建立对应层数的 blocks 列表）。

    返回:
        params (dict): 形如
            {
                "wte": ...,               # token embedding
                "wpe": ...,               # position embedding
                "blocks": [
                    {"attn": {"c_attn": {"w": ..., "b": ...}, ...}, ...},
                    ...  # 每层一个字典，共 settings["n_layer"] 个
                ],
                ...
            }
            的嵌套字典，键名与 checkpoint 中变量名的层级结构一一对应。

    风险/跨版本项（不改，仅标注上报）：
        - 若 ckpt_path 为 None（例如上游下载失败导致
          tf.train.latest_checkpoint 未找到任何 checkpoint），
          tf.train.list_variables(None) 会抛出异常，报错信息可能不够
          直观，但这是调用方传参错误导致，不属于本函数自身逻辑 bug。
        - variable_name_parts[0].startswith("h") 依赖 GPT-2 checkpoint
          固定的变量命名约定（层级变量以 "h<数字>" 命名），若未来
          OpenAI 更换命名规则，这里会静默地把变量错误分类到顶层
          而不是某一层，属于跨版本兼容性风险，暂不处理。
    """
    # 初始化参数字典：为每一层（共 n_layer 层）预先建立一个空字典占位
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历 checkpoint 中的每一个变量（name 是变量名字符串，如 "model/h0/attn/c_attn/w"）
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载该变量的实际数值，并用 np.squeeze 去掉多余的单一维度（如 shape (1, 768) -> (768,)）
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名，提取出有意义的层级部分（第一段固定是 "model"，跳过它）
        variable_name_parts = name.split("/")[1:]  # 跳过开头的 "model/" 前缀

        # 确定该变量应该写入到哪个目标字典：
        # 若名字以 "h" 开头（如 "h0"、"h1" ...），说明它属于某一层 Transformer block，
        # 需要定位到 params["blocks"][层号] 对应的字典；否则写入顶层 params。
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])  # 从 "h0" 中解析出层号 0
            target_dict = params["blocks"][layer_number]

        # 对中间的路径段（去掉第一段和最后一段）逐级建立/获取嵌套字典
        # 例如 "attn/c_attn/w" -> 中间段是 "attn"，需要 target_dict["attn"] = {}
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})  # 不存在则创建空字典，存在则复用

        # 最后一段路径作为字典的键，把变量值赋进去（如 "w" 或 "b"）
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
