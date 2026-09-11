# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
【中文说明】本文件用途与在《从零构建大语言模型》一书中的角色
================================================================
这是一个“工具脚本”，用来从 OpenAI 官方（或书作者提供的备份地址）下载 GPT-2 的
官方预训练权重（TensorFlow checkpoint 格式），并把这些权重解析、加载成一个
Python 嵌套字典（numpy 数组的形式），供本书后续章节（尤其是附录 E 以及第 5~7 章
关于“加载预训练权重 / 微调”的内容）中的 PyTorch GPT 模型直接使用。

它本身不包含任何 Transformer / 注意力机制的实现，而是纯粹的“数据搬运工”：
1. download_and_load_gpt2()  —— 对外的主入口：校验模型规格、下载所需文件、
   读取超参数（hparams.json）、解析 TensorFlow checkpoint，返回
   (settings, params) 两个对象。
2. download_file()           —— 通用的“带进度条 + 主备 URL 容错”的文件下载器。
3. load_gpt2_params_from_tf_ckpt() —— 把 TensorFlow checkpoint 里扁平的
   变量名（如 "model/h0/attn/c_attn/w"）解析还原成层级化的字典结构，方便后续
   代码按 "blocks"[层号]["attn"]["c_attn"]["w"] 这样的路径去取权重，再拷贝进
   自己用 PyTorch 写的 GPT 模型参数里。

读者在学习本书时，通常的使用方式是：
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
拿到 settings（模型超参数，如层数 n_layer、头数 n_head、embedding 维度 n_embd 等）
和 params（真实的权重数值），再用书中其他代码把这些权重赋值给自己实现的 GPT 模型。
"""


import os
import json
import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    """
    下载指定规格的 GPT-2 官方预训练权重，并将其加载为可用的 Python 对象。

    这是本文件对外的主入口函数，整体流程分三步：
      1. 校验 model_size 是否是 OpenAI 公开发布的四种规格之一；
      2. 依次下载该规格模型所需的 7 个文件（TensorFlow checkpoint 相关文件、
         超参数文件、BPE 分词器相关文件），支持主 URL 失败后自动切换备用 URL；
      3. 读取超参数 JSON，并解析 TensorFlow checkpoint 里的权重张量，
         组织成嵌套字典返回。

    参数:
        model_size (str): 模型规格，必须是 "124M"、"355M"、"774M"、"1558M" 之一，
            数字代表模型的参数量（约 1.24 亿、3.55 亿、7.74 亿、15.58 亿）。
        models_dir (str): 本地保存下载文件的根目录，实际文件会存放在
            os.path.join(models_dir, model_size) 目录下。

    返回:
        tuple(settings, params):
            settings (dict): 从 hparams.json 读出的模型超参数，
                典型字段包括 n_vocab（词表大小）、n_ctx（上下文长度）、
                n_embd（隐藏层/嵌入维度）、n_head（注意力头数）、n_layer（层数）。
            params (dict): 由 load_gpt2_params_from_tf_ckpt() 解析得到的权重字典，
                结构大致为:
                    {
                        "wte": (n_vocab, n_embd) 的 token embedding 矩阵,
                        "wpe": (n_ctx, n_embd) 的位置 embedding 矩阵,
                        "blocks": [ {每层的权重字典，如 attn/ mlp/ ln_1/ ln_2 等}, ... ],
                        "g", "b": 最终 LayerNorm 的缩放/偏置参数
                    }
                具体键名取决于 TensorFlow checkpoint 里原始的变量命名。
    """
    # Validate model size
    # 中文：GPT-2 官方只发布了这四种规格的权重，其余输入一律视为非法参数直接报错，
    # 避免用户传错名字后去下载一个根本不存在的远程文件路径。
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    # 中文：拼出本地保存目录，以及 OpenAI 官方托管地址 + 作者提供的备用（备份）地址。
    # 备用地址的存在是因为官方地址有时会因网络/地区问题访问失败，做了双重保险。
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]
    # 中文：这 7 个文件缺一不可——
    #   checkpoint                         : TF 用来记录“最新 checkpoint 是哪个”的元信息文件
    #   encoder.json / vocab.bpe           : BPE 分词器所需的词表与合并规则（构建 tokenizer 用）
    #   hparams.json                       : 模型的超参数（层数、头数、维度等）
    #   model.ckpt.data-00000-of-00001     : 真正存放权重数值的二进制数据文件
    #   model.ckpt.index / model.ckpt.meta : TensorFlow checkpoint 的索引/计算图元信息

    # Download files
    # 中文：确保目录存在（exist_ok=True 表示目录已存在也不报错），
    # 然后逐个文件下载，每个文件都同时准备好主 URL 和备用 URL 传给 download_file。
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    # 中文：tf.train.latest_checkpoint 会在目录里根据 "checkpoint" 元信息文件，
    # 找到最新（也是唯一）一个 checkpoint 的路径前缀（不含具体的 .data/.index 后缀）。
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    # 中文：读取超参数 JSON 文件为 Python 字典，例如 n_layer、n_head、n_embd 等，
    # 这些数值后续会决定 params["blocks"] 里应该建多少层的空字典。
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    # 中文：真正解析 TensorFlow checkpoint 中的权重张量，组织成嵌套字典结构。
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    """
    从给定 URL 下载单个文件到本地路径，并带有下载进度条与断点校验；
    若主 URL 下载失败，会自动尝试提供的备用 URL。

    参数:
        url (str): 主下载地址。
        destination (str): 保存到本地的完整文件路径。
        backup_url (str, 可选): 主地址失败时使用的备用下载地址；为 None 时不做重试。

    返回:
        None。函数通过副作用（把文件写入 destination）完成工作；
        下载失败时只打印错误信息，不抛出异常（不中断整体下载流程），
        以便一个文件下载失败不会导致其余文件也无法尝试下载。
    """
    def _attempt_download(download_url):
        """
        内部辅助函数：尝试从 download_url 下载文件到外层的 destination。

        参数:
            download_url (str): 具体要请求的下载地址（可能是主地址或备用地址）。

        返回:
            bool: 下载成功（或文件已存在且完整）返回 True；
                  下载过程中的网络异常不在这里捕获，会向上抛给调用者处理。
        """
        # 中文：stream=True 表示不一次性把整个响应体读入内存，而是边下载边写文件，
        # 这对几百 MB～几 GB 的大模型权重文件非常重要，避免内存占用过大。
        # timeout=60 防止请求卡死无限等待。
        response = requests.get(download_url, stream=True, timeout=60)
        # 中文：如果 HTTP 状态码是 4xx/5xx，这里会抛出异常，交由外层 try/except 处理。
        response.raise_for_status()

        # 中文：从响应头里读取文件总大小（字节数），用于进度条显示；
        # 如果服务器没有返回 Content-Length，则默认为 0。
        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has same size
        # 中文：简单的“断点/重复下载”优化——如果本地已经有同名文件，
        # 且文件大小与服务器声明的大小完全一致，就认为已经下载完整，直接跳过，
        # 避免重复下载几百 MB 的权重文件浪费时间和带宽。
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024  # 1 KB
        # 中文：以 1KB 为一个数据块，逐块读取并写入磁盘，
        # 配合下面的 tqdm 进度条实时展示下载进度。
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        # 中文：先尝试主 URL；如果 _attempt_download 返回 True（下载成功或文件已是最新），
        # 就直接 return，函数正常结束。
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        # 中文：捕获所有 requests 相关的网络异常（连接失败、超时、状态码错误等），
        # 说明主 URL 不可用，此时如果提供了备用 URL 就再尝试一次。
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                # 中文：备用 URL 也失败了，这里选择静默吞掉异常（pass），
                # 让代码继续往下走，统一打印下面的失败信息，而不是抛出两次异常。
                pass

        # If we reach here, both attempts have failed
        # 中文：走到这里说明主/备 URL 都尝试失败了，拼一条友好的错误提示信息，
        # 引导用户检查网络或去 GitHub Discussions 求助，而不是让程序直接崩溃退出。
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        # 中文：兜底捕获其他类型的异常（例如磁盘写入权限问题等），
        # 同样只打印不抛出，保证下载单个文件失败不会中断整个批量下载流程。
        print(f"An unexpected error occurred: {e}")


# Alternative way using `requests`
# 中文：下面这段是作者保留的“另一种写法”的参考实现（未启用，仅作为字符串常量存在，
# 不会被执行），思路与上面的 download_file 基本一致，只是没有主备 URL 容错逻辑，
# 供读者对比学习两种写法的差异。
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
    从 TensorFlow checkpoint 中读取所有权重变量，并将其解析、还原为层级化的
    嵌套字典结构，便于后续按名称路径直接取用每一层、每一个子模块的权重。

    背景说明：TensorFlow checkpoint 里的变量名是“扁平化 + 用斜杠分隔层级”的字符串，
    例如 "model/h0/attn/c_attn/w" 表示第 0 层（h0）注意力模块（attn）里
    c_attn 子模块的权重 w。本函数的核心工作就是把这种字符串路径“翻译”回
    Python 嵌套字典的结构，即 params["blocks"][0]["attn"]["c_attn"]["w"]。

    参数:
        ckpt_path (str): TensorFlow checkpoint 文件的路径前缀（不含具体后缀），
            通常由 tf.train.latest_checkpoint(model_dir) 得到。
        settings (dict): 模型超参数字典，这里主要用到 settings["n_layer"]
            （Transformer 的层数），用来预先创建对应数量的空字典占位。

    返回:
        dict: 层级化的权重字典 params，结构类似：
            {
                "wte": np.ndarray, shape 约为 (n_vocab, n_embd)  —— token embedding
                "wpe": np.ndarray, shape 约为 (n_ctx, n_embd)    —— 位置 embedding
                "g", "b": 最终 LayerNorm 的缩放/偏置，shape 约为 (n_embd,)
                "blocks": [
                    {  # 第 0 层
                        "attn": {"c_attn": {"w": ..., "b": ...}, "c_proj": {...}},
                        "mlp":  {"c_fc": {...}, "c_proj": {...}},
                        "ln_1": {"g": ..., "b": ...},
                        "ln_2": {"g": ..., "b": ...},
                    },
                    ...  # 共 settings["n_layer"] 层
                ]
            }
        具体各权重张量的形状取决于模型规格（n_embd、n_head 等），
        例如 "c_attn"/"w" 通常形如 (n_embd, 3 * n_embd)（同时产生 Q、K、V 三份投影）。
    """
    # Initialize parameters dictionary with empty blocks for each layer
    # 中文：先根据层数 n_layer 预先建好一个长度为 n_layer 的空字典列表，
    # 每个空字典之后会被逐步填充成该层 attn/mlp/ln_1/ln_2 等子模块的权重。
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    # 中文：tf.train.list_variables 会列出 checkpoint 里所有变量的 (名字, 形状) 信息，
    # 这里只用到名字（name），忽略形状（用 _ 占位）。
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        # 中文：tf.train.load_variable 按名字取出该变量的真实数值（numpy 数组），
        # np.squeeze 会把所有长度为 1 的维度去掉（例如 (1, 768) -> (768,)），
        # 因为 TensorFlow 训练时有些张量会带有多余的维度。
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        # 中文：变量名形如 "model/h0/attn/c_attn/w"，按 "/" 切分后第一段固定是
        # "model"，用 [1:] 跳过它，剩下 ["h0", "attn", "c_attn", "w"] 这样的路径片段。
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        # 中文：判断这个变量是不是属于某个 Transformer 层（名字以 "h" 开头，如 "h0"、"h11"），
        # 如果是，就把“写入目标”指向 params["blocks"][层号]这个字典；
        # 否则说明是全局参数（如 wte、wpe、最终 LayerNorm 的 g/b），直接写入 params 本身。
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        # 中文：路径中间的部分（除去层号和最后一个键名）需要逐级创建/进入嵌套字典，
        # 例如 "attn"、"c_attn" 这些中间层级；setdefault 的作用是“如果 key 不存在就创建一个
        # 空字典并返回它，如果已存在就直接返回原有的字典”，从而实现“按需自动建树”的效果。
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        # 中文：路径的最后一段（如 "w" 或 "b"）就是真正存放权重数值的键名，
        # 把上面加载好的 numpy 数组赋值进去，完成这一个变量的“落位”。
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
