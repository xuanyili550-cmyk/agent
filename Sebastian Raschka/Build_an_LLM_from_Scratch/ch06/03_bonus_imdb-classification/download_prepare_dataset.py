# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
下载并准备 IMDB 影评数据集(下载压缩包、解压、加载为 DataFrame,
并切分为 train / validation / test 三个 CSV 文件)。

Download and prepare the IMDB movie review dataset.
"""

import os
import sys
import tarfile
import time
import requests
import pandas as pd


def reporthook(count, block_size, total_size):
    """
    下载进度回调函数,风格上模仿 urllib.request.urlretrieve 的 reporthook 参数,
    用于在终端上实时打印下载进度、速度和已耗时。

    参数:
        count: 已经下载的数据块数量(从 0 开始计数)。
        block_size: 每个数据块的字节数。
        total_size: 文件总字节数(用于计算百分比)。
    """
    global start_time
    if count == 0:
        # 第一个数据块到达时记录起始时间,作为速度/耗时计算的基准
        start_time = time.time()
    else:
        duration = time.time() - start_time
        progress_size = int(count * block_size)  # 已下载的大致字节数

        # 原代码/为什么是bug: 原代码直接用 `count * block_size * 100 / total_size`
        # 计算百分比,没有对 total_size 为 0(服务器未返回 Content-Length,或返回 0)
        # 的情况做保护,一旦触发会抛出 ZeroDivisionError,导致下载在打印进度时崩溃。
        # 这是确定性 bug(只要 total_size<=0 必然复现),现修复为:total_size 未知时
        # 百分比按 0 处理,不再做除法。
        if total_size > 0:
            percent = count * block_size * 100 / total_size
        else:
            percent = 0

        speed = int(progress_size / (1024 * duration)) if duration else 0
        sys.stdout.write(
            f"\r{int(percent)}% | {progress_size / (1024**2):.2f} MB "
            f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
        )
        sys.stdout.flush()


def download_and_extract_dataset(dataset_url, target_file, directory):
    """
    下载 IMDB 数据集压缩包(.tar.gz)并解压到当前工作目录。

    参数:
        dataset_url: 数据集下载地址。
        target_file: 下载后保存的本地压缩包文件名。
        directory: 解压后期望得到的数据集目录名;若该目录已存在则跳过下载,
                   避免重复下载解压。
    """
    if not os.path.exists(directory):
        if os.path.exists(target_file):
            # 目标压缩包若已存在(比如上次下载中断留下的残余文件),先删除避免冲突
            os.remove(target_file)

        response = requests.get(dataset_url, stream=True, timeout=60)
        response.raise_for_status()  # 请求失败(如 404/超时)时直接抛出异常

        # 原代码/为什么是bug: 原代码定义了 reporthook 用于展示下载进度,
        # 但下载逻辑改用 requests 流式下载后,从未调用过 reporthook,
        # 导致该函数成为死代码,下载大文件(约 80MB)时终端没有任何进度提示,
        # 看起来像卡死。这是确定性 bug(每次运行都必然不显示进度)。
        # 现从响应头读取文件总大小,并在每个数据块写入后调用 reporthook 显示进度。
        block_size = 8192
        total_size = int(response.headers.get("content-length", 0))  # 可能为 0(未知长度)

        with open(target_file, "wb") as f:
            for count, chunk in enumerate(response.iter_content(chunk_size=block_size)):
                if chunk:
                    f.write(chunk)
                    reporthook(count, block_size, total_size)  # 打印/刷新下载进度

        print("\nExtracting dataset ...")
        with tarfile.open(target_file, "r:gz") as tar:
            tar.extractall()  # 解压到当前工作目录
    else:
        print(f"Directory `{directory}` already exists. Skipping download.")


def load_dataset_to_dataframe(basepath="aclImdb", labels={"pos": 1, "neg": 0}):
    """
    遍历解压后的 aclImdb 目录(train/test 各自的 pos/neg 子目录),
    把每条影评文本连同标签读入一个 pandas DataFrame,并整体打乱顺序。

    参数:
        basepath: 数据集解压后的根目录名。
        labels: 情感标签映射,pos(正面)映射为 1,neg(负面)映射为 0。

    返回:
        包含 "text"(评论文本)和 "label"(0/1 标签)两列、已打乱顺序的 DataFrame。
    """
    data_frames = []  # List to store each chunk of DataFrame  # 用于收集每个文件对应的小 DataFrame
    for subset in ("test", "train"):
        for label in ("pos", "neg"):
            path = os.path.join(basepath, subset, label)
            for file in sorted(os.listdir(path)):  # 排序保证遍历顺序确定,便于复现
                with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                    # Create a DataFrame for each file and add it to the list
                    # 为每个文件创建一个单行 DataFrame,并加入列表(后续统一合并)
                    data_frames.append(pd.DataFrame({"text": [infile.read()], "label": [labels[label]]}))
    # Concatenate all DataFrame chunks together
    # 把所有单行 DataFrame 一次性拼接成完整数据集
    df = pd.concat(data_frames, ignore_index=True)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # Shuffle the DataFrame  # 固定随机种子打乱顺序,保证可复现
    return df


def partition_and_save(df, sizes=(35000, 5000, 10000)):
    """
    将整体 DataFrame 打乱后按给定数量切分为 train / validation / test 三份,
    并分别保存为 train.csv、validation.csv、test.csv。

    参数:
        df: 包含全部样本的 DataFrame(通常来自 load_dataset_to_dataframe)。
        sizes: 三元组 (训练集大小, 验证集大小, 测试集大小)。默认三者之和为
               50000,恰好等于 IMDB 数据集的总样本数(train+test 各 25000 条)。
    """
    # Shuffle the DataFrame
    # 再次打乱顺序(独立于 load_dataset_to_dataframe 中的打乱,保证切分随机性)
    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Get indices for where to split the data
    # 计算切分边界索引
    train_end = sizes[0]
    val_end = sizes[0] + sizes[1]

    # Split the DataFrame
    # 按边界切分为训练集/验证集/测试集
    train = df_shuffled.iloc[:train_end]
    val = df_shuffled.iloc[train_end:val_end]
    test = df_shuffled.iloc[val_end:]

    # Save to CSV files
    # 分别保存为 CSV 文件,供后续训练脚本读取
    train.to_csv("train.csv", index=False)
    val.to_csv("validation.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    print("Downloading dataset ...")
    download_and_extract_dataset(dataset_url, "aclImdb_v1.tar.gz", "aclImdb")
    print("Creating data frames ...")
    df = load_dataset_to_dataframe()
    print("Partitioning and saving data frames ...")
    partition_and_save(df)
