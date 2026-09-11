# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

"""
本模块用途:针对 Gutenberg 预训练脚本(pretraining_simple.py)的内部单元测试。

测试思路:
1. 构造一个由重复片段("a b c d")拼接而成的简单文本文件,
   模拟 Gutenberg 语料的一个"书籍"文件,放到 gutenberg/data 目录下;
2. 以子进程方式调用 pretraining_simple.py,并传入 --debug true 参数,
   使脚本以调试模式(小规模、快速)运行一遍完整的预训练流程;
3. 检查脚本标准输出中是否包含 "Maximum GPU memory allocated" 字样,
   以此判断预训练脚本是否成功跑通(能正常执行到打印显存占用的阶段)。

本文件仅供仓库内部测试使用,并非面向读者的教学示例代码。
"""

from pathlib import Path
import os
import subprocess


def test_pretraining():
    # 测试目的:验证 pretraining_simple.py 在 --debug true 模式下能够
    # 完整跑通一次训练流程(包括数据加载、模型训练等),不会中途报错退出。

    sequence = "a b c d"          # 用作构造测试语料的基本重复片段
    repetitions = 1000            # 重复次数,用于生成足够长度的测试文本
    content = sequence * repetitions  # 拼接生成最终的测试文本内容

    folder_path = Path("gutenberg") / "data"  # 模拟 Gutenberg 语料存放目录
    file_name = "repeated_sequence.txt"       # 测试用的文本文件名

    os.makedirs(folder_path, exist_ok=True)  # 确保目标目录存在,不存在则创建

    # 将构造好的重复文本写入测试文件,作为待训练脚本读取的语料
    with open(folder_path/file_name, "w") as file:
        file.write(content)

    # 以子进程方式运行预训练脚本,开启 debug 模式(通常意味着更小的数据/步数,便于快速测试)
    result = subprocess.run(
        ["python", "pretraining_simple.py", "--debug", "true"],
        capture_output=True, text=True
    )
    print(result.stdout)  # 打印子进程的标准输出,便于测试失败时排查问题
    # 关键断言:只要脚本输出中出现了显存占用统计信息,
    # 就说明训练流程已经跑到了预期的收尾阶段,视为测试通过
    assert "Maximum GPU memory allocated" in result.stdout
