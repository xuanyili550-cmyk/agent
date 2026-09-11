"""
本文件用途说明（中文模块级说明）：

这是《从零构建大语言模型》（Build a Large Language Model From Scratch）一书
第 7 章（ch07，指令微调 / Instruction Fine-Tuning）配套代码中的内部单元测试文件。

它并不测试某个具体函数的返回值，而是采用「冒烟测试」（smoke test）的方式：
以子进程（subprocess）的形式，用 `--test_mode` 参数完整运行一遍
gpt_instruction_finetuning.py 这个指令微调训练脚本，只要脚本能够从头到尾
跑通且不抛出异常、返回码为 0，就认为该章节的示例代码在当前环境下是可用的。

这种测试方式常用于教学/示例代码仓库：目的是确保书中给出的完整训练流程
（数据加载 -> 模型构建/加载预训练权重 -> 指令微调 -> 生成/评估）在
CI（持续集成）环境中不会因为代码改动而跑挂，而不是对内部实现做细粒度断言。

该文件仅供内部维护者进行测试使用（见下方英文原注释 "File for internal use"），
不属于教材正文讲解的核心代码。
"""

# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)
# 内部使用文件（单元测试）


import subprocess  # 用于以子进程方式调用外部 Python 脚本，从而对整个训练脚本做端到端（end-to-end）测试


def test_gpt_class_finetune():
    """
    端到端冒烟测试：验证指令微调脚本 gpt_instruction_finetuning.py 能否正常跑通。

    测试思路：
    1. 构造命令行调用，附带 `--test_mode` 参数（脚本内部会据此使用极小规模的
       数据/模型配置，以便快速运行，避免在测试环境中进行完整的大模型训练）。
    2. 用 subprocess.run 启动一个独立的 Python 进程执行该脚本，
       捕获其标准输出/标准错误，并等待其执行完毕。
    3. 断言脚本的返回码（returncode）为 0，即进程正常退出、没有抛出未捕获的异常。
       如果失败，则在断言信息中附带脚本的 stderr 内容，方便排查问题。

    参数：无（pytest 风格的测试函数，不接受参数）。
    返回值：无返回值；测试失败时通过 assert 抛出 AssertionError。
    """
    # 组装要执行的命令：以 python 运行指令微调脚本，并传入 --test_mode 参数
    # 使其使用测试专用的小规模配置（而非完整训练配置），从而加快测试速度
    command = ["python", "ch07/01_main-chapter-code/gpt_instruction_finetuning.py", "--test_mode"]

    # 在子进程中执行该命令：
    # - capture_output=True 表示捕获子进程的 stdout/stderr
    # - text=True 表示以文本（字符串）而非字节形式获取输出内容
    result = subprocess.run(command, capture_output=True, text=True)
    # 核心断言：只要脚本退出码不是 0，就说明训练脚本运行过程中出现了异常，
    # 测试失败并打印脚本的 stderr 内容，便于定位问题原因
    assert result.returncode == 0, f"Script exited with errors: {result.stderr}"