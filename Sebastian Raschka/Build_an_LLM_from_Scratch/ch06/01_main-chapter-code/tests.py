# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

"""
模块级中文说明
==============
本文件是《从零构建大语言模型》(Build a Large Language Model From Scratch)
第 6 章「文本分类微调」配套代码的**内部单元测试文件**。

它并不属于书中正文讲解的示例代码，而是仓库作者用来做 CI（持续集成）自测的
辅助脚本：通过 `pytest` 运行本文件，可以自动验证同目录下的
`gpt_class_finetune.py`（GPT 模型微调训练脚本）在“测试模式”
（`--test_mode` 参数）下能够端到端跑通、不抛异常、正常退出。

测试模式通常会：
    - 使用一个很小的预训练权重/极少的训练轮数，跳过真实的下载与长时间训练；
    - 只是为了快速检查脚本整体流程（数据加载 -> 模型构建 -> 微调 -> 评估）
      是否存在语法错误或运行时错误。

因此本文件本身不测试模型的准确率等指标，只做「脚本能否成功运行完毕」的
冒烟测试（smoke test）。
"""

import subprocess  # 用于在子进程中调用外部 Python 脚本，并捕获其标准输出/错误和返回码


def test_gpt_class_finetune():
    """
    冒烟测试：验证 gpt_class_finetune.py 在测试模式下可以正常运行完毕。

    测试逻辑：
        1. 构造命令行，以 `--test_mode` 参数调用同目录下的
           `gpt_class_finetune.py` 脚本（该脚本负责基于 GPT 模型做
           垃圾邮件分类的微调训练）；
        2. 使用 `subprocess.run` 在子进程中执行该命令，并捕获
           stdout/stderr（`capture_output=True`），以文本形式返回
           （`text=True`）；
        3. 断言子进程的返回码 `returncode` 为 0，即脚本正常退出、
           没有抛出未捕获的异常；若返回码非 0，则在断言失败信息中
           打印子进程的 stderr，方便定位问题。

    参数：无（pytest 自动发现并调用此函数，无需手动传参）。

    返回值：无返回值；测试失败时会通过 `assert` 抛出 AssertionError，
    从而被 pytest 标记为失败用例。
    """
    # 待执行的命令：以测试模式运行微调脚本，避免完整训练/下载耗时过长
    command = ["python", "ch06/01_main-chapter-code/gpt_class_finetune.py", "--test_mode"]

    # 运行子进程，捕获输出内容（stdout/stderr）以便测试失败时打印调试信息
    result = subprocess.run(command, capture_output=True, text=True)
    # 核心断言：脚本必须成功退出（返回码为 0），否则测试失败并打印错误信息
    assert result.returncode == 0, f"Script exited with errors: {result.stderr}"
