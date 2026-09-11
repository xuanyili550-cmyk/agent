# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# File for internal use (unit tests)

"""
模块中文说明
============
本文件是第 4 章（ch04）配套代码的**单元测试文件**，用于 pytest 自动化测试，
目的是确保 `gpt.py` 中实现的 GPT 模型（含多头注意力、Transformer Block、
LayerNorm、GELU 前馈网络等第 4 章讲解的核心组件）在给定固定随机种子的情况下，
其前向计算与文本生成流程的输出与「预期基准输出」逐字符一致。

在《从零构建大语言模型》一书中的角色：
- 第 4 章讲解了如何从零搭建一个 GPT 风格的解码器（decoder-only）Transformer 架构，
  并在 `gpt.py` 的 `main()` 函数中演示了一个完整的最小可运行示例：
  1) 用 tiktoken 的 GPT-2 分词器将输入文本编码为 token id；
  2) 构建 124M 参数规模的 GPTModel（未训练，随机初始化权重，但固定随机种子）；
  3) 调用 `generate_text_simple`（贪心解码，逐 token 自回归生成）生成新的 token；
  4) 将生成的 token id 解码回文本并打印。
- 本测试文件通过捕获 `main()` 的标准输出（`capsys`），并与本文件中硬编码的
  `expected`（预期输出字符串）做比较，来验证代码改动没有破坏第 4 章讲解的
  模型结构或生成逻辑（即“回归测试”，regression test）。这是一种常见的
  教学代码质量保障手段：只要模型结构、参数初始化方式、随机种子、生成算法
  保持不变，输出就应当是完全确定、可复现的。
"""

from gpt import main  # 从 gpt.py 导入 main 函数：内部会构建 GPTModel 并执行一次文本生成演示

# 预期的标准输出（golden output）：
# 字符串内容与 gpt.py::main() 在固定随机种子（torch.manual_seed(123)）、
# 固定输入文本 "Hello, I am"、固定生成参数（max_new_tokens=10）下打印到控制台的内容
# 逐字符一致。其中：
#   - "IN" 部分展示了分词结果：输入文本 -> token id 列表 -> 张量形状 (1, 4)
#     （批大小 batch=1，序列长度 seq_len=4，即原始 4 个 token）；
#   - "OUT" 部分展示了生成结果：原始 4 个 token 之后追加了 10 个新生成的 token，
#     所以最终序列长度为 4 + 10 = 14（对应 "Output length: 14"）；
#   - "Output text" 是把最终的 14 个 token id 用 GPT-2 分词器解码回的文本。
#     由于模型是随机初始化、未经训练的，这里生成的文本是没有语义的乱码，
#     但只要随机种子和模型结构不变，其内容是完全确定的，因此可以作为回归测试基准。
expected = """
==================================================
                      IN
==================================================

Input text: Hello, I am
Encoded input text: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])


==================================================
                      OUT
==================================================

Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267,
         49706, 43231, 47062, 34657]])
Output length: 14
Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous
"""


def test_main(capsys):
    """
    pytest 测试函数：验证 gpt.py::main() 的控制台输出与预期基准输出一致。

    参数：
        capsys: pytest 内置 fixture（capture system output 的缩写），用于捕获
                被测代码在标准输出（stdout）/标准错误（stderr）中打印的内容，
                而不是让它们真的打印到终端。

    测试逻辑：
        1) 调用 main()，其内部会构建 GPTModel、执行一次前向推理与文本生成，
           并通过多次 print(...) 把结果输出到 stdout；
        2) 用 capsys.readouterr() 取回刚才被捕获的 stdout 内容（captured.out）；
        3) 对预期字符串 expected 和实际捕获的 captured.out 分别做「按行去除
           行尾空白字符」的归一化处理，避免因为不同操作系统/环境下的行尾空格、
           换行符差异导致误报测试失败；
        4) 用 assert 比较两者是否完全一致；只要模型结构、随机种子、生成算法
           没有被意外改动，该断言就应当恒为真（回归测试的核心思想）。

    返回值：
        无返回值。测试通过则函数正常结束；测试失败则 assert 语句抛出
        AssertionError，pytest 会将其报告为该用例失败。
    """
    main()  # 执行 gpt.py 中的演示流程：构建模型 -> 编码输入 -> 生成 -> 解码 -> 打印
    captured = capsys.readouterr()  # 取回 main() 打印到 stdout 的全部文本内容

    # Normalize line endings and strip trailing whitespace from each line
    # 中文：按行归一化——去掉每一行末尾的空白字符（包括换行符差异），
    # 这样比较时不会因为无关紧要的尾部空格/回车差异而导致断言失败
    normalized_expected = "\n".join(line.rstrip() for line in expected.splitlines())
    normalized_output = "\n".join(line.rstrip() for line in captured.out.splitlines())

    # Compare normalized strings
    # 中文：逐字符比较归一化后的实际输出与预期输出，完全相等才算测试通过
    assert normalized_output == normalized_expected