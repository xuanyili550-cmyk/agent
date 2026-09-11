"""加载 Qwen3 基座模型、文本生成与 KV 缓存（配套《从零构建推理模型》reasoning_from_scratch）。

本脚本演示三件事：
1. 检测当前可用的计算设备（CUDA / Intel XPU / Apple MPS / CPU）；
2. 下载并加载 Qwen3 小模型的分词器（先只下 tokenizer，再下完整权重）；
3. 对一句 prompt 做编码/解码，直观查看 token 与文本的对应关系。

依赖：需先安装配套包  ->  `pip install reasoning_from_scratch`
（否则第 12 行起会报 ModuleNotFoundError: No module named 'reasoning_from_scratch'）
"""

import torch
print(f"pyTorch{torch.__version__}")  # 打印 PyTorch 版本

# 依次检测可用的加速设备，命中一个就打印其信息。
# 【bug 修复】原代码判断顺序是 cuda -> xpu -> cpu -> mps。问题在于 torch.cpu.is_available()
# 恒为 True（CPU 总是“可用”的），所以只要没有 CUDA/XPU，就会先命中 cpu 分支，
# 导致下面的 mps 分支永远不可达——在 Apple Silicon 上本该识别为“Apple GPU(MPS)”却被误报成 CPU。
# 正确做法：把 mps 放在 cpu 之前，让 cpu 作为最后的兜底分支。此处已调整顺序。
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())        # NVIDIA GPU
elif torch.xpu.is_available():
    print(torch.xpu.get_device_name())         # Intel GPU (XPU)
elif torch.mps.is_available():
    print("Apple GPU")                          # Apple Silicon 的 Metal (MPS) 后端
elif torch.cpu.is_available():
    print(torch.cpu.get_capabilities())        # 兜底：CPU（打印指令集/核心数等能力信息）

# 从配套包导入下载函数，先“只下载分词器”(tokenizer_only=True)，存到 qwen3/ 目录，
# 这样不必等大权重下载完就能先跑分词相关的代码
from reasoning_from_scratch.qwen3 import download_qwen3_small
download_qwen3_small(kind="base", tokenizer_only=True, out_dir="qwen3")

from pathlib import Path
from reasoning_from_scratch.qwen3 import Qwen3Tokenizer
tokenizer_path = Path("qwen3") / "tokenizer-base.json"   # 刚下载的分词器文件路径
tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)  # 构造 Qwen3 分词器


prompt = "explain large language models."
input_token_ids_list = tokenizer.encode(prompt)  # 把文本编码成 token id 列表
input_token_ids_list  # 注意：这是脚本(.py)不是 notebook，单独一行表达式不会自动打印；如需查看请用 print(input_token_ids_list)

# 逐个 token 打印“id -> 对应文本”，直观看清 BPE 是怎么把句子切成子词的
# 注：这里对单个 token id 调用 decode(i)。若你的 Qwen3Tokenizer 版本要求传“列表”，
# 此行会报错，届时改成 tokenizer.decode([i]) 即可。
for i in input_token_ids_list:
    print(f"{i}->{tokenizer.decode(i)}")
tokenizer.decode(input_token_ids_list)  # 把整串 token id 解码回文本（同样，脚本里不会自动打印，需要看结果请套上 print(...)）

# 最后再下载完整的模型权重(tokenizer_only=False)，为后续加载模型做文本生成/KV 缓存做准备
download_qwen3_small(kind="base", tokenizer_only=False, out_dir="qwen3")
