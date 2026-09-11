# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script that processes the Project Gutenberg files into fewer larger files.

中文模块说明:
本脚本用于将 Project Gutenberg(古登堡计划)下载下来的大量零散小文本文件,
预处理后合并成少数几个体积较大的文件,便于后续做大语言模型的预训练
(因为大量小文件在数据加载时会带来很多 I/O 开销,合并成大文件效率更高)。
主要处理逻辑包括:
1. 判断文本是否主要为英文(过滤掉非英文语料);
2. 去除古登堡电子书文件中固定的版权/许可声明头尾(strip_headers);
3. 合并多余的空行,压缩文本体积;
4. 按照指定的单文件大小上限(max_size_mb),将多个文件的内容拼接成
   若干个 "combined_N.txt" 大文件,文件之间用特殊分隔符(如 "<|endoftext|>")隔开,
   该分隔符通常会被后续的分词器识别为文档结束标记。
"""

import argparse  # 用于解析命令行参数(如数据目录、输出目录、单文件大小上限等)
import os  # 用于文件路径拼接、目录创建、遍历文件系统等操作
import re  # 用于正则表达式匹配,这里用来压缩多余的空行
from tqdm import tqdm  # 用于显示文件处理进度条,并支持在进度条不被打断的情况下打印警告信息
from gutenberg.src.cleanup import strip_headers  # 古登堡计划配套工具库,用于去除电子书文本中的版权声明等固定头尾内容


def is_english(text, threshold=0.9):
    """
    粗略判断一段文本是否主要由英文(更准确地说是 ASCII 字符)组成。

    实现原理: 统计文本中 ASCII 字符(Unicode 码点 < 128,包含英文字母、数字、
    标点符号等)所占的比例,如果该比例超过给定阈值,则认为该文本主要是英文。
    这是一种简单但常用的启发式方法,并不做真正的语言检测,速度快、开销低,
    适合在处理海量古登堡文本时快速过滤掉非英语语料(如法语、德语等)。

    参数:
        text (str): 待检测的原始文本内容。
        threshold (float): ASCII 字符占比的阈值,默认 0.9,即要求 90% 以上
            字符为 ASCII 字符才判定为英文文本。

    返回:
        bool: 若 ASCII 字符占比大于 threshold,返回 True(判定为英文文本),
            否则返回 False。
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)  # 统计文本中 ASCII 字符(码点小于128)的数量
    return ascii_chars / len(text) > threshold  # ASCII 字符占比超过阈值即视为英文文本;注意若 text 为空会触发除零错误


def combine_files(file_paths, target_dir, max_size_mb=500, separator="<|endoftext|>", fallback_encoding="latin1"):
    """
    将多个古登堡文本文件清洗、过滤后合并为若干个大文件,写入目标目录。

    处理流程:
        1. 若目标目录不存在则创建;
        2. 依次读取每个文件内容(优先按 utf-8 编码读取,失败则回退到指定编码);
        3. 过滤掉非英文文本(调用 is_english);
        4. 调用 strip_headers 去除古登堡电子书固定的版权/许可声明等头尾内容;
        5. 用正则表达式将文本中多余的连续空行压缩为单个空行,减小体积;
        6. 累加统计当前缓存内容的字节大小,一旦超过 max_size_mb 上限,
           就将当前缓存的内容用 separator 拼接后写入一个新的 "combined_N.txt" 文件,
           并重新开始累积下一个文件;
        7. 循环结束后,将剩余未写出的内容再写入最后一个文件。

    参数:
        file_paths (list[str]): 待处理的原始文本文件路径列表。
        target_dir (str): 合并后输出文件所在的目标目录。
        max_size_mb (int): 每个合并输出文件的最大体积上限(单位: MB),默认 500。
        separator (str): 拼接不同源文件内容时使用的分隔符,默认使用
            "<|endoftext|>",这是常见的文档结束标记,便于分词器/训练脚本
            识别文档边界。
        fallback_encoding (str): 当 utf-8 解码失败时使用的回退编码,默认 "latin1"
            (即 ISO-8859-1),该编码可以映射任意单字节数据,不会再次抛出解码异常。

    返回:
        int: 最终写出的合并文件数量(即最后一个 "combined_N.txt" 的编号 N,
            也等于总共生成的文件计数)。
    """
    if not os.path.exists(target_dir):  # 若目标输出目录不存在
        os.makedirs(target_dir)  # 则递归创建该目录,确保后续写文件不会因目录缺失而报错

    current_content = []  # 当前正在累积、尚未写盘的文本内容列表(每个元素对应一个源文件清洗后的正文)
    current_size = 0  # 当前累积内容的字节大小(以 utf-8 编码字节数计),用于和 max_size_mb 比较
    file_counter = 1  # 输出文件的编号计数器,从 1 开始,决定 "combined_N.txt" 的文件名

    for file_path in tqdm(file_paths):  # 遍历所有待处理文件,并用 tqdm 显示处理进度
        try:
            with open(file_path, "r", encoding="utf-8") as file:  # 优先尝试以 utf-8 编码读取文件
                content = file.read()
        except UnicodeDecodeError:
            # Attempt to read the file with a fallback encoding
            # 若 utf-8 解码失败(说明文件实际编码可能不是 utf-8),打印警告并尝试用回退编码重新读取
            tqdm.write(f"Warning: UnicodeDecodeError encountered. Trying fallback encoding for {file_path}")  # 使用 tqdm.write 而非 print,避免打断进度条显示
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()

        if not is_english(content):  # 过滤掉非英文(ASCII 占比不足)的文本,不参与后续合并
            tqdm.write(f"Skipping {file_path} as it does not contain primarily English text.")
            continue  # 跳过当前文件,继续处理下一个
        content = strip_headers(content)  # 去除古登堡电子书固定的版权声明/许可条款等头尾模板内容,只保留正文

        # Regular expression to replace multiple blank lines with a single blank line
        # 用正则表达式将文本中"换行 + 若干空白字符 + 换行"的多重空行,统一替换为单个空行,压缩体积、规整格式
        content = re.sub(r"\n\s*\n", "\n\n", content)
        estimated_size = len(content.encode("utf-8"))  # 以 utf-8 编码估算当前文本内容占用的字节数,用于容量控制

        if current_size + estimated_size > max_size_mb * 1024 * 1024:  # 若加入当前文件后会超过单文件大小上限(MB 转换为字节)
            target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")  # 生成当前批次的输出文件路径,如 combined_1.txt
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(separator.join(current_content))  # 用分隔符将已累积的多个文件内容拼接后一次性写出
            file_counter += 1  # 输出文件编号自增,为下一批次做准备
            current_content = [content]  # 将当前超出容量的文件内容作为新一批次的起点重新开始累积
            current_size = estimated_size  # 重置累积大小为当前文件的大小
        else:
            current_content.append(content)  # 未超出容量上限,则将当前文件内容加入累积列表
            current_size += estimated_size  # 累加当前批次的总字节大小

    if current_content:  # 循环结束后,若仍有未写出的剩余内容(最后一批次不足 max_size_mb 也需要落盘)
        target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")  # 生成最后一个输出文件的路径
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            target_file.write(separator.join(current_content))  # 写出剩余内容
    return file_counter  # 返回最终生成的合并文件总数(编号计数器的最终值)


if __name__ == "__main__":  # 仅当该脚本被直接运行(而非作为模块导入)时,才执行以下命令行入口逻辑

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Preprocess and combine text files for pretraining")  # 创建命令行参数解析器;ArgumentDefaultsHelpFormatter 会在 --help 中自动显示各参数的默认值

    parser.add_argument("--data_dir", type=str, default="gutenberg/data/raw",
                        help="Directory containing the downloaded raw training data")  # 原始古登堡语料所在目录(未处理前的小文件)
    parser.add_argument("--max_size_mb", type=int, default=500,
                        help="The maximum file size for each concatenated file in megabytes")  # 合并后每个输出文件的最大体积(MB)
    parser.add_argument("--output_dir", type=str, default="gutenberg_preprocessed",
                        help="Directory where the preprocessed data will be saved")  # 预处理完成后合并文件的保存目录

    args = parser.parse_args()  # 解析命令行传入的实际参数值

    all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.data_dir)
                 for name in files if name.endswith((".txt", ".txt.utf8"))]  # 递归遍历 data_dir 目录树,收集所有以 .txt 或 .txt.utf8 结尾的文件的完整路径

    print(f"{len(all_files)} file(s) to process.")  # 打印待处理文件总数,便于用户了解本次任务规模
    file_counter = combine_files(all_files, args.output_dir, max_size_mb=args.max_size_mb)  # 调用核心函数执行清洗与合并,得到最终生成的文件数量
    print(f"{file_counter} file(s) saved in {os.path.abspath(args.output_dir)}")  # 打印结果摘要,显示生成了多少文件以及输出目录的绝对路径
