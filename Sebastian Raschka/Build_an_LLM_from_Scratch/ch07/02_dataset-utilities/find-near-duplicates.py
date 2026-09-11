
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""查找（并可选删除）指令数据集中的「近重复」样本。

配套《从零构建大语言模型》第7章的数据集工具脚本。做指令微调前，数据里常有语义几乎
相同、只是措辞略有差异的样本；重复样本会浪费训练算力、并可能让模型对某些表述过拟合。
本脚本用「字符级 n-gram TF-IDF 向量 + 余弦相似度」来度量任意两条文本的相似度，
相似度超过阈值即判定为近重复。支持对 instruction / input / output 各字段分别检测，
并可只按 input/output 字段删除重复（instruction 字段只报告不删除）。
用法示例：python find-near-duplicates.py --json_file data.json --threshold 0.9
"""

import argparse
import json
import re
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Sample JSON dataset
example_data = [
    {"instruction": "What is the capital of Italy?",
     "input": "", "output": "The capital of Italy is Rome."
     },
    {"instruction": "What's the capital city of Italy?",
     "input": "", "output": "The capital city is Rome."
     },
    {"instruction": "Identify the main verb in the sentence: 'The cat sleeps on the couch.'",
     "input": "", "output": "The verb is 'sleeps'."
     },
    {"instruction": "Identify the verb in the following sentence: The cat sleeps on the couch.",
     "input": "", "output": "The verb in the sentence is \"sleeps.\""
     },
    # ...
]


def preprocess_text(text):
    """文本预处理：转小写并去除标点，只保留单词字符和空白，减少无关差异对相似度的干扰。"""
    # Lowercase the text
    text = text.lower()  # 统一转小写，避免大小写造成的“伪差异”
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)  # 用正则去掉所有非「单词字符/空白」的字符（即标点）
    return text


def find_near_duplicates(json_data, threshold=0.75, key="instruction"):
    """The higher the threshold, the more similar the texts have to be to match

    中文说明：在数据集的某个字段（key）上查找近重复样本。
    参数：
        json_data —— 样本列表，每个元素是含 instruction/input/output 等键的字典；
        threshold —— 相似度阈值（0~1），越高要求越相似才算重复；
        key       —— 在哪个字段上比较（如 "instruction"/"input"/"output"）。
    返回：(filtered_json_data, near_duplicates)
        filtered_json_data —— 删除重复项后的样本列表（仅当 key 为 input/output 时才会真正删）；
        near_duplicates    —— 找到的近重复对列表，每项为 (样本A, 样本B, 相似度)。
    """

    # Extract instructions
    # 抽取该字段的文本并逐条预处理；跳过该字段为空的样本
    text = [preprocess_text(item[key]) for item in json_data if item[key]]
    near_duplicates = []          # 收集近重复对
    indices_to_remove = set()     # 记录待删除样本的下标（用集合去重）

    if not text:
        return {}, near_duplicates  # 该字段没有任何非空文本，直接返回

    # Vectorize the text data
    # 向量化：用「字符级」1~3-gram 的 TF-IDF，对措辞/拼写的细微差异更鲁棒（比词级更适合找近似重复）
    vectorizer = TfidfVectorizer(stop_words=None, analyzer="char", ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(text)  # 形状约为 (样本数, 特征数)

    # Compute cosine similarity between each pair of entries
    # 计算两两余弦相似度，得到 (n, n) 的对称相似度矩阵
    cos_sim_matrix = cosine_similarity(tfidf_matrix)

    # Find pairs of near-duplicate instructions based on the threshold

    # 只遍历上三角（j 从 i+1 开始），避免自身比自身以及重复统计同一对
    for i in range(len(cos_sim_matrix)):
        for j in range(i+1, len(cos_sim_matrix)):
            if cos_sim_matrix[i, j] > threshold:  # 相似度超过阈值 -> 判定为近重复
                if len(json_data[i][key]) <= 1 or len(json_data[j][key]) <= 1:
                    continue  # 任一方该字段内容过短（<=1 字符）时忽略，避免误判
                near_duplicates.append((json_data[i], json_data[j], cos_sim_matrix[i, j]))
                if key in ("input", "output"):  # Don't remove duplicates based on the instruction
                    # 只在 input/output 字段上删除重复；instruction 字段只报告不删（避免误删有效指令）
                    indices_to_remove.add(j)  # Mark the second entry for removal（保留前一条、标记后一条待删）

    # Remove the near-duplicate entries
    # 按标记下标过滤掉被判为重复的样本
    filtered_json_data = [item for index, item in enumerate(json_data) if index not in indices_to_remove]

    return filtered_json_data, near_duplicates


def find_print_and_remove_near_duplicates(json_data, remove_duplicates=False, threshold=0.75):
    """
    Searches each key in the first JSON object for duplicates across a list of JSON objects.
    Prints the duplicates if found.

    中文说明：以第一个样本的所有键为准，逐个字段在整个数据集里查找近重复并打印结果。
    remove_duplicates=True 时，会按 input/output 字段实际删除重复样本并返回清洗后的数据。
    注意：删除是逐字段累积进行的——先按某字段过滤后，后续字段在已过滤的数据上继续检测。
    """
    # 遍历数据集第一条样本的每个字段（假设所有样本字段结构一致）
    for key in json_data[0].keys():

        if remove_duplicates:
            # 需要删除时：接收过滤后的数据并覆盖 json_data，供下一字段继续处理
            json_data, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        else:
            # 只报告不删除时：丢弃过滤结果（用 _ 接收），只保留近重复对用于打印
            _, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        separator = 50 * "="  # 分隔线，纯粹为了输出美观
        print(f"\n\n{separator}\nSearching '{key}' for duplicates ...\n{separator}")
        if not near_duplicates:
            print("No duplicates found")
        else:
            # 打印每一对近重复样本及其相似度分数
            for dup in near_duplicates:
                print(
                    f"Duplicate pair found with similarity {dup[2]:.2f}:\n"
                    f"1. {dup[0][key]}\n2. {dup[1][key]}\n"
                )
    return json_data


if __name__ == "__main__":
    # 直接运行时的命令行入口
    print("scikit-learn version:", sklearn_version)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--json_file",
        type=str,
        help=("Path to the dataset JSON file")
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help=("A sensitivity threshold between 0 and 1 where 1 is strictest")
    )
    parser.add_argument(
        "--remove_duplicates",
        action="store_true",
        default=False,
        help=(
            "Removes duplicates based on the 'input' or 'output' keys "
            " (but not the 'instruction') and saves the cleaned JSON file as --json_output_file"
        )
    )
    parser.add_argument(
        "--json_output_file",
        type=str,
        help=("Path to the dataset JSON file")
    )

    args = parser.parse_args()

    # 要删除重复就必须指定输出文件，否则无处保存清洗结果
    if args.remove_duplicates and not args.json_output_file:
        raise ValueError(
            "Provide an output file via --json_output_file "
            "to save the cleaned JSON data."
        )

    # 未提供输入文件时，使用文件顶部内置的 example_data 作演示
    if not args.json_file:
        json_data = example_data

    else:
        with open(args.json_file, "r") as file:
            json_data = json.load(file)  # 从 JSON 文件读入样本列表

    # 执行查找（及可选删除），打印结果
    json_data = find_print_and_remove_near_duplicates(
        json_data=json_data,
        remove_duplicates=args.remove_duplicates,
        threshold=args.threshold
    )

    # 若开启了删除，则把清洗后的数据写回输出文件（缩进 4 空格便于阅读）
    if args.remove_duplicates:
        with open(args.json_output_file, "w") as file:
            json.dump(json_data, file, indent=4)