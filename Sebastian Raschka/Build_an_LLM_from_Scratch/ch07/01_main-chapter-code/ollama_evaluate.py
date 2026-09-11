# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# A minimal instruction finetuning file based on the code in chapter 7

"""
本模块用途说明（中文）：
这是《从零构建大语言模型》（Build a Large Language Model From Scratch）第 7 章
配套代码的一部分，用于对「指令微调（instruction finetuning）」之后的模型输出
进行自动化质量评估。

整体思路：
1. 先检查本机是否已经启动 Ollama（一个可以在本地运行开源大模型的工具/服务）。
2. 读取一份测试数据集（JSON 文件），其中每条数据包含指令（instruction）、
   输入（input）、参考答案（output）以及待评估模型生成的回答（model_response）。
3. 对每一条数据，构造一个「评分提示词（prompt）」，把参考答案和模型生成的回答
   都喂给 Ollama 本地运行的裁判模型（默认使用 llama3），让裁判模型给出一个
   0~100 的分数，用于衡量模型回答的质量。
4. 汇总所有分数，计算平均分，作为该指令微调模型整体表现的量化指标。

在书中的角色：
第 7 章讲解「指令微调」，即让预训练语言模型学会理解并执行人类指令。微调完成后
需要一种自动化、可复现的方式来评估模型输出质量，而不必逐条人工阅读。本文件
展示了一种「用另一个大模型当裁判（LLM-as-a-judge）」的评估思路：借助本地部署
的 Ollama 服务调用 llama3 模型，对指令微调模型的回答打分，从而得到一个可比较
的量化评估结果。
"""

import json
import psutil
from tqdm import tqdm  # 用于在评分循环中显示进度条，方便观察评估进度
import requests  # 用于向本地 Ollama HTTP 服务发送请求


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    """
    向本地运行的 Ollama 服务发送一次对话请求，获取模型的回复文本。

    作用：
        封装对 Ollama `/api/chat` 接口的调用，将传入的 prompt 作为用户消息发送，
        并以流式（stream）方式接收模型逐段生成的回复，最终拼接成完整字符串返回。

    参数：
        prompt (str): 发送给模型的用户提示词内容。
        model (str): 使用的 Ollama 模型名称，默认为 "llama3"。
        url (str): Ollama 服务的 API 地址，默认指向本地 11434 端口的 /api/chat。

    返回值：
        str: 模型返回的完整文本内容（拼接所有流式片段后的结果）。
    """
    # Create the data payload as a dictionary
    # 构造请求体：指定模型、对话消息列表，以及为了保证评分结果可复现而设置的参数
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            # 中文说明：以下参数是为了让模型输出「确定性」结果（同样输入得到同样输出），
            # 这样评分才具有可重复性，便于多次运行结果对比。
            "seed": 123,        # 固定随机种子
            "temperature": 0,   # 温度设为 0，尽量减少生成的随机性
            "num_ctx": 2048     # 上下文窗口长度
        }
    }

    # Send the POST request
    # 使用 stream=True 以流式方式接收响应，因为 Ollama 会按行（每行一个 JSON）逐步返回生成内容
    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()  # 若请求失败（如 Ollama 未启动或接口出错），立即抛出异常
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue  # 跳过空行
            response_json = json.loads(line)  # 每一行是一个独立的 JSON 对象
            if "message" in response_json:
                # 将每个片段的文本内容依次拼接，还原出完整回复
                response_data += response_json["message"]["content"]

    return response_data


def check_if_running(process_name):
    """
    检测系统中是否存在名称包含 process_name 的正在运行的进程。

    作用：
        用于在调用 Ollama 接口之前，先确认 Ollama 服务/进程确实已经启动，
        避免因为服务未运行而导致后续请求失败却难以定位原因。

    参数：
        process_name (str): 要匹配的进程名（子串匹配，如 "ollama"）。

    返回值：
        bool: 如果找到匹配的进程则返回 True，否则返回 False。
    """
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def format_input(entry):
    """
    将一条数据（包含 instruction 和可选的 input 字段）格式化为统一的指令提示文本。

    作用：
        构造与指令微调训练阶段一致的输入格式（Alpaca 风格的
        "### Instruction" / "### Input" 模板），确保评估阶段使用的提示词
        与训练/推理阶段保持一致，从而让裁判模型能正确理解上下文。

    参数：
        entry (dict): 单条数据样本，需包含键 "instruction"，可选包含键 "input"。

    返回值：
        str: 拼接好的指令文本（如果 entry["input"] 非空，则会附加 "### Input" 部分）。
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 只有当 input 字段非空时才追加 "### Input" 部分，与训练时的数据格式保持一致
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def main(file_path):
    """
    评估流程的主入口函数。

    作用：
        1. 检查 Ollama 是否已运行，未运行则直接报错终止。
        2. 加载测试数据集（JSON 文件）。
        3. 调用 generate_model_scores 对每条数据的 "model_response" 字段打分。
        4. 打印成功评分的数量以及平均分，作为该模型整体表现的评估结果。

    参数：
        file_path (str): 测试数据集 JSON 文件的路径，其中每条数据应包含
            "output"（参考答案）和 "model_response"（待评估模型的回答）等字段。

    返回值：
        None（结果通过打印输出）。
    """
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        # 如果 Ollama 服务未启动，后续所有请求都会失败，因此提前终止并给出明确提示
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))

    with open(file_path, "r") as file:
        test_data = json.load(file)  # 加载包含指令、参考答案和模型生成回答的测试集

    model = "llama3"  # 使用 llama3 作为「裁判模型」对回答进行打分
    scores = generate_model_scores(test_data, "model_response", model)
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    # 计算所有成功评分的平均值，作为模型整体表现的量化指标
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")


def generate_model_scores(json_data, json_key, model="llama3"):
    """
    对测试数据集中的每一条样本，使用 Ollama 本地模型作为裁判，为指定字段的
    模型回答打分（0~100分）。

    作用：
        遍历数据集，针对每条数据构造一个「打分提示词」，将指令、参考答案与
        待评估的模型回答一并交给裁判模型，请其输出一个整数分数，最终收集
        所有分数用于计算整体评估指标。

    参数：
        json_data (list[dict]): 测试数据集，每条数据应包含 "instruction"、
            "input"、"output" 以及由 json_key 指定的待评估回答字段。
        json_key (str): 待评估的模型回答在每条数据中对应的字段名
            （例如 "model_response"）。
        model (str): 用作裁判的 Ollama 模型名称，默认为 "llama3"。

    返回值：
        list[int]: 每条数据对应的评分列表（跳过无法解析为整数的评分结果）。
    """
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        if entry[json_key] == "":
            # 如果模型回答为空字符串，直接判 0 分，不必再调用裁判模型
            scores.append(0)
        else:
            # 构造评分提示词：同时提供「输入」「正确答案」「模型回答」，
            # 并明确要求裁判模型只返回一个 0~100 的整数分数，便于程序解析
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            score = query_model(prompt, model)
            try:
                scores.append(int(score))  # 尝试将裁判模型的回复解析为整数分数
            except ValueError:
                # 裁判模型有时可能返回非纯数字的文本，此时跳过该条评分，
                # 避免程序因单条异常而整体崩溃
                print(f"Could not convert score: {score}")
                continue

    return scores


if __name__ == "__main__":

    import argparse

    # 命令行入口：允许用户通过 --file_path 参数指定待评估的测试数据集路径
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate model responses with ollama"
    )
    parser.add_argument(
        "--file_path",
        required=True,
        help=(
            "The path to the test dataset `.json` file with the"
            " `'output'` and `'model_response'` keys"
        )
    )
    args = parser.parse_args()

    main(file_path=args.file_path)
