"""
================================================================================
 Chapter 11 · 监督微调 SFT + 聊天模板 —— 原始课程笔记（大量说明 + 代码骨架）
================================================================================
 这份原始文件以概念说明为主(聊天模板差异/SFT何时用/训练参数/监测)，夹着一些
 SFTTrainer 代码骨架。注意：下面 max_steps=1000 的真训练在 Mac 上很久，且后面几段
 SFTTrainer(dataset=整个DatasetDict) 是示意性写法。能真跑、可上简历的版本见同目录：
   · Chapter11_聊天模板与SFT_学习笔记.py  apply_chat_template 真跑 + SFT 概念系统讲
   · Chapter11_LoRA微调实战.py           真 LoRA 微调闭环(dolly + SmolLM2，几十秒跑通)
 版本：trl 1.12 / peft 0.20 已装；SFTConfig 不支持 warmup_ratio(用 warmup_steps)。
================================================================================
"""
#常用模板格式
import transformers
import trl
from transformers import AutoModelForCausalLM, AutoTokenizer

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "What's the weather?"},
]
# 格式之间的主要区别包括：
#
# 系统消息处理：
#
# Llama 2 将系统消息封装在<<SYS>>标签中
# Llama 3 使用带结尾的<|system|>标签</s>
# Mistral 在第一条指令中包含系统消息
# Qwen 使用带有标签的显式system角色<|im_start|>
# ChatGPT 使用SYSTEM:前缀
# 消息边界：
#
# 羊驼 2 的用途[INST]和[/INST]标签
# Llama 3 使用角色特定的标签（<|system|>，，），并<|user|>带有结尾<|assistant|></s>
# 米斯特拉尔使用[INST]和[/INST]与<s></s>
# Qwen 使用角色特定的开始/结束标记
# 特殊代币：
#
# Llama 2 的用途<s>以及</s>用于对话边界
# Llama 3 用于</s>结束每条消息
# 米斯特拉尔的用途<s>和</s>转弯边界
# Qwen 使用角色特定的开始/结束标记
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
#
# # 这些将自动使用不同的模板
# mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# smol_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
#
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"},
# ]
#
# # 每个消息都会根据其模型的模板进行格式化
# mistral_chat = mistral_tokenizer.apply_chat_template(messages, tokenize=False)
# qwen_chat = qwen_tokenizer.apply_chat_template(messages, tokenize=False)
# smol_chat = smol_tokenizer.apply_chat_template(messages, tokenize=False)

# 高级功能
# 聊天模板不仅可以处理简单的对话互动，还可以处理更复杂的场景，包括：
#
# 工具使用：当模型需要与外部工具或 API 交互时
# 多模态输入：用于处理图像、音频或其他媒体类型
# 函数调用：用于结构化函数执行
# # 多轮上下文：用于维护对话历史记录
# messages = [
#     {
#         "role": "system",
#         "content": "You are a helpful vision assistant that can analyze images.",
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "What's in this image?"},
#             {"type": "image", "image_url": "https://example.com/image.jpg"},
#         ],
#     },
# ]
# messages = [
#     {
#         "role": "system",
#         "content": "You are an AI assistant that can use tools. Available tools: calculator, weather_api",
#     },
#     {"role": "user", "content": "What's 123 * 456 and is it raining in Paris?"},
#     {
#         "role": "assistant",
#         "content": "Let me help you with that.",
#         "tool_calls": [
#             {
#                 "tool": "calculator",
#                 "parameters": {"operation": "multiply", "x": 123, "y": 456},
#             },
#             {"tool": "weather_api", "parameters": {"city": "Paris", "country": "France"}},
#         ],
#     },
#     {"role": "tool", "tool_name": "calculator", "content": "56088"},
#     {
#         "role": "tool",
#         "tool_name": "weather_api",
#         "content": "{'condition': 'rain', 'temperature': 15}",
#     },
# ]

# 最佳实践
# 使用聊天模板时，请遵循以下关键实践：
#
# 格式一致：在整个申请过程中始终使用相同的模板格式。
# 明确角色定义：为每条消息明确指定角色（系统、用户、助手、工具）。
# 上下文管理：维护对话历史记录时，请注意令牌限制。
# 错误处理：包含针对工具调用和多模态输入的适当错误处理
# 验证：在将消息发送到模型之前，验证消息结构。

# #练习
# from datasets import load_dataset
# dataset = load_dataset("HuggingFaceTB/smoltalk")
# def convert_to_chatml(example):
#     return {
#         "messages": [
#             {"role": "user", "content": example["input"]},
#             {"role": "assistant", "content": example["output"]},
#         ]
#     }
#监督式微调
# 需要进行SFT，则是否进行SFT取决于两个主要因素：
#
# 模板控件
# SFT 允许对模型的输出结构进行精确控制。当您需要模型执行以下操作时，这一点尤其重要：
#
# 以特定聊天模板格式生成回复
# 遵循严格的输出格式。
# 保持所有回复的样式一致
# 领域自适应
# 在特定领域工作时，SFT 通过以下方式帮助模型与特定领域的需求保持一致：
#
# 教授领域术语和概念
# 执行专业标准
# 妥善处理技术咨询
# 遵循行业特定准则
#
# 训练配置
# 微调的成功很大程度上取决于训练参数的选择。让我们来探讨每个重要参数以及如何有效地配置它们：
#
# SFTTrainer 的配置需要考虑多个控制训练过程的参数。让我们逐一了解每个参数及其用途：
#
# 训练时长参数：
#
# num_train_epochs控制总训练时长
# max_steps：替代 epoch 的方案，设置最大训练步数
# 更多的训练轮数可以带来更好的学习效果，但也有过拟合的风险。
# 批次大小参数：
#
# per_device_train_batch_size决定内存使用情况和训练稳定性
# gradient_accumulation_steps：可实现更大的有效批量
# 更大的批次可以提供更稳定的梯度，但需要更多内存。
# 学习率参数：
#
# learning_rate控制权重更新的大小
# warmup_ratio用于学习速率热身的训练部分
# 高值会导致不稳定，低值会导致学习速度缓慢。
# 监测参数：
#
# logging_steps指标记录频率
# eval_steps多久评估一次验证数据
# save_steps模型检查点保存频率

#采用 TRL 进行实施
from datasets import  load_dataset
from trl import SFTConfig, SFTTrainer
import torch
# 设置设备
device='cuda' if torch.cuda.is_available() else 'cpu'
# 加载数据集
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")
# 配置模型和分词器
model_name = "HuggingFaceTB/SmolLM2-135M"
model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(device)
## 设置聊天模板
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
# 配置训练器
training_args=SFTConfig(
    output_dir="./sft_output" ,
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy='steps',
    eval_steps=50,
)
# 初始化训练器
trainer=SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset[ "train" ],
    eval_dataset=dataset[ "test" ],
)
trainer.train()
#打包数据集
training_args = SFTConfig(packing=True)
trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args)
trainer.train()
#自定义格式化函数
def formatting_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text


training_args = SFTConfig(packing=True)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
)

#监测培训进度
# 了解损失模式
# 训练损失通常遵循三个不同的阶段：
#
# 初始急剧下降：快速适应新的数据分布
# 逐步稳定：随着模型微调，学习率逐渐减慢
# 收敛：损失值趋于稳定，表明训练完成
# 有效的监测包括跟踪定量指标和评估定性指标。可用的指标有：
#
# 训练损失
# 验证损失
# 学习率提升
# 梯度范数



