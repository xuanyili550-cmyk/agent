"""
================================================================================
 Chapter 12 · GRPO 强化学习(RL for LLMs) —— 原始课程笔记（含大量 CUDA-only 代码）
================================================================================
 GRPO = DeepSeek-R1 训练“推理模型”的核心算法。这份原始文件覆盖：GRPO 伪代码、
 PyTorch 手写 GRPO 内部、trl GRPOTrainer、各类奖励函数、LoRA+GRPO 微调、Unsloth 版。
 ⚠ 但里面大量东西在你这台 Mac 上跑不了：flash_attention_2 / adamw_8bit /
   paged_adamw_8bit(bitsandbytes) / unsloth / vLLM 全是 CUDA-only；trl 的 GRPOTrainer
   本机还缺 mergekit。所以本文件当“知识点清单”看，能真跑、可上简历的版本见同目录：
   · Chapter12_GRPO原理与奖励_学习笔记.py   GRPO 三大件(组内优势/奖励函数/裁剪损失)可跑可测
   · Chapter12_GRPO微调实战.py             手写 GRPO 训练循环 + LoRA，Mac 真跑、看合规率上升
   · Chapter12_案例闯关_GRPO实战.py         菜单式实战
 注：下面第 37 行原有 `from pyexpat.errors import messages` 是笔误(messages 未定义)，已删。
================================================================================
"""
#GRPO算法的伪代码
# 输入：
# - initial_policy：开始训练的模型
# - reward_function：评估输出的函数
# - training_prompts：训练示例集 -
# group_size：每个提示的输出数量（通常为
# 4 - 16）
#
# 算法GRPO：
# 1.
# 对于每次训练迭代：
# a.设置
# reference_policy = initial_policy（当前策略的快照）
# b.对于批次中的每个提示：
# i.使用
# initial_policy
# 生成
# group_size
# 个不同的输出
# ii.使用
# reward_function
# 计算每个输出的奖励
# iii.对组内奖励进行归一化：
# normalized_advantage = (reward - mean(rewards)) / std(rewards)
# iv.通过最大化裁剪比率
#
# 来更新策略： min(prob_ratio * normalized_advantage,
#                 clip(prob_ratio, 1 - epsilon, 1 + epsilon) * normalized_advantage)
# - kl_weight * KL(初始策略 | | 参考策略)
#
# 其中
# prob_ratio
# 为
# current_prob / reference_prob
#
# 输出：优化后的策略模型

#加载模型并生成响应 PyTorch 中实现 GRPO
import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_name = "Qwen/Qwen2-Math-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# 如果可用，则将模型移至 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 输入提示
prompt = "Solve y = 2x + 1 for x = 2, y = "  # 正确答案：5
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(device) # 形状：(1, prompt_len)
attention_mask = inputs["attention_mask"].to(device)

# 步骤 1：生成 8 个响应（B = 2 组，G = 每组 4 个响应）
batch_size, num_generations = 2, 4
outputs = model.generate(
    input_ids=input_ids,   # 形状：(1, prompt_len)
    attention_mask=attention_mask,
    max_new_tokens=1,   # seq_len = 1 (每个响应一个 token)
    num_return_sequences=batch_size * num_generations,  # 总共 8 个响应
    do_sample=True,
    top_k=10,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
)
#计算各组奖励的均值和标准差
# 形状：(B * G,) = (8,) 因为我们有 2 组 4 代数据，需要将其展平。
rewards = torch.tensor([1, 0, 0, 1, 0, 0, 1, 1], dtype=torch.float32)
num_generations = 4

# 分组奖励：形状（B，G）= 2，4）
rewards_grouped = rewards.view(-1, num_generations)

# 每组平均值：形状 (B,) = (2,)
mean_grouped_rewards = rewards_grouped.mean(dim=1)

# 每组的标准差：形状 (B,) = (2,)
std_grouped_rewards = rewards_grouped.std(dim=1)

# 广播以匹配奖励并进行归一化：形状 (B * G,) = (8,)
# 为什么需要广播？因为我们需要计算组内每个响应的优势值
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
#
# Grouped Rewards: tensor([[1., 0., 0., 1.],
#                         [0., 0., 1., 1.]])
# Mean per group: tensor([0.5000, 0.5000])
# Std per group: tensor([0.5774, 0.5774])
# Broadcasted Mean: tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000])
# Broadcasted Std: tensor([0.5774, 0.5774, 0.5774, 0.5774, 0.5774, 0.5774, 0.5774, 0.5774])

# Advantages: Shape (B * G,) = (8,)
## 优势=：形状 (B * G,) = (8,)
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)

# 将形状 (B * G, 1) = (8, 1) 调整为与 logits 形状匹配
advantages = advantages.unsqueeze( 1 )
# 计算新旧政策之间的概率比
# 形状：(B*G, seq_len) seq_len 是输出的长度，即生成的标记数，为了简单起见，这里我们假设它是 1 # (8, 1)
ratio = torch.exp(
    new_per_token_logps - per_token_logps
)
# 裁剪函数
eps = self.cliprange   # 例如 0.2
pg_losses1 = -advantages * ratio   # 形状：(B*G, seq_len) #(8, 1)
pg_losses2 = -advantages * torch.clamp(
    ratio, 1.0 - eps, 1.0 + eps
)   # 形状：(B*G, seq_len) #(8, 1)
pg_loss_max = torch.max ( pg_losses1, pg_losses2)   # 形状：(B*G, seq_len) #(8, 1)


# 现在与 KL 惩罚项结合 # 形状：(B*G, seq_len) #(8, 1)
per_token_loss = pg_loss_max + self.beta * per_token_kl

# Shape: (B*G, seq_len) #(8, 1)
per_token_kl = F.kl_div(
    F.log_softmax(new_per_token_logps, dim=-1),
    F.softmax(per_token_logps, dim=-1),
    reduction="none",
).sum(dim=-1, keepdim=True)


#在 TRL 中实施 GRPO
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# 1. 加载数据集
dataset = load_dataset("your_dataset", split="train")


# 2. 定义一个简单的奖励函数
def reward_func(completions, **kwargs):
    """Example: Reward longer completions"""
    return [float(len(completion)) for completion in completions]


#3. 配置培训
training_args = GRPOConfig(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
)

# 4. 初始化和训练
trainer = GRPOTrainer(
    model="your_model",  # e.g. "Qwen/Qwen2-0.5B-Instruct"
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_func,
)
trainer.train()
# 奖励函数
# 示例 1：基于完成长度的奖励
def reward_length(completions, **kwargs):
    return [float(len(completion)) for completion in completions]


# 示例 2：基于匹配模式的奖励
import re


def reward_format(completions, **kwargs):
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return [1.0 if re.match(pattern, c) else 0.0 for c in completions]

#训练配置
training_args = GRPOConfig(
    # 基本参数
    output_dir="output",
    num_train_epochs=3,
    num_generation=4,  # 每个提示要生成的完成次数
    per_device_train_batch_size=4,  # 我们希望在一个设备批次中获得所有代数
    # 可选但有用
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    logging_steps=10,
    # GRPO 特有（可选）
    use_vllm=True,  # 加快生成速度
)

# 成功秘诀
# 内存管理：根据您的GPU内存per_device_train_batch_size进行调整。gradient_accumulation_steps
# 速度：use_vllm=True如果您的模型受支持，请启用此功能以加快生成速度。
# 监控：观察训练期间记录的指标：
# reward：完成任务的平均奖励
# reward_std奖励组内的标准差
# kl：与参考模型的KL散度
#. 基于长度的奖励
def reward_len(completions, **kwargs):
    ideal_length = 20
    return [-abs(ideal_length - len(completion)) for completion in completions]
#基于规则的、针对可验证任务的奖励
def problem_reward(completions, answers, **kwargs):
    """Reward function for math problems with verifiable answers
    completions: list of completions to evaluate
    answers: list of answers to the problems from the dataset
    """

    rewards = []
    for completion, correct_answer in zip(completions, answers):
        # Extract the answer from the completion
        try:
            # 这是一个简化的示例- 您需要进行正确的解析
            answer = extract_final_answer(completion)
            # 二元奖励：答对得1分，答错得0分；
            reward = 1.0 if answer == correct_answer else 0.0
            rewards.append(reward)
        except:
            # 如果无法解析答案，则给予较低的奖励
            rewards.append(0.0)

    return rewards

#基于形式的奖励
def format_reward(completions, **kwargs):
    """Reward completions that follow the desired format"""
    # 示例：检查补全是否符合“思考-回答”格式
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"

    rewards = []
    for completion in completions:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            # 检查两个部分是否都有实质性内容
            think_content = match.group(1).strip()
            answer_content = match.group(2).strip()

            if len(think_content) > 20 and len(answer_content) > 0:
                rewards.append(1.0)
            else:
                rewards.append(
                    0.5
                )   # 格式正确但内容有限者可获得部分奖励，
        else:
            rewards.append(0.0)   # 格式错误不给予奖励

    return rewards

#使用 GRPO 对模型进行微调
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
dataset = load_dataset("mlabonne/smoltldr")
print(dataset)
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 加载 LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

# 奖励函数
ideal_length = 50
def  reward_len ( completions, **kwargs ):
     return [ -abs (ideal_length - len (completion)) for completion in completions]

# 训练论点
training_args = GRPOConfig(
    output_dir="GRPO",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=96,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=1,
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_len],
    args=training_args,
    train_dataset=dataset["train"],
)

# 训练模型
wandb.init(project="GRPO")
trainer.train()

merged_model = trainer.model.merge_and_unload()
merged_model.push_to_hub(
    "SmolGRPO-135M", private=False, tags=["GRPO", "Reasoning-Course"]
)

#利用该模型生成文本
# Generate text
from transformers import pipeline

generator = pipeline("text-generation", model="SmolGRPO-135M")

## 或者使用我们之前定义的模型和分词器
# generator = pipeline("文本生成", model=model, tokenizer=tokenizer)

generate_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.5,
    "min_p": 0.1,
}

generated_text = generator(messages, generate_kwargs=generate_kwargs)

print(generated_text)


#使用 Unsloth 进行 GRPO 编程
from unsloth import FastLanguageModel
import torch


max_seq_length = 1024   # 可以增加此值以生成更长的推理轨迹
lora_rank = 32   # rank 值越大，算法越智能，但速度越慢

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= "google/gemma-3-1b-it" ,
    max_seq_length=max_seq_length,
    load_in_4bit= True ,   # 对于 LoRA 16 位，设置为 False
    fast_inference= True ,   # 启用 vLLM 快速推理
    max_lora_rank=lora_rank,
    gpu_memory_utilization= 0.6 ,   # 如果内存不足则降低
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,   # 选择任意大于 0 的数字！建议选项：8、16、32、64、128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # 如果内存不足，则移除 QKVO
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # 启用长时间上下文微调
    random_state=3407,
)
#数据准备
# 定义系统提示，指示模型使用特定格式
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""
import re
from datasets import load_dataset, Dataset


## 用于从不同格式中提取答案的辅助函数
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# 用于准备 GSM8K 数据集的函数
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


dataset = get_gsm8k_questions()

#定义奖励函数
# 检查答案是否正确的奖励函数
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


# 检查答案是否正确的奖励函数是一个整数
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# 检查补全是否符合严格格式的奖励函数
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# 检查补全是否遵循更宽松格式的奖励函数
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# 奖励函数，用于统计 XML 标签并惩罚额外内容
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

#与 GRPO 一起接受培训

from trl import GRPOConfig, GRPOTrainer

max_prompt_length = 256

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # 增加到 4 以获得更平滑的训练
    num_generations=6,  # 如果内存不足则减少
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # 设置为 1 以进行完整训练
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",  # 可以使用权重和偏差
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)
# 该GRPOConfig设置用于训练的各种超参数：
#
# use_vllm：支持vLLM的快速推理
# learning_rate控制模型学习的速度
# num_generations：每个提示要生成的完成次数
# max_steps：要执行的训练步骤总数

trainer.train()

#模型测试
model.save_lora("grpo_saved_lora")
from vllm import SamplingParams

text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Calculate pi."},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = (
    model.fast_generate(
        text,
        sampling_params=sampling_params,
        lora_request=model.load_lora("grpo_saved_lora"),
    )[0]
    .outputs[0]
    .text
)

print(output)

# 保存为 16 位精度
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")