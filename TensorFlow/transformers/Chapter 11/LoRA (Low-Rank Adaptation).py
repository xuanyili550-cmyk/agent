"""
================================================================================
 Chapter 11 · LoRA（低秩自适应）—— 原始课程笔记（仅概念 + 加载适配器片段）
================================================================================
 这份原始文件主要是概念说明 + 一小段“加载现成 LoRA 适配器”的代码。
 能在你机器上真跑、且能上简历的**完整 LoRA 微调闭环**见同目录：
   · Chapter11_LoRA微调实战.py       真训练(dolly + SmolLM2-135M)：LoRA只训0.34%参数→
                                     保存1.9MB适配器→加载推理→merge_and_unload→QLoRA参考
   · Chapter11_聊天模板与SFT_学习笔记.py  聊天模板 apply_chat_template + SFT 概念(可跑)
 注：下面 PeftModel.from_pretrained 会下载 opt-350m 基座(~660MB)，只为演示“加载别人训好的
     LoRA 适配器”。peft 0.20 已装，API 兼容。
================================================================================
"""
#LoRA（低秩自适应）
# LoRA的主要优势
# 内存效率：
#
# GPU内存中仅存储适配器参数。
# 基础模型权重保持不变，可以以较低精度加载。
# 支持在消费级GPU上对大型模型进行微调
# 训练功能：
#
# 原生 PEFT/LoRA 集成，设置极简
# 支持 QLoRA（量化 LoRA），以实现更高的内存效率
# 适配器管理：
#
# 检查点期间适配器重量减轻
# 将适配器合并回基础模型的功能
#使用 PEFT 加载 LoRA 适配器
from peft import PeftConfig,PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
# 加载数据集
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")
# 加载 LoRA 配置
config = PeftConfig.from_pretrained( "ybelkada/opt-350m-lora" )
# 加载基础模型对应的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
# 将 LoRA 权重加载到基础模型上
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora" )
# 使用 trl 和 SFTTrainer 对 LLM 进行微调，并结合 LoRA 使用。
# SFTTrainer通过PEFT库与 LoRa 适配器集成。这意味着我们可以像使用SFT一样微调模型，
# 但使用 LoRa 来减少需要训练的参数数量。trl
#
# 我们将使用LoRAConfigPEFT 中的类作为示例。设置过程只需要几个配置步骤：
#
# 定义 LoRa 配置（rank、alpha、dropout）
# 使用 PEFT 配置创建 SFTrainer
# 训练并保存适配器权重。

# LoRA 配置
# 让我们一起来了解一下 LoRA 的配置和关键参数。
#
# 范围	描述
# r（秩）	用于权重更新的低秩矩阵的维度。通常介于 4 到 32 之间。较低的值可以提供更高的压缩率，但可能会降低表达能力。
# lora_alpha	LoRA层的缩放因子通常设置为秩值的2倍。更高的值会带来更强的自适应效果。
# lora_dropout	LoRA 层的 Dropout 概率通常为 0.05-0.1。较高的值有助于防止训练过程中过拟合。
# bias	控制偏置项的训练。选项包括“none”、“all”或“lora_only”。为了提高内存效率，通常选择“none”。
# target_modules	指定要应用 LoRa 的模型模块。可以是“全线性”或特定模块，例如“q_proj,v_proj”。模块越多，适应性越强，但内存占用也会增加。

#将 TRL 与 PEFT 结合使用
from peft import LoraConfig # TODO：配置 LoRA 参数# r：LoRA 更新矩阵的秩维度（越小压缩率越高）
rank_dimension = 6 # lora_alpha：LoRA 层的缩放因子（越高自适应能力越强）
lora_alpha = 8 # lora_dropout：LoRA 层的 dropout 概率（有助于防止过拟合）
lora_dropout = 0.05
# 最大序列长度
max_seq_length = 512
peft_config = LoraConfig(
    r=rank_dimension,   # 秩维度 - 通常介于 4-32 之间
    lora_alpha=lora_alpha,   # LoRA 缩放因子 - 通常为秩的 2 倍
    lora_dropout=lora_dropout,   # LoRA 层的 dropout 概率
    bias= "none" ,   # LoRA 的偏置类型。相应的偏置将在训练期间更新。
    target_modules= "all-linear" ,   # 要应用 LoRA 的模块
    task_type= "CAUSAL_LM" ,   # 模型架构的任务类型
)
#上面我们会device_map="auto"自动将型号分配给正确的设备。
# 您也可以使用手动方式将型号分配给特定设备device_map={"": device_index}。

#我们还需要定义SFTTrainerLoRA 配置。
# 使用 LoRA 配置创建 SFTTrainer
args = SFTConfig(
    output_dir="./results",
    max_length=max_seq_length,     # 最大序列长度
)
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset[ "train" ],
    peft_config=peft_config,   # LoRA 配置
    max_seq_length=max_seq_length,   # 最大序列长度
    processing_class=tokenizer,
)

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel # 1. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained( "base_model_name" , torch_dtype=torch.float16, device_map= "auto"
) # 2. 加载带有适配器的 PEFT 模型
peft_model = PeftModel.from_pretrained(
    base_model, "path/to/adapter" , torch_dtype=torch.float16
) # 3. 将适配器权重与基础模型合并
merged_model = peft_model.merge_and_unload()

# 同时保存模型和分词器
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")


