"""
================================================================================
 Chapter 2 · 微调模型 (Fine-tuning) —— 系统学习笔记
================================================================================
 配套 HuggingFace 课程「Fine-tuning a model」章节。
 主线：拿一个预训练模型(bert-base-uncased) → 在自己的数据(MRPC)上微调 → 得到能用的分类器。

 一张总流程图（务必记住这条主线）：
   原始数据集        分词处理           训练               评估
   load_dataset  →  map(tokenize)  →  Trainer.train()  →  compute_metrics
   (MRPC句子对)     +DataCollator     (更新模型权重)      (accuracy/f1)

 本文件结构：
   第 0 部分  微调是什么？为什么不用从零训练？
   第 1 部分  数据处理：map + tokenize_function + DataCollatorWithPadding
   第 2 部分  Trainer 高层 API：TrainingArguments / Trainer 每个参数
   第 3 部分  评估：compute_metrics + evaluate(glue/mrpc)
   第 4 部分  进阶 TrainingArguments（fp16 / 梯度累积 / 学习率调度 / 提前停止）
   第 5 部分  手写训练循环（不用 Trainer，理解底层每一步）
   第 6 部分  Accelerate：几行改动支持多卡/混合精度
   第 7 部分  ★常见报错速查（你这一路踩过的坑，全在这）

 术语速记：
   微调 fine-tuning   在预训练模型基础上，用少量任务数据继续训练，让它学会你的任务
   epoch             把整个训练集完整过一遍叫一个 epoch
   batch             一次喂给模型的一小撮样本（如 8 条）
   loss 损失          模型预测和真实标签的差距，训练目标就是把它降下来
   optimizer 优化器   根据 loss 的梯度更新模型权重（如 AdamW）
   scheduler 调度器   训练过程中动态调整学习率（如 linear 线性衰减）
   MRPC              GLUE 基准里的一个任务：判断两句话是否同义（二分类）
================================================================================
"""

# ==============================================================================
# 第 0 部分：微调是什么？
# ==============================================================================
# 预训练模型(bert-base-uncased)已经在海量文本上学会了“语言的通用规律”，但它不认识
# 你的具体任务（比如“判断两句话是否同义”）。微调 = 在它已有的知识上，用你的小数据
# 继续训练一小会儿，加一个“任务头”(分类头)并调整权重，让它专精你的任务。
#
# 为什么不从零训练？从零训练 BERT 要几百 GB 语料 + 几千 GPU 小时，个人不可能。
# 微调只需几千条数据 + 几分钟~几小时，就能得到很好的效果。这就是迁移学习的威力。
#
# ⚠️ 你会看到这个提示（正常，不是错误）：
#   "classifier.weight/bias | MISSING | newly initialized"
#   意思：bert-base-uncased 没有分类头，加载成分类模型时新建了一个随机分类头，
#   提示你“记得训练它”——微调干的就是这件事。


# ==============================================================================
# 第 1 部分：数据处理 —— 从原始数据集到模型能吃的 batch
# ==============================================================================
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("nyu-mll/glue", "mrpc")   # ← 注意 namespace/name，见第7部分
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# ---- tokenize_function：把“句子对”一起编码 ----
def tokenize_function(example):
    # example 是一个 batch（dict，值是列表）。sentence1/sentence2 一起传 → 句子对编码，
    # 会自动用 token_type_ids 区分 A/B 句（这正是 Chapter 1 学的）。
    # 注意：不在这里 padding！留给 DataCollator 动态填充（更省算力）。
    return tokenizer(list(example["sentence1"]), list(example["sentence2"]), truncation=True)
    # ↑ list(...) 是因为新版 datasets 里 example["列"] 是 Column 对象，不是 list（见第7部分）

# ---- map：把函数批量应用到整个数据集 ----
# batched=True：一次处理一批（几千条），比逐条快得多
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# ---- DataCollatorWithPadding：动态填充 ----
# 作用：组 batch 时，把这一批补齐到“批内最长”，而不是固定 512。省显存、省算力。
# （原理在 Chapter 1 综合案例场景3 用 25.6 倍算力对比演示过）
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---- 手写训练循环还需要这三步“后处理”（Trainer 会自动做，手写要自己来）----
# 1) 删掉模型不认识的列（原始文本列、idx）
# 2) label 列改名成 labels（模型 forward 期望参数名叫 labels）
# 3) 设成 torch 格式（返回张量而非 list）
# tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# tokenized_datasets.set_format("torch")
# → 处理后每条有：input_ids / attention_mask / token_type_ids / labels


# ==============================================================================
# 第 2 部分：Trainer 高层 API —— 最省事的微调方式
# ==============================================================================
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer

# ---- TrainingArguments：所有训练超参数的容器 ----
# 最简形式：只给一个输出目录，其余全用默认值，就够基础微调了。
training_args = TrainingArguments("test-trainer")
#
# 常用参数（第4部分讲进阶的）：
#   output_dir                  必填。保存模型和检查点的目录
#   eval_strategy="epoch"       每个 epoch 结束时在验证集上评估一次（"no"/"steps"/"epoch"）
#   num_train_epochs=3          训练几轮（默认 3）
#   per_device_train_batch_size 每个设备的训练 batch 大小（默认 8）
#   per_device_eval_batch_size  评估 batch 大小
#   learning_rate=5e-5          学习率（微调常用 1e-5 ~ 5e-5）
#   weight_decay=0.01           权重衰减，正则化防过拟合
#   logging_steps=10            每多少步打印一次训练指标
#   save_strategy / save_steps  保存检查点的策略
#   ⚠️ fp16=True 只能在 NVIDIA GPU 上用，你的 Mac(mps) 不支持，别开（见第7部分）

# ---- 模型：加载预训练权重 + 新建 num_labels 个输出的分类头 ----
# num_labels=2：MRPC 是二分类（同义 / 不同义）
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# ---- Trainer：把模型、参数、数据、collator、分词器打包，一键训练 ----
trainer = Trainer(
    model,                                         # 要训练的模型
    training_args,                                 # 超参数
    train_dataset=tokenized_datasets["train"],     # 训练集
    eval_dataset=tokenized_datasets["validation"], # 验证集
    data_collator=data_collator,                   # 动态填充
    processing_class=tokenizer,                    # 分词器（新版参数名，旧版叫 tokenizer=）
)
# trainer.train()      # ← 真正开始微调，会更新 model 的权重
# trainer.predict(...) # ← 用训练好的模型做预测


# ==============================================================================
# 第 3 部分：评估 —— 训练完到底准不准？
# ==============================================================================
import numpy as np
import evaluate

# trainer.predict 返回一个具名元组，含三部分：
#   .predictions  模型输出的 logits，形状 [样本数, 标签数]，如 [408, 2]
#   .label_ids    真实标签
#   .metrics      损失和耗时等
#
# predictions = trainer.predict(tokenized_datasets["validation"])
# preds = np.argmax(predictions.predictions, axis=-1)   # logits → 取最大 → 预测类别
#
# metric = evaluate.load("glue", "mrpc")                # MRPC 官方指标：accuracy + f1
# print(metric.compute(predictions=preds, references=predictions.label_ids))
# # {'accuracy': 0.857..., 'f1': 0.899...}

# ---- compute_metrics：让 Trainer 在每个 epoch 自动报告指标 ----
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds            # Trainer 会把 (logits, labels) 传进来
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 把 compute_metrics 传给 Trainer，配合 eval_strategy="epoch"，
# 训练时每个 epoch 结束就会自动打印 accuracy / f1：
# training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
# trainer = Trainer(..., compute_metrics=compute_metrics)


# ==============================================================================
# 第 4 部分：进阶 TrainingArguments（性能与稳定性调优）
# ==============================================================================
# ⚠️ 下面很多是 GPU 场景的技巧，Mac(mps) 上部分不适用，已标注。
#
# ---- 混合精度 fp16（加速+省显存）----
# TrainingArguments("test-trainer", fp16=True)
#   ❌ 只在 NVIDIA GPU 有效。你的 Mac 开了会报错，别用。
#
# ---- 梯度累积（用小 batch 模拟大 batch，省显存）----
# TrainingArguments(..., per_device_train_batch_size=4, gradient_accumulation_steps=4)
#   有效 batch = 4 × 4 = 16。显存不够时的救命技巧，Mac 上可用。
#
# ---- 学习率调度 ----
# TrainingArguments(..., learning_rate=2e-5, lr_scheduler_type="cosine")
#   默认 linear 线性衰减；可换 cosine 等。
#
# ---- 提前停止（防过拟合）----
# from transformers import EarlyStoppingCallback
# TrainingArguments(..., eval_strategy="steps", eval_steps=100,
#                   save_strategy="steps", save_steps=100,
#                   load_best_model_at_end=True,
#                   metric_for_best_model="eval_loss", greater_is_better=False,
#                   num_train_epochs=10)   # 设高一点，靠 early stop 自动停
# Trainer(..., callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
#   patience=3：验证指标连续 3 次没变好就停。


# ==============================================================================
# 第 5 部分：手写训练循环 —— 不用 Trainer，看清底层每一步
# ==============================================================================
# Trainer 把很多事自动化了。手写一遍能真正理解“训练”在干嘛。五个核心步骤：
#   forward（前向算 loss）→ backward（反向算梯度）→ step（更新权重）
#   → scheduler.step（调学习率）→ zero_grad（清空梯度）
#
# from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from transformers import get_scheduler
# import torch
#
# # 1) DataLoader：负责按 batch 迭代、打乱、调用 collator 填充
# train_dl = DataLoader(tokenized_datasets["train"], shuffle=True,
#                       batch_size=8, collate_fn=data_collator)
# eval_dl  = DataLoader(tokenized_datasets["validation"],
#                       batch_size=8, collate_fn=data_collator)
#
# # 2) 优化器 + 学习率调度器
# optimizer = AdamW(model.parameters(), lr=5e-5)
# num_epochs = 3
# num_training_steps = num_epochs * len(train_dl)
# lr_scheduler = get_scheduler("linear", optimizer=optimizer,
#                              num_warmup_steps=0, num_training_steps=num_training_steps)
#
# # 3) 设备：Mac 用 mps
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# model.to(device)
#
# # 4) 训练循环
# model.train()                                  # 训练模式
# for epoch in range(num_epochs):
#     for batch in train_dl:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)               # 前向，模型自动算 loss（因为有 labels）
#         loss = outputs.loss
#         loss.backward()                        # 反向传播，算梯度
#         optimizer.step()                       # 用梯度更新权重
#         lr_scheduler.step()                    # 调整学习率
#         optimizer.zero_grad()                  # 清空梯度（否则会累加）
#
# # 5) 评估循环
# import evaluate
# metric = evaluate.load("glue", "mrpc")
# model.eval()                                   # 评估模式
# for batch in eval_dl:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():                      # 评估不算梯度
#         outputs = model(**batch)
#     preds = torch.argmax(outputs.logits, dim=-1)
#     metric.add_batch(predictions=preds, references=batch["labels"])
# print(metric.compute())


# ==============================================================================
# 第 6 部分：Accelerate —— 手写循环的“分布式升级版”
# ==============================================================================
# Accelerate 让同一份手写循环，不改结构就能跑在多卡/混合精度/TPU 上。
# 相比第5部分，只有 3 处改动：
#   ① accelerator.prepare() 包装 dataloader/model/optimizer
#   ② 删掉手动 .to(device)（accelerator 自动处理）
#   ③ loss.backward() → accelerator.backward(loss)
#
# from accelerate import Accelerator
# accelerator = Accelerator()
# train_dl, eval_dl, model, optimizer = accelerator.prepare(
#     train_dataloader, eval_dataloader, model, optimizer)
# ...
#     outputs = model(**batch)          # 不用再 .to(device)
#     accelerator.backward(outputs.loss)  # 替代 loss.backward()
#
# 你单机单 Mac 用不上多卡，但理解它：这就是 Trainer 底层用来管设备的东西
# （所以第一步要装 accelerate）。


# ==============================================================================
# 第 7 部分：★常见报错速查
# ==============================================================================
# ┌────────────────────────────────────────────┬──────────────────────────────┐
# │ 报错                                          │ 原因 & 解决                   │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ 'list' object has no attribute 'size'        │ return_tensor 少了 s。         │
# │                                              │ → 改 return_tensors='pt'      │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ HfUriError: Repository id must be            │ 裸名 "glue" 新版不支持。        │
# │ 'namespace/name', got 'glue'                 │ → load_dataset("nyu-mll/glue",│
# │                                              │   "mrpc")                     │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ ValueError: text input must be of type str.. │ dataset["列"] 在 datasets 5.x │
# │                                              │ 返回 Column 不是 list。        │
# │                                              │ → 用 list(...) 包一层          │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ ImportError: Trainer requires accelerate     │ 没装 accelerate。              │
# │                                              │ → pip install accelerate      │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ ImportError: need scikit-learn, scipy        │ GLUE 指标依赖 sklearn。        │
# │ (for evaluate glue)                          │ → pip install scikit-learn    │
# │                                              │   scipy（不是 sklearn！）      │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ fp16 相关报错 / mps 上训练异常                 │ mps 不支持 fp16 混合精度。      │
# │                                              │ → 别设 fp16=True               │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ ModuleNotFoundError: No module named 'wandb' │ 课程示例用了 wandb 实验追踪，   │
# │                                              │ 没装就别跑那段（可选功能）。    │
# └────────────────────────────────────────────┴──────────────────────────────┘
#
# Mac 微调额外提醒：
#   * 设备用 mps：torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#   * 全量 MRPC(3668条) × 3 epoch 在 M4 上要几分钟~十几分钟。想先验证流程，
#     用 tokenized_datasets["train"].select(range(200)) 取小子集跑通再说。
#   * 别开 fp16；wandb/bitsandbytes 这些可选的，没装就跳过。
