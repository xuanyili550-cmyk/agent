"""
================================================================================
 综合案例 · 一个分类器的完整生命周期（串起 Chapter 1 + Chapter 2）
================================================================================
 这是把 Ch1（pipeline 内部原理）和 Ch2（微调）拼成的“端到端真实项目”。
 直接运行本文件即可（会在 mps 上做一次小规模微调，约 30~60 秒）：
     python3 Chapter2_综合案例_预训练到微调到上线.py

 六个阶段（每个阶段都标了它来自哪一章的知识点）：
   阶段1  加载预训练模型 + 分词      [Ch1 分词器 / Ch2 map+collator]
   阶段2  微调（Trainer）            [Ch2 Trainer]
   阶段3  评估                       [Ch2 compute_metrics + evaluate]
   阶段4  保存模型到磁盘             [Ch1 save_pretrained]
   阶段5  重新加载 + 手动推理        [Ch1 分词→logits→softmax→标签]
   阶段6  用 pipeline 封装成“产品”   [Ch1 pipeline 一行封装]

 任务：MRPC —— 判断两句话是否“同义”（二分类：0 不同义 / 1 同义）
================================================================================
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer, pipeline)
import evaluate


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CHECKPOINT = "bert-base-uncased"
SAVE_DIR = "my-mrpc-model"          # 微调后模型保存目录
device = pick_device()
print(f"设备：{device}\n")


# ==============================================================================
# 阶段 1：加载预训练模型 + 数据分词  [Ch1 分词器 + Ch2 map/collator]
# ==============================================================================
print("=" * 72)
print("  阶段 1：加载预训练模型 + 分词")
print("=" * 72)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
# ↑ 提示 classifier 权重是新初始化的（随机分类头），正常——阶段2就是去训练它

raw = load_dataset("nyu-mll/glue", "mrpc")

def tokenize_function(example):
    # [Ch1] 句子对一起编码，自动加 token_type_ids 区分两句；不在此填充
    return tokenizer(list(example["sentence1"]), list(example["sentence2"]), truncation=True)

tokenized = raw.map(tokenize_function, batched=True)     # [Ch2] map 批量分词
collator = DataCollatorWithPadding(tokenizer=tokenizer)  # [Ch2] 动态填充

# 小子集：让整个流程 1 分钟内跑完（真实项目用全量）
train_ds = tokenized["train"].select(range(500))
eval_ds = tokenized["validation"].select(range(200))
print(f"训练集 {train_ds.num_rows} 条，验证集 {eval_ds.num_rows} 条（已取小子集）\n")


# ==============================================================================
# 阶段 2：微调  [Ch2 Trainer]
# ==============================================================================
print("=" * 72)
print("  阶段 2：微调（Trainer 更新模型权重）")
print("=" * 72)

def compute_metrics(eval_preds):                          # [Ch2] 评估函数
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

args = TrainingArguments(
    output_dir="test-trainer-lifecycle",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    eval_strategy="epoch",       # 每 epoch 评估，观察指标变化
    logging_steps=30,
    report_to="none",            # 不连 wandb
    # 注意：mps 不支持 fp16，绝不设 fp16=True
)
trainer = Trainer(model, args, train_dataset=train_ds, eval_dataset=eval_ds,
                  data_collator=collator, processing_class=tokenizer,
                  compute_metrics=compute_metrics)
print("训练中……（每个 epoch 结束会打印 accuracy/f1）")
trainer.train()
print()


# ==============================================================================
# 阶段 3：评估  [Ch2 predict + evaluate]
# ==============================================================================
print("=" * 72)
print("  阶段 3：在验证集上评估最终效果")
print("=" * 72)

pred = trainer.predict(eval_ds)                           # [Ch2] predict → logits
preds = np.argmax(pred.predictions, axis=-1)              # [Ch1] argmax 取类别
metric = evaluate.load("glue", "mrpc")
result = metric.compute(predictions=preds, references=pred.label_ids)
print(f"最终指标：accuracy={result['accuracy']:.3f}  f1={result['f1']:.3f}")
print("（小子集演示；全量 3668 条训练可到 0.85+）\n")


# ==============================================================================
# 阶段 4：保存模型  [Ch1 save_pretrained]
# ==============================================================================
print("=" * 72)
print("  阶段 4：把微调好的模型存到磁盘")
print("=" * 72)

model.save_pretrained(SAVE_DIR)          # 存权重 config.json + model.safetensors
tokenizer.save_pretrained(SAVE_DIR)      # 分词器也要一起存，否则没法用
print(f"已保存到 ./{SAVE_DIR}/ ，包含：", os.listdir(SAVE_DIR), "\n")


# ==============================================================================
# 阶段 5：重新加载 + 手动推理  [Ch1 分词→模型→softmax→标签]
# ==============================================================================
print("=" * 72)
print("  阶段 5：重新加载模型，手动走一遍推理流程")
print("=" * 72)

# 假装是另一个程序/另一天，从磁盘加载刚训练的模型
loaded_tok = AutoTokenizer.from_pretrained(SAVE_DIR)
loaded_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device)
loaded_model.eval()

labels = {0: "不同义", 1: "同义"}
test_pairs = [
    ("He loves playing football.", "He enjoys playing soccer."),          # 期望 同义
    ("The weather is sunny.", "The stock market crashed."),               # 期望 不同义
]
print("手动推理（Ch1 四步：分词 → 模型 → softmax → 取标签）：")
for s1, s2 in test_pairs:
    enc = loaded_tok(s1, s2, truncation=True, return_tensors="pt").to(device)  # [Ch1] return_tensors 带 s
    with torch.no_grad():                                                       # [Ch1] 推理不算梯度
        logits = loaded_model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]                                     # [Ch1] logits→概率
    idx = probs.argmax().item()
    print(f"  「{s1}」vs「{s2}」→ {labels[idx]}（{probs[idx]:.2f}）")
print()


# ==============================================================================
# 阶段 6：用 pipeline 封装成“产品级”接口  [Ch1 pipeline 一行封装]
# ==============================================================================
print("=" * 72)
print("  阶段 6：用 pipeline 把模型封装成一行调用（上线形态）")
print("=" * 72)

# pipeline 内部就是阶段5那四步（Ch1 第9关验证过）。句子对用 text/text_pair 传入。
device_id = 0 if device in ("cuda", "mps") else -1
clf = pipeline("text-classification", model=SAVE_DIR,
               device=device if device == "mps" else device_id)
for s1, s2 in test_pairs:
    out = clf({"text": s1, "text_pair": s2})
    print(f"  「{s1[:25]}…」vs「{s2[:25]}…」→ {out}")

print("\n" + "=" * 72)
print("  ✅ 全生命周期跑完：预训练 → 分词 → 微调 → 评估 → 保存 → 重载 → 推理 → pipeline")
print("  这就是把一个通用大模型，变成解决你具体任务的产品的完整链路。")
print("=" * 72)
