"""
================================================================================
 Chapter 2 挖空练习 · 微调全流程（数据 → Trainer → 评估）
================================================================================
 玩法：
   1) 把每个 ______ 换成正确代码（凭记忆）
   2) 运行：python3 Chapter2_挖空练习.py  （用小子集，几十秒；mps 加速）
   3) 没填的报 NameError；填错报错或指标异常
   4) 卡住 → 文件底部「答案区」
 目标：在 MRPC 小子集上微调 BERT，并打印 accuracy/f1。
 ⚠ 本机说明：此练习用 datasets 库(load_dataset/map)，而本机 Python3.14 下 datasets 的 pickle 坏了
   (会报 Pickler._batch_setitems)，需在【正常环境(Colab/云)】跑。本机可跑的等价版(pandas 直读)见
   ../实战练习/chapter实战/分章项目/Ch2_文本分类微调.py。答案区仍是标准 datasets 写法，可照记。
================================================================================
"""
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
import evaluate

ckpt = "bert-base-uncased"

# 练习1：加载 GLUE 的 MRPC 数据集
#   提示：load_dataset("命名空间/名字", "子任务")；GLUE 在 nyu-mll/glue，子任务 mrpc
raw = load_dataset("nyu-mll/glue", "mrpc")

tokenizer = AutoTokenizer.from_pretrained(ckpt)

# 练习2：分词函数——句子对一起编码、截断、不填充
#   提示：新版 datasets 里 example["列"] 是 Column，要用 list(...) 包一层
def tokenize_function(example):
    return tokenizer(list(example["sentence1"]), list(example["sentence2"]), truncation=True)

# 练习3：把分词函数批量应用到整个数据集
#   提示：raw.map(函数, batched=?)
tokenized = raw.map(tokenize_function, batched=True)

# 练习4：动态填充器
#   提示：DataCollatorWithPadding(tokenizer=?)
collator =DataCollatorWithPadding(tokenizer=tokenizer)

# 小子集（不用填）
train_ds = tokenized["train"].select(range(500))
eval_ds = tokenized["validation"].select(range(200))

# 练习5：加载模型，指定 2 个分类标签
#   提示：AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=?)
model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)

# 练习6：评估函数——logits 取 argmax 后和标签比
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)          # 提示：logits→类别，用哪个 numpy 函数？
    return metric.compute(predictions=preds, references=labels)

# 练习7：训练参数
#   要求：输出目录随便起、训 1 轮、每轮评估、不连 wandb
#   提示：TrainingArguments(输出目录, num_train_epochs=?, eval_strategy=?, report_to=?)
args = TrainingArguments(
    "test-trainer-practice",
    num_train_epochs=1,
    eval_strategy="epoch",     # 每个 epoch 评估 → 填 "epoch"
    report_to="none",
)

# 练习8：组装 Trainer
#   提示：Trainer(model, args, train_dataset=?, eval_dataset=?,
#                data_collator=?, processing_class=?, compute_metrics=?)
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# 练习9：开始训练（提示：trainer 的哪个方法？）
trainer.train()

print("✅ 训练完成，上面每个 epoch 会打印 eval_accuracy / eval_f1")


# ==============================================================================
#  答案区（卡住再看！）
# ------------------------------------------------------------------------------
#  1: load_dataset("nyu-mll/glue", "mrpc")
#  2: list(example["sentence2"]), truncation=True
#  3: tokenized = raw.map(tokenize_function, batched=True)
#  4: collator = DataCollatorWithPadding(tokenizer=tokenizer)
#  5: num_labels=2
#  6: preds = np.argmax(logits, axis=-1)
#  7: num_train_epochs=1, eval_strategy="epoch"
#  8: train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator,
#     processing_class=tokenizer, compute_metrics=compute_metrics
#  9: trainer.train()
# ==============================================================================
