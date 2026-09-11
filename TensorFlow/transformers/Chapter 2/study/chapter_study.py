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
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

CHECKPOINT = "bert-base-uncased"
SAVE_DIR = "my-mrpc-model"

print("=" * 72)
print("  阶段 1：加载预训练模型 + 分词")
print("=" * 72)
tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT)
model=AutoModelForSequenceClassification.from_pretrained(CHECKPOINT,num_labels=2)
raw=load_dataset('nyu-mll/glue','mrpc')
def tokenize_function(example):
    return tokenizer(list(example["sentence1"]),list(example["sentence2"]),truncation=True)
tokenized=raw.map(tokenize_function,batched=True)
collator=DataCollatorWithPadding(tokenizer=tokenizer)

train_ds=tokenized["train"].select(range(500))
eval_ds=tokenized["validation"].select(range(200))
print(f"训练集 {train_ds.num_rows} 条，验证集 {eval_ds.num_rows} 条（已取小子集）\n")

print("=" * 72)
print("  阶段 2：微调（Trainer 更新模型权重）")
print("=" * 72)

def compute_metrics(eval_preds):
    metrics=evaluate.load("glue","mrpc")
    logits,labels=eval_preds
    pred=np.argmax(logits,axis=-1)
    return metrics.compute(predictions=pred,references=labels)

args = TrainingArguments(
    output_dir="test-trainer-lifecycle",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    eval_strategy="epoch",       # 每 epoch 评估，观察指标变化
    logging_steps=30,
    report_to="none",            # 不连 wandb
    fp16=False
    # 注意：mps 不支持 fp16，绝不设 fp16=True
)
trainer=Trainer(model, args, train_dataset=train_ds,eval_dataset=eval_ds,
                data_collator=collator,processing_class=tokenizer
                ,compute_metrics=compute_metrics
                )
print("训练中……（每个 epoch 结束会打印 accuracy/f1）")
trainer.train()

print("=" * 72)
print("  阶段 3：在验证集上评估最终效果")
print("=" * 72)


pred=trainer.predict(eval_ds)
preds=np.argmax(pred.predictions,axis=-1)
metric=evaluate.load('glue','mrpc')
result=metric.compute(predictions=preds,references=pred.label_ids)
print(f"最终指标：accuracy={result['accuracy']:.3f}  f1={result['f1']:.3f}")
print("（小子集演示；全量 3668 条训练可到 0.85+）\n")


print("=" * 72)
print("  阶段 4：把微调好的模型存到磁盘")
print("=" * 72)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"已保存到 ./{SAVE_DIR}/ ，包含：", os.listdir(SAVE_DIR), "\n")


print("=" * 72)
print("  阶段 5：重新加载模型，手动走一遍推理流程")
print("=" * 72)

loaded_tok = AutoTokenizer.from_pretrained(SAVE_DIR)
loaded_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(pick_device())
loaded_model.eval()
labels = {0: "不同义", 1: "同义"}
test_pairs = [
    ("He loves playing football.", "He enjoys playing soccer."),
    ("The weather is sunny.", "The stock market crashed."),
]
print("手动推理（Ch1 四步：分词 → 模型 → softmax → 取标签）：")

for s1,s2 in test_pairs:
    enc=loaded_tok(s1,s2,truncation=True,return_tensors='pt').to(pick_device())
    with torch.no_grad():
        loghits=loaded_model(**enc).logits
    probs=torch.softmax(loghits,dim=-1)[0]
    idx=probs.argmax().item()
    print(f"  「{s1}」vs「{s2}」→ {labels[idx]}（{probs[idx]:.2f}）")


print("=" * 72)
print("  阶段 6：用 pipeline 把模型封装成一行调用（上线形态）")
print("=" * 72)

device_id = 0 if pick_device() in ("cuda", "mps") else -1
clf = pipeline("text-classification", model=SAVE_DIR,
               device=pick_device() if pick_device() == "mps" else device_id)
for s1, s2 in test_pairs:
    out = clf({"text": s1, "text_pair": s2})
    print(f"  「{s1[:25]}…」vs「{s2[:25]}…」→ {out}")
print("\n" + "=" * 72)
print("  ✅ 全生命周期跑完：预训练 → 分词 → 微调 → 评估 → 保存 → 重载 → 推理 → pipeline")
print("  这就是把一个通用大模型，变成解决你具体任务的产品的完整链路。")
print("=" * 72)
