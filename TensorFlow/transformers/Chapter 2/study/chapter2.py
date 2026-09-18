import sys
import torch
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

CHECKPOINT = "bert-base-uncased"

def load_tokenized():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    raw=load_dataset('nyu-mll/glue','mrpc')
    tok=AutoTokenizer.from_pretrained(CHECKPOINT)
    def tokenize_function(example):
        return tok(list(example['sentence1']),list(example['sentence2']),truncation=True)
    tokenized=raw.map(tokenize_function,batched=True)
    return raw, tok, tokenized
from transformers import DataCollatorWithPadding
raw, tok, tokenized=load_tokenized()
collator=DataCollatorWithPadding(tokenizer=tok)
samples = tokenized["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
batch=collator(samples)
from transformers import TrainingArguments
args=TrainingArguments(
    output_dir='test-trainer',
    eval_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none",
)
from transformers import (TrainingArguments, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer)

raw, tok, tokenized = load_tokenized()
collator = DataCollatorWithPadding(tokenizer=tok)
small_train = tokenized["train"].select(range(300))
small_eval = tokenized["validation"].select(range(100))
model=AutoModelForSequenceClassification.from_pretrained(CHECKPOINT,num_labels=2)
args = TrainingArguments(
    output_dir="test-trainer-demo",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    eval_strategy="no",
    logging_steps=10,  # 每 10 步打印一次 loss
    report_to="none",
    use_cpu=False,  # 允许用 mps/gpu
)
trainer = Trainer(model, args, train_dataset=small_train, eval_dataset=small_eval,
                  data_collator=collator, processing_class=tok)
trainer.train()

import numpy as np
import evaluate
from transformers import (TrainingArguments, AutoModelForSequenceClassification,
                            DataCollatorWithPadding, Trainer)

raw, tok, tokenized = load_tokenized()
collator = DataCollatorWithPadding(tokenizer=tok)
small_train = tokenized["train"].select(range(500))
small_eval = tokenized["validation"].select(range(200))

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
args = TrainingArguments("test-trainer-eval", num_train_epochs=1,
                            per_device_train_batch_size=8, eval_strategy="no",
                            logging_steps=20, report_to="none")
trainer = Trainer(model, args, train_dataset=small_train, eval_dataset=small_eval,
                    data_collator=collator, processing_class=tok)
trainer.train()

pred = trainer.predict(small_eval)
preds = np.argmax(pred.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
result = metric.compute(predictions=preds, references=pred.label_ids)
import collections
majority = collections.Counter(pred.label_ids.tolist()).most_common(1)[0][1] / len(pred.label_ids)

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (AutoModelForSequenceClassification,
                            DataCollatorWithPadding, get_scheduler)

raw, tok, tokenized = load_tokenized()
 # 手写循环需要的三步后处理：删列 / 改名 labels / 转 torch 格式
ds = tokenized.remove_columns(["sentence1", "sentence2", "idx"])
ds = ds.rename_column("label", "labels")
ds.set_format("torch")
collator = DataCollatorWithPadding(tokenizer=tok)

small_train = ds["train"].select(range(200))
train_dl = DataLoader(small_train, shuffle=True, batch_size=8, collate_fn=collator)

device = pick_device()
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
num_steps = len(train_dl)   # 1 个 epoch
scheduler = get_scheduler("linear", optimizer=optimizer,
                            num_warmup_steps=0, num_training_steps=num_steps)

model.train()
print(f"开始手写循环（200 条，共 {num_steps} 步）……")
for i, batch in enumerate(train_dl):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)      # ① 前向：有 labels 时模型自动算 loss
    loss = outputs.loss
    loss.backward()               # ② 反向：算梯度
    optimizer.step()              # ③ 更新权重
    scheduler.step()              # ④ 调整学习率
    optimizer.zero_grad()         # ⑤ 清空梯度（不清会累加）
    if i % 5 == 0:
        print(f"  step {i:2d}  loss={loss.item():.4f}")

import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained(CHECKPOINT)
device = pick_device()
pairs = [
    ("The company reported strong earnings.", "The firm announced solid profits."),
    ("He loves playing football.", "He enjoys playing soccer."),
    ("The weather is sunny today.", "The stock market crashed yesterday."),
]
labels = {0: "不同义", 1: "同义"}


def predict(model):
    model.eval()
    out = []
    for s1, s2 in pairs:
        enc = tok(s1, s2, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1)[0]
        idx = p.argmax().item()
        out.append((labels[idx], p[idx].item()))
    return out


# ---- 微调前：分类头是随机初始化的，预测基本靠猜 ----
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2).to(device)
before = predict(model)

# ---- 微调：小子集快速训练 ----
raw = load_dataset("nyu-mll/glue", "mrpc")
tokenized = raw.map(lambda e: tok(list(e["sentence1"]), list(e["sentence2"]), truncation=True),
                    batched=True)
collator = DataCollatorWithPadding(tokenizer=tok)
args = TrainingArguments("test-trainer-cmp", num_train_epochs=2,
                         per_device_train_batch_size=8, eval_strategy="no",
                         logging_steps=50, report_to="none")
trainer = Trainer(model, args, train_dataset=tokenized["train"].select(range(500)),
                  data_collator=collator, processing_class=tok)
trainer.train()
after = predict(model)


