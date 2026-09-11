"""
================================================================================
 分章项目 · Ch2 · 文本分类微调（贴 HF 微调章：Trainer 高层 + 手写循环看内部）
================================================================================
 HF 微调章的核心，用可运行代码复现两条路(在真实数据 ag_news 上微调 distilbert)：
   ① 高层：Trainer API —— 一个 compute_metrics + Trainer.train()，工程细节它全包了(生产首选)。
   ② 底层：手写 PyTorch 训练循环 —— 看清 Trainer 内部到底在做什么(前向/反向/优化器步)。
   贯穿两条路的关键点：DataCollatorWithPadding【动态填充】——组 batch 时才补到本批最长(省算力)。
   更完整端到端见 ../综合项目/项目1；生产多卡见 ../生产架构/生产03。
 跑：python3 Ch2_文本分类微调.py     # 真训练小子集(Trainer + 手写各一遍)，出准确率，~2 分钟
================================================================================
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "distilbert-base-uncased"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
tok = AutoTokenizer.from_pretrained(CKPT)
collator = DataCollatorWithPadding(tokenizer=tok)   # ★动态填充：组 batch 时才补到本批最长


def load(split, n):
    """数据：pandas 直读 HF parquet(本机 load_dataset 兜底)。先只分词不 padding，交给 collator 动态补。"""
    df = pd.read_parquet(f"hf://datasets/fancyzhx/ag_news/data/{split}-00000-of-00001.parquet")
    df = df.sample(n=n, random_state=42)
    enc = tok(df["text"].tolist(), truncation=True, max_length=128)
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
             "labels": int(df["label"].iloc[i])} for i in range(len(df))]


def accuracy(preds, labels):
    return float((np.array(preds) == np.array(labels)).mean())


# ==============================================================================
# ① 高层：Trainer API（compute_metrics + Trainer.train()）
# ==============================================================================
def train_with_trainer(train_ds, test_ds):
    # print("=" * 70, "\n① Trainer API(高层，工程细节全包)\n" + "=" * 70)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=4)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"accuracy": accuracy(logits.argmax(-1), labels)}

    args = TrainingArguments(
        output_dir="/tmp/ch2-trainer", num_train_epochs=1,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=3e-5, logging_steps=20, report_to="none", disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
                      data_collator=collator, compute_metrics=compute_metrics)
    trainer.train()
    m = trainer.evaluate()
    print(f"  Trainer 评估准确率 = {m['eval_accuracy']:.3f}")


# ==============================================================================
# ② 底层：手写 PyTorch 训练循环（看清 Trainer 内部）
# ==============================================================================
def train_manual(train_ds, test_ds):
    # print("\n" + "=" * 70, "\n② 手写训练循环(前向→反向→优化器步，Trainer 内部就是这些)\n" + "=" * 70)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT, num_labels=4).to(DEV)
    train = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collator)
    test = DataLoader(test_ds, batch_size=32, collate_fn=collator)
    opt = AdamW(model.parameters(), lr=3e-5)

    model.train()
    for batch in train:
        batch = {k: v.to(DEV) for k, v in batch.items()}
        model(**batch).loss.backward()      # 反向：算梯度
        opt.step(); opt.zero_grad()          # 优化器步 + 清梯度

    model.eval(); preds, labels = [], []
    for batch in test:
        batch = {k: v.to(DEV) for k, v in batch.items()}
        with torch.no_grad():
            preds += model(**batch).logits.argmax(-1).cpu().tolist()
        labels += batch["labels"].cpu().tolist()
    print(f"  手写循环准确率 = {accuracy(preds, labels):.3f}  (设备={DEV})")


if __name__ == "__main__":
    print(f">>> 设备={DEV}  数据=ag_news(真实)\n")
    train_ds, test_ds = load("train", 1000), load("test", 400)
    train_with_trainer(train_ds, test_ds)
    train_manual(train_ds, test_ds)
    print("\n✅ Ch2 跑通：同一份数据，Trainer 高层 vs 手写循环，都靠动态填充省算力。")
    # print("面试：Q 动态填充比固定 max_length 好在哪? Q Trainer vs 自定义循环怎么选? Q compute_metrics 干嘛的?"
          # " (见 ../面试高频题库.py 二)")
