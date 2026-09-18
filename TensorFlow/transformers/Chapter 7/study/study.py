import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

@dataclass
class TrainConfig:
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "fancyzhx/ag_news"
    label_names: list = field(default_factory=lambda: ["World", "Sports", "Business", "Sci/Tech"])
    max_length: int = 128
    batch_size: int = 16
    grad_accum: int = 2  # 有效 batch = batch_size * grad_accum = 32
    lr: float = 3e-5  # 微调大模型的经验区间 1e-5~5e-5
    epochs: int = 1
    warmup_ratio: float = 0.1  # 前 10% 步做学习率热身
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # 梯度裁剪阈值
    subset_train: int = 3000  # 快速验证用小子集；--full 用全量
    subset_eval: int = 800
    patience: int = 2  # 早停：宏F1 连续 N 轮不提升就停
    seed: int = 42
    output_dir: str = "ag_news-distilbert"

def compute_metrics(preds, labels, num_classes):
    preds,labels=np.asarray(preds),np.asarray(labels)
    acc=float((preds==labels).mean())
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s))}

def build_dataloaders(cfg, tokenizer):
    from datasets import load_dataset
    from transformers import DataCollatorWithPadding
    ds=load_dataset(cfg.dataset_name)
    train = ds["train"].shuffle(seed=cfg.seed)
    test = ds["test"].shuffle(seed=cfg.seed)
    if cfg.subset_train:
        train = train.select(range(min(cfg.subset_train, len(train))))
        test = test.select(range(min(cfg.subset_eval, len(test))))
    def tok_fn(batch):
        return tokenizer(batch['text'],truncation=True, max_length=cfg.max_length)

    train = train.map(tok_fn, batched=True, remove_columns=["text"])
    test = test.map(tok_fn, batched=True, remove_columns=["text"])
    train = train.rename_column("label", "labels")
    test = test.rename_column("label", "labels")
    train.set_format("torch")
    test.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return (DataLoader(train, shuffle=True, batch_size=cfg.batch_size, collate_fn=collator),
            DataLoader(test, batch_size=cfg.batch_size, collate_fn=collator))

def grouped_params(model, weight_decay):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm"]
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0}]

def train(cfg: TrainConfig):
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_scheduler, set_seed)
    from accelerate import Accelerator
    set_seed(cfg.seed)
    accelerator=Accelerator()
    tokenizer=AutoTokenizer.from_pretrained(cfg.model_name)
    id2label={i:n for n ,i in enumerate(cfg.label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(cfg.label_names),
        id2label=id2label, label2id={v: k for k, v in id2label.items()})
    train_loader, eval_loader = build_dataloaders(cfg, tokenizer)
    optimizer = AdamW(grouped_params(model, cfg.weight_decay), lr=cfg.lr)
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=int(total_steps * cfg.warmup_ratio),
                              num_training_steps=total_steps)
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler)

    def evaluate():
        model.eval()
        all_preds, all_labels = [], []
        for batch in eval_loader:
            with torch.no_grad():
                logits = model(**batch).logits
            preds = logits.argmax(-1)
            # 多卡时要 gather 汇总各卡结果(单卡也安全)
            all_preds.append(accelerator.gather_for_metrics(preds).cpu().numpy())
            all_labels.append(accelerator.gather_for_metrics(batch["labels"]).cpu().numpy())
        return compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels),
                               len(cfg.label_names))

    best_f1, no_improve = -1.0, 0
    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for i, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss / cfg.grad_accum  # 梯度累积：先除
            accelerator.backward(loss)
            running += loss.item() * cfg.grad_accum
            if (i + 1) % cfg.grad_accum == 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)  # 梯度裁剪
                optimizer.step();
                scheduler.step();
                optimizer.zero_grad()
                global_step += 1
                if global_step % 20 == 0:  # 结构化日志
                    accelerator.print(f"  epoch {epoch} step {global_step}/{total_steps} "
                                      f"loss={running / (i + 1):.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        metrics = evaluate()
        accelerator.print(f">>> epoch {epoch} 评估: acc={metrics['accuracy']:.4f} "
                          f"macro_f1={metrics['macro_f1']:.4f}")

        # 最优检查点 + 早停(以宏F1为准)
        if metrics["macro_f1"] > best_f1:
            best_f1, no_improve = metrics["macro_f1"], 0
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(cfg.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(cfg.output_dir)
            accelerator.print(f"    ✅ 新最优 macro_f1={best_f1:.4f}，已保存到 {cfg.output_dir}/")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                accelerator.print(f"    ⏹ 早停：macro_f1 连续 {cfg.patience} 轮没提升");
                break

    accelerator.print(f">>> 训练完成，最佳 macro_f1={best_f1:.4f}，模型在 {cfg.output_dir}/")
    return best_f1


def inference_demo(cfg: TrainConfig):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not os.path.isdir(cfg.output_dir):
        print("(还没训练出模型，跳过推理演示)");
        return
    tok = AutoTokenizer.from_pretrained(cfg.output_dir)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.output_dir).eval()
    samples = [
        "The stock market rallied today as tech shares surged.",
        "The team won the championship after a thrilling final match.",
        "Scientists discovered a new exoplanet using the space telescope.",
    ]
    enc = tok(samples, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        pred = model(**enc).logits.argmax(-1)
    print("\n>>> 推理演示(加载保存的模型)：")
    for s, p in zip(samples, pred.tolist()):
        print(f"   [{model.config.id2label[p]:8}] {s}")
