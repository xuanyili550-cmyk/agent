"""
================================================================================
 Chapter 2 案例闯关 · 微调实战（6 关，从数据处理到真·微调，全部能在 Mac 上跑）
================================================================================
 用法（在本文件所在目录的终端）：
   python3 Chapter2_案例闯关_微调实战.py         # 看菜单
   python3 Chapter2_案例闯关_微调实战.py 1        # 只跑第 1 关
   python3 Chapter2_案例闯关_微调实战.py all       # 全部
 也可以在 PyCharm 直接点运行 → 按提示输入关号。

 关卡地图（3~6 关会真的训练，已用小子集保证几十秒内跑完）：
   1  数据处理全流程     map + DataCollatorWithPadding + 看一个 batch 的形状
   2  TrainingArguments  构造训练参数并打印关键项（理解每个超参数）
   3  Trainer 微调       ★真跑：小子集上微调 BERT，看 loss 下降
   4  评估指标           predict + accuracy/f1，对比“瞎猜基线”
   5  手写训练循环       不用 Trainer，看清 forward→backward→step 五步
   6  微调前 vs 后对比    同几句话，微调前后预测怎么变（微调的意义）

 ★ Mac 提醒：全部用 mps 加速、绝不开 fp16、report_to='none' 不连 wandb。
================================================================================
"""

import sys
import torch


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def title(n, text):
    print("\n" + "=" * 72)
    print(f"  第 {n} 关：{text}")
    print("=" * 72)


CHECKPOINT = "bert-base-uncased"


# 公共：加载并分词 MRPC（各关按需调用；结果会被 datasets 缓存，不会每次重下）
def load_tokenized():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    raw = load_dataset("nyu-mll/glue", "mrpc")
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(example):
        # 句子对一起编码；不在此 padding（留给 DataCollator 动态填充）
        return tok(list(example["sentence1"]), list(example["sentence2"]), truncation=True)

    tokenized = raw.map(tokenize_function, batched=True)
    return raw, tok, tokenized


# ==============================================================================
# 第 1 关：数据处理全流程
# ==============================================================================
def case_01():
    title(1, "数据处理全流程：map + DataCollatorWithPadding + 看 batch")
    from transformers import DataCollatorWithPadding

    raw, tok, tokenized = load_tokenized()
    print("原始数据集结构：")
    print("  train 条数：", raw["train"].num_rows)
    print("  一条样本：", {k: raw["train"][0][k] for k in ["sentence1", "label"]})
    print("  分词后新增字段：", [c for c in tokenized["train"].column_names
                                 if c not in raw["train"].column_names])

    # 取前 8 条，去掉非模型字段，用 collator 动态填充成一个 batch
    collator = DataCollatorWithPadding(tokenizer=tok)
    samples = tokenized["train"][:8]
    samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
    print("\n填充前，每条 input_ids 长度（长短不一）：", [len(x) for x in samples["input_ids"]])
    batch = collator(samples)
    print("动态填充后 batch 各字段形状（补齐到本批最长）：")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("👉 token_type_ids 存在 → 句子对任务；补齐到本批最长而非512 → 省算力")


# ==============================================================================
# 第 2 关：TrainingArguments 参数详解
# ==============================================================================
def case_02():
    title(2, "TrainingArguments：训练超参数容器")
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir="test-trainer",     # 必填：保存模型/检查点的目录
        eval_strategy="epoch",         # 每个 epoch 评估一次
        num_train_epochs=3,            # 训练轮数
        per_device_train_batch_size=8, # 训练 batch 大小
        learning_rate=5e-5,            # 学习率
        weight_decay=0.01,             # 权重衰减（正则化）
        report_to="none",              # 不连 wandb 等外部追踪
    )
    print("关键超参数：")
    for name in ["output_dir", "num_train_epochs", "per_device_train_batch_size",
                 "learning_rate", "weight_decay", "eval_strategy", "fp16"]:
        print(f"  {name:32s} = {getattr(args, name)}")
    print("👉 fp16=False：Mac(mps) 不支持混合精度，保持关闭。")
    print("   只改这几个就能覆盖大多数微调需求，其余用默认即可。")


# ==============================================================================
# 第 3 关：用 Trainer 微调（★真跑，小子集）
# ==============================================================================
def case_03():
    title(3, "Trainer 微调：在小子集上真训练，看 loss 下降")
    from transformers import (TrainingArguments, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer)

    raw, tok, tokenized = load_tokenized()
    collator = DataCollatorWithPadding(tokenizer=tok)
    # 小子集：训练 300 条、验证 100 条，保证几十秒跑完（全量要十几分钟）
    small_train = tokenized["train"].select(range(300))
    small_eval = tokenized["validation"].select(range(100))

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
    args = TrainingArguments(
        output_dir="test-trainer-demo",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        eval_strategy="no",
        logging_steps=10,          # 每 10 步打印一次 loss
        report_to="none",
        use_cpu=False,             # 允许用 mps/gpu
    )
    trainer = Trainer(model, args, train_dataset=small_train, eval_dataset=small_eval,
                      data_collator=collator, processing_class=tok)
    print("开始微调（300 条 × 1 epoch，注意 loss 逐步下降）……")
    trainer.train()
    print("✅ 微调完成。上面 log 里的 loss 一路下降 = 模型正在学。")


# ==============================================================================
# 第 4 关：评估指标（predict + accuracy/f1）
# ==============================================================================
def case_04():
    title(4, "评估：微调后在验证集上算 accuracy/f1，对比瞎猜基线")
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
    print("predict 输出 logits 形状：", pred.predictions.shape, "（样本数 × 2 标签）")
    preds = np.argmax(pred.predictions, axis=-1)
    metric = evaluate.load("glue", "mrpc")
    result = metric.compute(predictions=preds, references=pred.label_ids)
    # 瞎猜基线：全预测多数类的准确率
    import collections
    majority = collections.Counter(pred.label_ids.tolist()).most_common(1)[0][1] / len(pred.label_ids)
    print(f"\n微调后： accuracy={result['accuracy']:.3f}  f1={result['f1']:.3f}")
    print(f"瞎猜基线（全押多数类）：accuracy={majority:.3f}")
    print("👉 只训了 500 条 1 轮就已超过基线；全量训练会更高（课程能到 0.85+）。")


# ==============================================================================
# 第 5 关：手写训练循环（看清底层五步）
# ==============================================================================
def case_05():
    title(5, "手写训练循环：forward→backward→step→scheduler→zero_grad")
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
    print("设备：", device)
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
    print("✅ 五步循环就是所有深度学习训练的核心，Trainer 只是帮你把它自动化。")


# ==============================================================================
# 第 6 关：微调前 vs 后对比（微调到底改变了什么）
# ==============================================================================
def case_06():
    title(6, "微调前 vs 后：同几句话，预测怎么变")
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, TrainingArguments, Trainer)
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    device = pick_device()

    # 几对测试句子（前两对同义=1，后一对不同义=0）
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
    print("微调中（500 条 × 2 epoch）……")
    trainer.train()
    after = predict(model)

    print("\n对比（真实标签：同义/同义/不同义）：")
    for (s1, s2), b, a in zip(pairs, before, after):
        print(f"  「{s1[:30]}…」vs「{s2[:30]}…」")
        print(f"      微调前: {b[0]}({b[1]:.2f})   →   微调后: {a[0]}({a[1]:.2f})")
    print("👉 微调前分类头随机、预测无意义；微调后学会了判断句子是否同义。")


# ------------------------------------------------------------------------------
# 调度器
# ------------------------------------------------------------------------------
CASES = {1: case_01, 2: case_02, 3: case_03, 4: case_04, 5: case_05, 6: case_06}

MENU = """\
用法：
  python3 Chapter2_案例闯关_微调实战.py <关号>   跑单关，如 1 / 3 / 6
  python3 Chapter2_案例闯关_微调实战.py all       跑全部

关卡列表（3~6 会真训练，已用小子集，几十秒完成）：
  1  数据处理全流程     2  TrainingArguments 详解   3  Trainer 微调(真跑)
  4  评估 accuracy/f1   5  手写训练循环             6  微调前后对比
"""


def run_choice(choice: str):
    choice = choice.strip().lower()
    if choice == "all":
        for n in sorted(CASES):
            CASES[n]()
    else:
        try:
            CASES[int(choice)]()
        except (ValueError, KeyError):
            print(f"没有第 {choice} 关，请输入 1~6 或 all。")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        run_choice(args[0])
    else:
        print(MENU)
        while True:
            choice = input("请输入关号（1~6，或 all 全部，回车/q 退出）：")
            if choice.strip().lower() in ("", "q", "quit", "exit"):
                print("已退出。")
                break
            run_choice(choice)
            print()
