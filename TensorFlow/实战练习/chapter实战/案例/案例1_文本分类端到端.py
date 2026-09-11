"""
================================================================================
 案例1 · 文本分类端到端（整合 Ch1 + Ch2 + Ch6）—— 完整可跑，照着手写练熟
================================================================================
 目标：从原始数据到能用的分类器，把三章技术串成一条线。
 整合了哪几章、每步为什么用这个技术：
   [Ch6 分词器]  真实文本 → input_ids：模型只认数字，分词器是“文字⇄数字”的翻译官。
                 用“快速分词器”是因为它快、还能批量并行；truncation/padding 见下。
   [Ch2 数据管道] map 批量分词 + DataCollatorWithPadding 动态填充：
                 为什么动态填充——每个 batch 只补到“本批最长”，比补到全局最长省大量算力/显存。
   [Ch1 模型]    AutoModelForSequenceClassification：在基座上加个分类头，输出 logits。
   [Ch1 后处理]  softmax 把 logits 变概率、argmax 取类别 —— 这就是 pipeline 内部干的事。
   [Ch2 训练]    标准微调循环(优化器+反向+更新) + 评估(准确率)。

 直接运行：python3 案例1_文本分类端到端.py     # 真训练小子集，~1 分钟，出真实准确率
================================================================================
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, get_scheduler)
import pandas as pd


# ============================================================================
# [知识点·评估指标] 为什么光看 accuracy 不够？
#   准确率在“类别不均衡”时会骗人：99% 样本是负类，模型全猜负类也有 99% 准确率，
#   但正类一个都没抓到。所以生产/面试都要看 精确率(Precision)/召回率(Recall)/F1 + 混淆矩阵。
#   · 精确率 = 预测为该类里，真的是该类的比例 (查得准不准) = TP/(TP+FP)
#   · 召回率 = 真实该类里，被找出来的比例   (查得全不全) = TP/(TP+FN)
#   · F1     = 精确率与召回率的调和平均 (两者兼顾)      = 2PR/(P+R)
#   · 混淆矩阵 cm[真实][预测]：对角线是对的，非对角线告诉你“哪两类爱混”(诊断价值最高)。
#   下面纯 numpy 手写，不依赖 sklearn(便于看清定义)。
# ============================================================================
def classification_report(preds, labels, class_names):
    """手写混淆矩阵 + 每类 P/R/F1 + macro 平均。preds/labels 是等长的类别下标列表。"""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for p, y in zip(preds, labels):
        cm[y][p] += 1                                    # 行=真实类，列=预测类
    # print("\n   混淆矩阵 (行=真实, 列=预测):")
    print("       " + " ".join(f"{c[:6]:>7}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"   {c[:6]:>6} " + " ".join(f"{cm[i][j]:>7}" for j in range(n)))
    print(f"\n   {'类别':<10}{'精确率P':>9}{'召回率R':>9}{'F1':>9}{'支持数':>8}")
    f1s = []
    for i, c in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp                          # 别的类被误判成 i
        fn = cm[i, :].sum() - tp                          # i 被误判成别的类
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        print(f"   {c:<10}{p:>9.3f}{r:>9.3f}{f1:>9.3f}{cm[i].sum():>8}")
    print(f"   {'macro-F1(各类F1求平均，不被大类主导)':<10} = {np.mean(f1s):.3f}")


def load_hf_parquet(repo, path):
    """直接用 pandas 读 HuggingFace 上的 parquet，绕开 datasets 库。
    (本机 datasets 在 py3.14 上 load_dataset 会触发 dill 报错，pandas 直读最稳。)"""
    return pd.read_parquet(f"hf://datasets/{repo}/{path}")


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def main():
    device = pick_device()
    ckpt = "distilbert-base-uncased"
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    # [自检] 设备/基座/任务 配置回显（噪音，已静音）：device / distilbert-base-uncased / ag_news 4 类

    # ---- [Ch6] 分词器：把文本变成模型能吃的数字 ----
    # 为什么用快速分词器：Rust 实现、批量并行快；truncation 防超长报错，padding 对齐成矩形张量。
    tok = AutoTokenizer.from_pretrained(ckpt)

    # ---- [Ch2] 数据管道：加载真实数据 + 批量分词 ----
    df_train = load_hf_parquet("fancyzhx/ag_news", "data/train-00000-of-00001.parquet")
    df_test = load_hf_parquet("fancyzhx/ag_news", "data/test-00000-of-00001.parquet")

    def to_dataset(df, n):
        df = df.sample(n=n, random_state=42)           # pandas 随机取子集(练手够快)
        texts, labels_ = df["text"].tolist(), df["label"].tolist()
        # 整列批量分词(小子集足够快)。先不 padding：留给 DataCollator 动态补齐。
        enc = tok(texts, truncation=True, max_length=128)
        return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
                 "labels": labels_[i]} for i in range(len(texts))]

    train = to_dataset(df_train, 1500)
    test = to_dataset(df_test, 400)

    # ---- [知识点·类别不均衡] 训练前先看每类样本数，别等模型学偏了才发现 ----
    # 为什么重要：某类样本太少 → 模型倾向多数类、少数类召回率极低。
    # 常用解法(按代价从低到高)：① 给损失加“类别权重”(少数类权重大，见下)
    #   ② 过采样少数类/欠采样多数类 ③ WeightedRandomSampler 采样均衡 ④ 收集更多少数类数据。
    counts = np.bincount([d["labels"] for d in train], minlength=len(labels))
    # 逆频率权重：样本越少的类，权重越大 → 交叉熵里“答错少数类”被罚得更重
    class_weights = counts.sum() / (len(labels) * np.maximum(counts, 1))
    # [自检] 各类样本数/逆频率权重（噪音，已静音）：ag_news 较均衡，各类≈375，权重≈[1,1,1,1]
    #        不均衡时把 class_weights 传给 nn.CrossEntropyLoss(weight=...)

    # 为什么用 DataCollatorWithPadding：动态填充——每个 batch 只补到本批最长，省显存/算力。
    collator = DataCollatorWithPadding(tokenizer=tok)
    train_loader = DataLoader(train, shuffle=True, batch_size=16, collate_fn=collator)
    test_loader = DataLoader(test, batch_size=32, collate_fn=collator)

    # ---- [Ch1] 模型：基座 + 分类头 ----
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=len(labels),
        id2label=dict(enumerate(labels))).to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    steps = len(train_loader) * 2
    # [知识点·学习率 warmup] 前 10% 步用小学习率“热身”再升到目标：训练初期参数还很乱，
    # 一上来就大 lr 容易把预训练学到的东西冲垮(loss 飞/NaN)。linear 调度后半段再线性衰减，收尾更稳。
    sched = get_scheduler("linear", optimizer, num_warmup_steps=steps // 10,
                          num_training_steps=steps)

    # ---- [Ch2] 训练循环：前向→loss→反向→更新 ----
    # [知识点·为什么用交叉熵] 分类的目标是“预测分布逼近真实分布(one-hot)”，
    #   交叉熵 = -log(正确类的预测概率)：越把正确类的概率压低，惩罚越大(趋于无穷)，梯度信号强。
    #   而且它和 softmax 是绝配——log_softmax 的梯度形式极简(pred-target)，数值稳、优化好。
    #   模型内部 model(**batch).loss 已经用 logits+labels 算好交叉熵，无需自己写。
    # [知识点·早停/过拟合] 下面为演示每个 epoch 后在测试集看一眼指标：
    #   若“训练 loss 还在降、但验证指标不升反降”=过拟合信号 → 该早停(EarlyStopping)、
    #   加 weight_decay/dropout、或减小模型/加数据。生产会用独立验证集 + patience 计数。
    def quick_eval():
        model.eval()
        preds, ys = [], []
        for b in test_loader:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.no_grad():
                preds += model(**b).logits.argmax(-1).tolist()
            ys += b["labels"].tolist()
        model.train()
        return np.mean(np.array(preds) == np.array(ys)), preds, ys

    model.train()
    for epoch in range(2):
        running = 0.0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss              # 模型内部已算好交叉熵 loss
            loss.backward()                          # 反向传播算梯度
            optimizer.step(); sched.step(); optimizer.zero_grad()   # 更新→调 lr→清梯度
            running += loss.item()
            if i % 40 == 0:
                print(f"   epoch {epoch} step {i}/{len(train_loader)} loss={running/(i+1):.4f}")
        val_acc, _, _ = quick_eval()
        print(f"   >> epoch {epoch} 结束: 训练loss={running/len(train_loader):.4f}  验证准确率={val_acc:.3f}"
              f"   (盯住这两条——训练loss降但验证不升即过拟合)")

    # ---- [Ch1 后处理 + Ch2 评估] softmax→argmax，算准确率 + 完整指标 ----
    acc, preds, ys = quick_eval()
    print(f"\n>>> 测试准确率 = {acc:.3f}  (但准确率会骗人，看下面每类的 P/R/F1 + 混淆矩阵)")
    classification_report(preds, ys, labels)

    # ---- 推理演示：一条新文本走完整链路 ----
    text = "Apple unveiled a new chip that doubles AI performance."
    enc = tok(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, -1)[0]   # ← Ch1 后处理：logits→概率
    top = int(probs.argmax())
    print(f">>> 推理: {text!r}\n    → {labels[top]} (置信度 {probs[top]:.2f})")
    print("\n✅ 端到端跑通：分词(Ch6)→数据管道(Ch2)→模型(Ch1)→训练(Ch2)→softmax后处理(Ch1)。")


# ==============================================================================
# 面试题(文本分类，这个案例会被问什么)
# ==============================================================================
# Q1: 类别不均衡怎么处理？
# A : ①损失加类别权重(逆频率) ②过/欠采样、WeightedRandomSampler ③换评估指标(看 F1/PR 而非 acc)
#     ④阈值调整(不一定用 0.5) ⑤收集更多少数类数据。先看不均衡到什么程度再决定组合。
# Q2: 为什么不能只看准确率？精确率/召回率怎么取舍？
# A : 不均衡时准确率虚高。看场景：垃圾邮件宁可漏放(高精确率，别误杀正常邮件)；癌症筛查宁可错报
#     (高召回率，别漏诊)。要兼顾就看 F1；多类看 macro-F1(各类平权)还是 weighted-F1(按样本量)。
# Q3: 学习率 warmup 为什么有用？linear/cosine 调度怎么选？
# A : 初期参数乱，大 lr 易冲垮预训练权重；warmup 先小后大更稳。linear 简单够用；cosine 收尾更平滑,
#     大模型/长训练常用 cosine。都配前 5-10% warmup。
# Q4: 怎么判断过拟合？有哪些手段缓解？
# A : 训练 loss 降但验证 loss/指标回升、两者差距拉大。缓解：早停、weight_decay、dropout、数据增强/
#     加数据、减小模型、交叉验证。小数据尤其容易过拟合。
# Q5: 为什么分类用交叉熵而不是 MSE？
# A : 交叉熵直接优化“正确类概率”，配 softmax 梯度形式简洁(pred-target)、不易梯度消失；MSE 用在分类上
#     梯度小、收敛慢，且不符合“分布匹配”的概率解释。分类=交叉熵，回归=MSE。


if __name__ == "__main__":
    main()
