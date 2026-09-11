"""
================================================================================
 Chapter 1 挖空练习 · pipeline 内部原理（分词 → 模型 → softmax → 标签）
================================================================================
 玩法：
   1) 把每个 ______ 换成你认为对的代码（凭记忆，别翻笔记）
   2) 全部填完后运行：python3 Chapter1_挖空练习.py
   3) 没填的地方会报 NameError（告诉你漏了哪行）；填错会报错或结果不对
   4) 实在卡住 → 翻到文件最底部「答案区」（先尽力想，别马上看）
 目标输出：两句话各自的情感标签 + 概率。
================================================================================
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ckpt = "distilbert-base-uncased-finetuned-sst-2-english"

# 练习1：加载分词器（提示：AutoTokenizer 的哪个方法？参数是 ckpt）
tokenizer = AutoTokenizer.from_pretrained(ckpt)

# 练习2：加载「带分类头」的模型（提示：AutoModelForSequenceClassification 的哪个方法？）
model = AutoModelForSequenceClassification.from_pretrained(ckpt)

texts = ["I love this movie!", "This is terrible."]

# 练习3：把 texts 编码成模型输入
#   要求：补齐(padding)、截断(truncation)、返回 PyTorch 张量
#   提示：tokenizer(texts, padding=?, truncation=?, return_tensors=?)
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

print("编码结果的三个字段：", list(inputs.keys()))   # 应该有 input_ids / attention_mask 等

# 练习4：把 inputs 喂给模型，拿到 logits
#   提示：model(**inputs) 的输出里，属性名叫什么？
outputs = model(**inputs)
logits = outputs.logits

# 练习5：logits → 概率（提示：哪个函数？在最后一维 dim=-1 上做）
probs = torch.nn.functional.softmax(logits, dim=-1)

# 练习6：取每行概率最大的下标（预测类别）
#   提示：probs.argmax(dim=?)
pred_ids = probs.argmax(dim=-1)

# 打印结果（这部分不用填）
for text, pid, prob in zip(texts, pred_ids, probs):
    label = model.config.id2label[pid.item()]
    print(f"  {text!r:30s} -> {label} ({prob.max().item():.4f})")


# ==============================================================================
# 自检：填对了应看到类似
#   I love this movie!            -> POSITIVE (0.99..)
#   This is terrible.             -> NEGATIVE (0.99..)
# ==============================================================================


# ==============================================================================
#  答案区（卡住再看，别偷看！先自己想）
# ------------------------------------------------------------------------------
#  1: AutoTokenizer.from_pretrained(ckpt)
#  2: AutoModelForSequenceClassification.from_pretrained(ckpt)
#  3: padding=True, truncation=True, return_tensors="pt"
#  4: logits = outputs.logits
#  5: torch.nn.functional.softmax(logits, dim=-1)
#  6: probs.argmax(dim=-1)
# ==============================================================================
