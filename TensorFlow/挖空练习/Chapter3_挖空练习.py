"""
================================================================================
 Chapter 3 挖空练习 · 使用预训练模型（fill-mask 完形填空）
================================================================================
 玩法：
   1) 把每个 ______ 换成正确代码（凭记忆）
   2) 运行：python3 Chapter3_挖空练习.py
   3) 没填的报 NameError；卡住 → 文件底部「答案区」
 目标：① 用 pipeline 做 fill-mask  ② 手动 fill-mask（分词→模型→找mask→softmax→topk）
================================================================================
"""
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

ckpt = "bert-base-uncased"

# ---------- Part A：pipeline 版 ----------
# 练习1：创建 fill-mask 的 pipeline（提示：pipeline("任务名", model=ckpt)）
fm = pipeline("fill-mask", model=ckpt)

# 练习2：动态取 mask 符号（别写死 [MASK]！）
#   提示：fm.tokenizer.??? —— 属性名叫什么？
mask = fm.tokenizer.mask_token

text = f"The capital of France is {mask}."
print("pipeline 结果（top1）：", fm(text)[0]["token_str"])   # 应为 paris


# ---------- Part B：手动版（结合 Ch1）----------
tokenizer = AutoTokenizer.from_pretrained(ckpt)
# 练习3：加载「掩码语言模型」头（提示：AutoModelForMaskedLM 的哪个方法？）
model =AutoModelForMaskedLM.from_pretrained(ckpt)
model.eval()

sentence = f"The weather today is very {tokenizer.mask_token}."

# 练习4：分词，返回 PyTorch 张量
inputs = tokenizer(sentence, return_tensors="pt")

# 练习5：找到 [MASK] 所在的位置下标
#   提示：在 input_ids 里找等于 tokenizer.mask_token_id 的位置
mask_pos = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

# 练习6：前向，拿 logits；推理不算梯度
with torch.no_grad():                       # 提示：推理用哪个上下文管理器？
    logits = model(**inputs).logits

# 练习7：取 mask 位置那一行 logits → softmax → top5
mask_logits = logits[0, mask_pos]
probs = torch.softmax(mask_logits, dim=-1)     # 提示：最后一维
top = torch.topk(probs, 5)

print("手动 fill-mask top5：")
for score, tid in zip(top.values, top.indices):
    print(f"  {tokenizer.decode([tid]):12s} {score.item():.3f}")


# ==============================================================================
#  答案区（卡住再看！）
# ------------------------------------------------------------------------------
#  1: pipeline("fill-mask", model=ckpt)
#  2: fm.tokenizer.mask_token
#  3: model = AutoModelForMaskedLM.from_pretrained(ckpt)
#  4: return_tensors="pt"
#  5: tokenizer.mask_token_id
#  6: with torch.no_grad():
#  7: dim=-1
# ==============================================================================
