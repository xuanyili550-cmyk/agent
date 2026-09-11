"""
================================================================================
 中文版 · 情感分析（用中文模型，中文文本才准）
================================================================================
 为什么要单独的中文模型：英文模型(distilbert-sst2)喂中文会失准(它没学过中文)。
 这里用在中文点评数据上微调的 uer/roberta-base-finetuned-dianping-chinese，中文情感才靠谱。
 (对应英文版 ../分章项目/Ch1_情感分析服务.py)

 中文情感模型可选：
   uer/roberta-base-finetuned-dianping-chinese   大众点评评论微调(本文件用)
   uer/roberta-base-finetuned-jd-binary-chinese  京东评论二分类
   lxyuan/distilbert-...-sentiments-student       多语言(中英都行，标签更干净)
 跑：python3 中文_情感分析.py     (首次下模型 ~400MB)
================================================================================
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ckpt = "uer/roberta-base-finetuned-dianping-chinese"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()

texts = ["这个产品太棒了，我非常喜欢，强烈推荐！",
         "质量太差了，用了一次就坏，很失望，要退货。",
         "还行吧，一般般，没什么特别的。"]

# 中文分词 → 模型 → softmax(和英文流程一样，只是换了中文模型和中文分词器)
enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=-1)

# 该模型标签: 0=negative(1-3星) 1=positive(4-5星)，映射成中文
id2cn = {0: "负面", 1: "正面"}
for text, p in zip(texts, probs):
    i = int(p.argmax())
    print(f"  {text}\n    → {id2cn[i]} (置信度 {p[i]:.3f})")

print("\n✅ 中文情感分析跑通(用中文微调模型)。英文文本请用英文模型，别混用。")
# print("面试：Q 为什么不能用英文模型跑中文? A 模型只认它训练过的语言/词表，跨语言会失准，要用对应语种的模型。")
