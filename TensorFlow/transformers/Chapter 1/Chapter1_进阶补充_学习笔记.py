"""
================================================================================
 Chapter 1 · 进阶补充（补齐审计发现的缺口）—— 可直接运行
================================================================================
 Chapter 1 的整理已经很扎实，审计只找到两个“最底层实证”缺口，本文件补上：
   1) ★手搓单序列前向：tokenize → convert_tokens_to_ids → tensor([ids]) → model(ids)
      不走 tokenizer 一步到位、不加 batch，看最裸的“文字→id→logits”链路。
   2) ★padding 会污染 logits 的实证：同一句“单独跑” vs “放进补齐的 batch 里跑”，
      不给 attention_mask 时结果会变；给了 mask 才恢复一致。这解释了 mask 为什么必须有。

 另外说明(非本文件重点)：部署章节里 vLLM / TGI / llama.cpp 的“原生库/Docker 命令”原文
 有实操代码，但都依赖 GPU/Linux；Mac 上已在“优化推理部署_案例闯关.py”用 mlx-lm 等价平替。
 那属客观硬件限制，非知识缺口。

 直接运行：python3 Chapter1_进阶补充_学习笔记.py
 用 distilbert-sst2(情感分类，若已缓存则秒开)。
================================================================================
"""

import torch


def banner(t):
    print("\n" + "=" * 72 + f"\n {t}\n" + "=" * 72)


from transformers import AutoTokenizer, AutoModelForSequenceClassification

ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()


# ==============================================================================
# 1) ★手搓单序列前向：最裸的“文字 → id → logits”
# ==============================================================================
banner("1) 手搓单序列前向(不用 tokenizer 一步到位、不加 batch)")
sentence = "I love this movie"

# 第①步 tokenize：字符串 → 子词 token(注意：这里不自动加 [CLS]/[SEP])
tokens = tok.tokenize(sentence)
print("  ① tokenize      :", tokens)

# 第②步 convert_tokens_to_ids：token → 词表里的数字 id
ids = tok.convert_tokens_to_ids(tokens)
print("  ② tokens→ids    :", ids)

# 第③步 手动包成张量(注意 [ids] 外面加一层 → 变成 (1, seq_len) 的“单条 batch”)
input_ids = torch.tensor([ids])
print("  ③ tensor 形状   :", tuple(input_ids.shape))

# 第④步 直接喂给模型，拿 logits(这条最裸路径缺了 [CLS]/[SEP]，仅为演示底层机制)
with torch.no_grad():
    logits = model(input_ids).logits
print("  ④ model → logits:", logits.tolist())
print("  对比：平时 tok('文本', return_tensors='pt') 是把①②③和加特殊标记一步全包了。")


# ==============================================================================
# 2) ★padding 污染 logits 的实证：为什么必须有 attention_mask
# ==============================================================================
banner("2) padding 会改变 logits —— attention_mask 的必要性(实证)")

s_short = "great"                      # 短句
s_long = "this movie is absolutely wonderful and moving"   # 长句

# (A) 短句“单独”编码 + 前向 → 这是“正确答案”
a = tok(s_short, return_tensors="pt")
with torch.no_grad():
    logit_alone = model(a["input_ids"]).logits
print("  (A) 短句单独跑        logits =", [round(x, 4) for x in logit_alone[0].tolist()])

# (B) 把短句放进 batch，用 padding 补到和长句一样长；但故意“只喂 input_ids，不喂 mask”
b = tok([s_short, s_long], padding=True, return_tensors="pt")
with torch.no_grad():
    logit_nomask = model(b["input_ids"]).logits          # ← 不传 attention_mask
print("  (B) 补齐后不给 mask   logits =", [round(x, 4) for x in logit_nomask[0].tolist()],
      " ← 和 (A) 不一样了! 被 [PAD] 污染")

# (C) 同样的 batch，但把 attention_mask 一起喂进去 → 恢复正确
with torch.no_grad():
    logit_mask = model(b["input_ids"], attention_mask=b["attention_mask"]).logits
print("  (C) 补齐后给 mask     logits =", [round(x, 4) for x in logit_mask[0].tolist()],
      " ← 又和 (A) 一致了")

diff_nomask = (logit_alone - logit_nomask[0:1]).abs().max().item()
diff_mask = (logit_alone - logit_mask[0:1]).abs().max().item()
print(f"\n  与正确答案(A)的最大偏差：不给mask={diff_nomask:.4f}  给mask={diff_mask:.4f}")
print("  结论：padding 补的 [PAD] 若不被 mask 掉，会参与注意力计算、污染结果；")
print("        attention_mask=0 告诉模型“这些是填充，别理它”，所以补齐后必须配 mask。")

print("\n✅ 全部跑完。补齐了 Chapter 1 最底层的两处实证：手搓前向 + padding 污染。")
