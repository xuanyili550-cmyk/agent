"""
================================================================================
 Chapter 6 挖空练习 · 分词器：快速分词器的 offset + 实体合并
================================================================================
 玩法：
   1) 把每个 ______ 换成正确代码（凭记忆，别翻笔记）
   2) 运行：python3 Chapter6_挖空练习.py   （只下载 bert 分词器文件，几秒）
   3) 没填的报 NameError（告诉你漏哪行）；卡住 → 文件底部「答案区」
 目标：用快速分词器的 offset，把按 token 预测的实体“对回原文”并合并成完整单词。
       （这是 Ch6 最核心、也是 NER/QA 的看家本领）
================================================================================
"""
import numpy as np
from transformers import AutoTokenizer

ckpt = "bert-base-cased"
# 练习1：加载分词器（AutoTokenizer 的哪个方法？参数 ckpt）
tokenizer = AutoTokenizer.from_pretrained(ckpt)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."

# 练习2：编码，并要求返回 offset 映射（参数名？值填 True）
#   提示：只有“快速分词器”才有 offset，能把 token 对回原文字符位置
enc = tokenizer(example, return_offsets_mapping=True)

# 练习3：拿到 token 文本列表（方法名？）
tokens = enc.tokens()
# 练习4：拿到 offset 列表（它是 encoding 的哪个键？）
offsets = enc["offset_mapping"]

print("is_fast(必须为 True 才有 offset) =", tokenizer.is_fast)
print("tokens =", tokens)

# 模拟一份“模型预测标签”（真实里来自 NER 模型；这里写死，专注练 offset+合并逻辑）
def fake_label(t):
    if t in {"S", "##yl", "##va", "##in"}: return "I-PER"
    if t in {"Hu", "##gging", "Face"}:     return "I-ORG"
    if t == "Brooklyn":                    return "I-LOC"
    return "O"

labels = [fake_label(t) for t in tokens]
scores = [0.99 if l != "O" else 1.0 for l in labels]

# —— 把连续同类的 I-XXX 合并成一个实体，用 offset 还原原文子串 ——
results = []
idx = 0
while idx < len(labels):
    label = labels[idx]
    if label != "O":
        # 练习5：去掉 "I-"/"B-" 前缀，只留实体类型（字符串切片，从第几位开始？）
        etype = label[2:]
        # 练习6：取当前 token 在原文的“起始字符”（offsets[idx] 是 (start, end)，取哪个？）
        start, _ = offsets[idx]
        all_scores = []
        # 练习7：只要下一个 token 仍是同类 I-etype，就继续合并（循环条件里比较什么？）
        while idx < len(labels) and labels[idx] == f"I-{etype}":
            all_scores.append(scores[idx])
            _, end = offsets[idx]      # 不断把“结束字符”推到当前 token 末尾
            idx += 1
        results.append({
            "entity_group": etype,
            "score": float(np.mean(all_scores)),
            # 练习8：用 offset 把碎 token 还原成原文完整单词（对 example 做切片）
            "word": example[start:end],
            "start": start, "end": end,
        })
    else:
        idx += 1

print("\n识别出的实体（连续同类已合并）：")
for r in results:
    print(f"  {r['entity_group']:4} {r['word']!r:16} [{r['start']},{r['end']}) score={r['score']:.3f}")
print("\n自检：应得到 PER='Sylvain' / ORG='Hugging Face' / LOC='Brooklyn'")


# ==============================================================================
#  答案区（卡住再看！）
# ------------------------------------------------------------------------------
#  1: AutoTokenizer.from_pretrained(ckpt)
#  2: tokenizer(example, return_offsets_mapping=True)
#  3: enc.tokens()
#  4: enc["offset_mapping"]
#  5: label[2:]                      # 去掉 "I-" 两个字符
#  6: start, _ = offsets[idx]        # offset 是 (start, end)，取 start
#  7: labels[idx] == f"I-{etype}"    # 下一个仍是同类才继续合并
#  8: example[start:end]             # 用原文字符区间还原完整单词
# ==============================================================================
