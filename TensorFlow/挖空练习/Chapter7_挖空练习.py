"""
================================================================================
 Chapter 7 挖空练习 · NER 标签对齐（本章第 1 个任务的核心难点）
================================================================================
 玩法：
   1) 把每个 ______ 换成正确代码（凭记忆，别翻笔记）
   2) 运行：python3 Chapter7_挖空练习.py   （只下载 bert 分词器文件，几秒）
   3) 没填的报 NameError（告诉你漏哪行）；卡住 → 文件底部「答案区」
 目标：把“每个词一个”的 NER 标签，对齐到分词器切出来的“子词级 token”。
       这是 Token classification 任务能不能训起来的关键一步。

 规则回顾（务必记牢）：
   · 特殊标记([CLS]/[SEP]) 和 “子词的非首片” → -100（交叉熵忽略，不算 loss/不评估）
   · 一个词的第一个子词 → 用这个词本来的标签
   · 词内后续子词 → 若该词是 B-XXX(奇数,实体开头)，续片要 +1 变成 I-XXX
================================================================================
"""
from transformers import AutoTokenizer

label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
              'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# 练习1：加载 bert-base-cased 分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

words = ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."]
tags = [3, 0, 7, 0, 0, 0, 7, 0, 0]     # B-ORG O B-MISC O O O B-MISC O O

# 练习2：对“已经分好词的列表”分词——要告诉分词器 is_split_into_words=True
inputs = tokenizer(words, is_split_into_words=True)


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # 新词的第一个子词
            current_word = word_id
            # 练习3：特殊标记(word_id 为 None)填 -100，否则用该词的标签 labels[word_id]
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # 特殊标记
            new_labels.append(-100)
        else:
            # 同一个词的后续子词
            label = labels[word_id]
            # 练习4：若是 B-XXX(奇数)，改成 I-XXX（怎么判断奇数？怎么改？）
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels


# 练习5：拿到每个 token 属于第几个单词（快速分词器的哪个方法？）
word_ids = inputs.word_ids()
aligned = align_labels_with_tokens(tags, word_ids)

print("tokens :", inputs.tokens())
print("word_ids:", word_ids)
print("原词标签:", [label_names[t] for t in tags])
print("对齐后 :", [label_names[a] if a != -100 else "-100" for a in aligned])
print("\n自检：'lamb'→'la','##mb'，首片保留标签、续片是 -100；[CLS]/[SEP] 也是 -100。")


# ==============================================================================
#  答案区（卡住再看！）
# ------------------------------------------------------------------------------
#  1: AutoTokenizer.from_pretrained("bert-base-cased")
#  2: tokenizer(words, is_split_into_words=True)
#  3: label = -100 if word_id is None else labels[word_id]
#  4: if label % 2 == 1: label += 1        # 奇数=B-XXX，+1 变 I-XXX
#  5: inputs.word_ids()
# ==============================================================================
