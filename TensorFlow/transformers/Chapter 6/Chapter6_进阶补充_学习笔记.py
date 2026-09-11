"""
================================================================================
 Chapter 6 · 进阶补充（补齐审计发现的缺口）—— 可直接运行
================================================================================
 这是对 Chapter6_分词器_学习笔记.py 的补充。之前那份把算法/offset/QA打分讲透了，
 但为了免下载，NER/QA 用的是“写死标签/假 logits”；本文件补上被略过的点：
   1) ★NER 真实模型端到端 + aggregation_strategy 四策略(simple/first/max/average)
   2) ★QA 长文本“多块推理→汇总选全局最佳答案”的完整流程(真实模型)
   3) 快速分词器的双向映射方法族(word/token/char 互转 4 个方法)
   4) truncation / padding 参数取值表
   5) overflow_to_sample_mapping 批量多句演示
   6) 保存复用：train_new 后 save_pretrained、tokenizer.save/from_file、本地文件训练
   7) 专用分词器类(BertTokenizerFast/GPT2TokenizerFast) 与 padding_side

 直接运行：python3 Chapter6_进阶补充_学习笔记.py
 第 1/2 节要下小模型(NER ~430MB、QA ~260MB，若已缓存则秒开)；下不动会自动跳过，不影响其余。
================================================================================
"""

import numpy as np


def banner(t):
    print("\n" + "=" * 72 + f"\n {t}\n" + "=" * 72)


# ==============================================================================
# 1) ★NER 真实模型端到端 + aggregation_strategy 四策略
# ==============================================================================
banner("1) NER 真实模型 pipeline + 四种聚合策略(实体得分怎么算)")
try:
    from transformers import (pipeline, AutoTokenizer,
                              AutoModelForTokenClassification)
    ck = "huggingface-course/bert-finetuned-ner"
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForTokenClassification.from_pretrained(ck)   # 加载一次，复用
    sentence = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    print("  句子：", sentence)
    # aggregation_strategy 决定“把子词得分合成实体得分”的算法，四种给出不同 score：
    for strat in ["simple", "first", "max", "average"]:
        ner = pipeline("token-classification", model=model, tokenizer=tok,
                       aggregation_strategy=strat)
        got = [(e["entity_group"], e["word"], round(float(e["score"]), 3)) for e in ner(sentence)]
        print(f"  [{strat:7}] {got}")
    print("  simple/average=子词得分平均；first=取首子词；max=取最高。多词实体上 average 按单词平均。")
except Exception as e:
    print("  (跳过：需要联网下 NER 模型)", type(e).__name__, e)


# ==============================================================================
# 2) ★QA 长文本：多块推理 → 汇总选全局最佳答案(真实模型)
# ==============================================================================
banner("2) QA 长文本端到端：stride 切块 → 每块跑模型 → 跨块选最优答案")
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    ck = "distilbert-base-cased-distilled-squad"
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForQuestionAnswering.from_pretrained(ck).eval()

    question = "Which deep learning libraries back Transformers?"
    long_context = ("Transformers provides thousands of pretrained models. " * 6 +
                    "Transformers is backed by Jax, PyTorch and TensorFlow. " +
                    "It is straightforward to train your models. " * 6)

    # 切成重叠块：max_length 限长、truncation="only_second"(只截 context)、stride 重叠
    inputs = tok(question, long_context, max_length=64, truncation="only_second",
                 stride=16, return_overflowing_tokens=True, return_offsets_mapping=True,
                 padding=True, return_tensors="pt")
    n_chunks = inputs["input_ids"].shape[0]
    offsets_all = inputs.pop("offset_mapping")
    inputs.pop("overflow_to_sample_mapping")
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # 在每个块里找最佳 (start,end)，再跨块比总分，选全局最优
    best = {"score": -1e9, "text": ""}
    for c in range(n_chunks):
        seq_ids = inputs.sequence_ids(c)
        offs = offsets_all[c]
        sl, el = out.start_logits[c], out.end_logits[c]
        for s in np.argsort(sl.numpy())[-5:]:            # 每块取 start/end 各前 5
            for e in np.argsort(el.numpy())[-5:]:
                if seq_ids[s] != 1 or seq_ids[e] != 1:   # 必须落在 context 里
                    continue
                if e < s or e - s > 30:                  # 非法/过长丢掉
                    continue
                score = (sl[s] + el[e]).item()
                if score > best["score"]:
                    best = {"score": score,
                            "text": long_context[int(offs[s][0]):int(offs[e][1])]}
    print(f"  长 context 切成 {n_chunks} 块；跨块选出的最佳答案 = {best['text']!r}")
    print("  要点：长文放不下→stride 切重叠块→每块各自打分→汇总取总分最高，别让答案被切断漏掉。")
except Exception as e:
    print("  (跳过：需要联网下 QA 模型)", type(e).__name__, e)


# ==============================================================================
# 3) 快速分词器的双向映射方法族(4 个互转方法)
# ==============================================================================
banner("3) word / token / char 三者互转(不止 word_to_chars 一个)")
from transformers import AutoTokenizer

btok = AutoTokenizer.from_pretrained("bert-base-cased")
ex = "Sylvain works here"
enc = btok(ex)
print("  tokens:", enc.tokens())
print("  word_to_chars(0) =", enc.word_to_chars(0), "-> 单词0在原文的字符区间")
print("  token_to_chars(1)=", enc.token_to_chars(1), "-> token1 在原文的字符区间")
print("  char_to_token(3) =", enc.char_to_token(3), "-> 原文第3个字符属于哪个 token")
print("  char_to_word(3)  =", enc.char_to_word(3), "-> 原文第3个字符属于哪个单词")
print("  word_ids()       =", enc.word_ids(), "-> 每个 token 属于第几个单词")


# ==============================================================================
# 4) truncation / padding 参数取值表
# ==============================================================================
banner("4) truncation / padding 参数速查")
print("""
  truncation(截断)：
    False           不截断(超长直接留着，后面可能报错)
    True            超长就截(单句/句子对都截)
    "only_first"    只截第一段(句子对里截问题那段)
    "only_second"   只截第二段(QA 里只截 context，别切问题)
  padding(填充)：
    False / "do_not_pad"   不补(默认)
    True / "longest"       补到“当前 batch 最长”(最常用、最省)
    "max_length"           补到 max_length 指定的固定长度
""")


# ==============================================================================
# 5) overflow_to_sample_mapping 批量多句
# ==============================================================================
banner("5) 批量多句切块：每块来自第几条原句")
sents = ["This sentence is not too long but we split it anyway.",
         "This one is shorter but still gets split."]
enc = btok(sents, truncation=True, max_length=8, stride=2,
           return_overflowing_tokens=True)
print("  2 句 → 切成", len(enc["input_ids"]), "个块")
print("  overflow_to_sample_mapping =", enc["overflow_to_sample_mapping"])
print("  (数字表示每块来自原来第几句；批量处理时靠它把块和原句对上号)")


# ==============================================================================
# 6) 保存复用：save_pretrained / tokenizer.save / from_file / 本地文件训练
# ==============================================================================
banner("6) 分词器的保存与复用")
import tempfile, os
work = tempfile.mkdtemp()

# 6.1 train_new_from_iterator 重训后，用 save_pretrained 存下来下次直接加载
gpt2 = AutoTokenizer.from_pretrained("gpt2")
new_tok = gpt2.train_new_from_iterator([["def f(): return 1"] * 4], vocab_size=300)
new_tok.save_pretrained(os.path.join(work, "my-tokenizer"))
print("  6.1 save_pretrained：重训的分词器已存 →", os.path.join(work, "my-tokenizer"))
print("      下次 AutoTokenizer.from_pretrained('my-tokenizer') 直接加载，不用重训。")

# 6.2 底层 tokenizers 库：tokenizer.save('x.json') / Tokenizer.from_file('x.json')
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.train_from_iterator(["hello world foo bar"] * 20,
                       trainer=trainers.WordPieceTrainer(vocab_size=50,
                                                         special_tokens=["[UNK]"]))
json_path = os.path.join(work, "tk.json")
tk.save(json_path)                                   # 存成单个 json
tk2 = Tokenizer.from_file(json_path)                 # 一行读回
print("  6.2 tokenizer.save / from_file：存成单文件 json 再读回，tokens =",
      tk2.encode("hello foo").tokens)

# 6.3 本地文本文件训练(与 train_from_iterator 并列的另一条路)
txt_path = os.path.join(work, "corpus.txt")
with open(txt_path, "w") as f:
    f.write("hello world\nfoo bar baz\n" * 20)
tk3 = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tk3.pre_tokenizer = pre_tokenizers.Whitespace()
tk3.train([txt_path], trainer=trainers.WordPieceTrainer(vocab_size=50,
                                                        special_tokens=["[UNK]"]))
print("  6.3 tokenizer.train(['corpus.txt'])：直接喂文本文件训练完成，词表",
      tk3.get_vocab_size(), "个")


# ==============================================================================
# 7) 专用分词器类 + padding_side
# ==============================================================================
banner("7) 专用类包装 + padding_side")
print("""
  · 包装成 HF 分词器有两种写法：
      通用：PreTrainedTokenizerFast(tokenizer_object=tk, unk_token=..., ...)  # 特殊标记全手动
      专用：BertTokenizerFast(tokenizer_object=tk)   # BERT 式，更省事
            GPT2TokenizerFast(tokenizer_object=tk)   # GPT-2 式
            XLNetTokenizerFast(tokenizer_object=tk)  # XLNet 式
    专用类知道自己该有哪些特殊标记，不用你一个个填。
  · padding_side：填充加在左边还是右边。
      多数模型右填充(默认)；但 XLNet 等、以及很多“生成类”场景用左填充(padding_side='left')，
      因为生成时要让真实 token 贴着右边、句尾对齐。
""")

print("✅ 全部跑完。这些补齐了 Chapter 6 之前为省下载而略过的“真实模型端到端 + 保存复用”等点。")
