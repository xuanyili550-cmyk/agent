"""
================================================================================
 Chapter 6 · 分词器(Tokenizers) 全流程 —— 系统学习笔记（可直接运行）
================================================================================
 本笔记严格「按同目录三个原始文件」的内容与顺序来写，边跑边学：
     第一部分 ← pre-tokenization.py     规范化/预分词 + 手写 BPE + 手写 WordPiece
     第二部分 ← Unigram tokenization.py  手写 Unigram(完整训练) + 逐块搭三种分词器
     第三部分 ← new tokenizer.py         生成器语料/重训词表 + 快速分词器 + NER + QA
 直接运行即可一路看到 print 输出：
     python3 Chapter6_分词器_学习笔记.py
 只下载几个“分词器文件”(bert/gpt2/t5，都很小)，不下大模型权重；NER/QA 用写死的
 标签/logits 演示“后处理逻辑”，所以很快。原始三文件里那些要下大数据/大模型的重活，
 这里用小语料等价演示，并在注释里指回原文件。

 ★ 为什么要专门学分词器？模型只认「数字 id」，分词器就是「文字 ⇄ id」的翻译官。
   它切得好不好，直接决定：序列多长(显存/速度)、生僻词会不会变 [UNK]、
   以及 NER/QA 能不能把预测「对回原文的字符位置」。

 4 步流水线(整章总纲)：规范化 → 预分词 → 模型(切子词) → 后处理(加[CLS]/[SEP])
 三大算法一句话：BPE 合并“最高频”对；WordPiece 合并“最黏(得分最高)”对；
                Unigram 从大词表删“对损失影响最小”的 token。
================================================================================
"""

from collections import defaultdict
from math import log

from transformers import AutoTokenizer

# 第一、二部分共用的迷你语料（纯 Python 跑、几秒出结果，还能看出算法效果）
CORPUS = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]


def banner(text):
    print("\n" + "=" * 72 + f"\n {text}\n" + "=" * 72)


# ##############################################################################
# 第一部分 ← pre-tokenization.py
#   规范化 → 预分词(BERT/GPT-2/T5) → 三算法对比 → 手写 BPE → 手写 WordPiece
# ##############################################################################
banner("第一部分 ← pre-tokenization.py：规范化 / 预分词 / 手写 BPE / 手写 WordPiece")

# ------------------------------------------------------------------------------
# 1.1 规范化 Normalization（清洗：小写、去重音、Unicode 规整）
# ------------------------------------------------------------------------------
# backend_tokenizer 才是底层 Rust 快速分词器，能拿到 normalizer 组件。
tok_uncased = AutoTokenizer.from_pretrained("bert-base-uncased")
print("[1.1] 规范化 'Héllò hôw are ü?' ->",
      tok_uncased.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
# bert-uncased 会小写+去重音；bert-cased 保留大小写。模型和它的分词器是配套的，别混用。

# ------------------------------------------------------------------------------
# 1.2 预分词 Pre-tokenization（先粗切成“词”，并记住每个词的字符区间 (start,end)）
# ------------------------------------------------------------------------------
bert = AutoTokenizer.from_pretrained("bert-base-cased")
gpt2 = AutoTokenizer.from_pretrained("gpt2")
t5 = AutoTokenizer.from_pretrained("t5-small")
sample = "Hello, how are  you?"
print("\n[1.2] 三种流派对同一句的预分词：")
print("  BERT(按空格+标点切) :", bert.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sample))
print("  GPT-2(Ġ 表词前空格) :", gpt2.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sample))
print("  T5(▁ 表空格)        :", t5.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sample))

# ------------------------------------------------------------------------------
# 1.3 三大算法对比（原文件里的表，浓缩成一段）
# ------------------------------------------------------------------------------
# ┌──────────┬───────────────┬────────────────────┬────────────────────┐
# │          │ BPE(GPT-2)    │ WordPiece(BERT)    │ Unigram(T5/XLNet)  │
# │ 起点     │ 小词表(字符)  │ 小词表(字符,带##)  │ 超大词表           │
# │ 训练动作 │ 合并最高频对  │ 合并得分最高对     │ 删损失最小的 token │
# │ 依据     │ 对频          │ 对频/(左频×右频)   │ 语料总损失         │
# └──────────┴───────────────┴────────────────────┴────────────────────┘

# 统计词频（第 1.4/1.5/2.x 都要用）
def word_freqs_with(pretok):
    freqs = defaultdict(int)
    for text in CORPUS:
        for word, _ in pretok.pre_tokenize_str(text):
            freqs[word] += 1
    return freqs

# ------------------------------------------------------------------------------
# 1.4 ★手写 BPE —— 合并“出现次数最多”的相邻对
# ------------------------------------------------------------------------------
# 训练：①拆单字符 ②反复找最高频相邻对合并、记一条合并规则 ③到目标词表大小。
print("\n[1.4] 手写 BPE：")
bpe_freqs = word_freqs_with(gpt2.backend_tokenizer.pre_tokenizer)
bpe_splits = {w: list(w) for w in bpe_freqs}                # 每个词先拆成单字符


def bpe_pair_freqs(splits):
    pf = defaultdict(int)
    for word, freq in bpe_freqs.items():
        s = splits[word]
        for i in range(len(s) - 1):
            pf[(s[i], s[i + 1])] += freq                    # 相邻对频率(乘词频)
    return pf


def bpe_merge(a, b, splits):
    for word in bpe_freqs:
        s, i = splits[word], 0
        while i < len(s) - 1:
            if s[i] == a and s[i + 1] == b:
                s = s[:i] + [a + b] + s[i + 2:]             # 把 (a,b) 合并成 "ab"
            else:
                i += 1
        splits[word] = s
    return splits


bpe_vocab = sorted({c for w in bpe_freqs for c in w})
bpe_merges = {}
while len(bpe_vocab) < 45:
    pf = bpe_pair_freqs(bpe_splits)
    if not pf:
        break
    best = max(pf, key=pf.get)                              # ← BPE 核心：选最高频对
    bpe_splits = bpe_merge(*best, bpe_splits)
    bpe_merges[best] = best[0] + best[1]
    bpe_vocab.append(best[0] + best[1])
print("  前 6 条合并规则：", list(bpe_merges.items())[:6])


def bpe_tokenize(text):
    out = []
    for word, _ in gpt2.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text):
        s = list(word)
        for pair, merge in bpe_merges.items():              # 严格按“学到的顺序”套用
            i = 0
            while i < len(s) - 1:
                if s[i] == pair[0] and s[i + 1] == pair[1]:
                    s = s[:i] + [merge] + s[i + 2:]
                else:
                    i += 1
        out += s
    return out


print("  'This is not a token.' ->", bpe_tokenize("This is not a token."))

# ------------------------------------------------------------------------------
# 1.5 ★手写 WordPiece —— 合并“得分最高(最黏)”的对；子词带 ## 前缀
# ------------------------------------------------------------------------------
# 与 BPE 唯一区别是选哪对合并：score = 对频 /(左频 × 右频)。
# 为什么：分母压制“各自都常见”的对，偏向“几乎只成对出现”的 → 更像一个真词。
print("\n[1.5] 手写 WordPiece：")
wp_freqs = word_freqs_with(bert.backend_tokenizer.pre_tokenizer)
wp_splits = {w: [c if i == 0 else f"##{c}" for i, c in enumerate(w)] for w in wp_freqs}


def wp_scores(splits):
    lf, pf = defaultdict(int), defaultdict(int)
    for word, freq in wp_freqs.items():
        s = splits[word]
        if len(s) == 1:
            lf[s[0]] += freq
            continue
        for i in range(len(s) - 1):
            lf[s[i]] += freq
            pf[(s[i], s[i + 1])] += freq
        lf[s[-1]] += freq
    return {p: f / (lf[p[0]] * lf[p[1]]) for p, f in pf.items()}


def wp_merge(a, b, splits):
    for word in wp_freqs:
        s, i = splits[word], 0
        while i < len(s) - 1:
            if s[i] == a and s[i + 1] == b:
                merged = a + b[2:] if b.startswith("##") else a + b   # 合并去掉后词 ##
                s = s[:i] + [merged] + s[i + 2:]
            else:
                i += 1
        splits[word] = s
    return splits


wp_vocab = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
for w in wp_freqs:
    wp_vocab.add(w[0])
    wp_vocab.update(f"##{c}" for c in w[1:])
while len(wp_vocab) < 70:
    sc = wp_scores(wp_splits)
    if not sc:
        break
    best = max(sc, key=sc.get)                              # ← WordPiece 核心：选得分最高对
    wp_splits = wp_merge(*best, wp_splits)
    wp_vocab.add(best[0] + best[1][2:] if best[1].startswith("##") else best[0] + best[1])


def wp_encode_word(word):
    # 编码：从词首贪心找“词表里最长前缀”，剩下加 ## 继续；切不动→[UNK]
    tokens = []
    while word:
        i = len(word)
        while i > 0 and word[:i] not in wp_vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if word:
            word = f"##{word}"
    return tokens


print("  'Hugging' ->", wp_encode_word("Hugging"),
      "  'HOgging' ->", wp_encode_word("HOgging"), "(未知字符→[UNK])")


# ##############################################################################
# 第二部分 ← Unigram tokenization.py
#   Unigram 直觉 → 手写 Unigram(建模/encode/loss/scores/删词) → 逐块搭 3 种分词器
# ##############################################################################
banner("第二部分 ← Unigram tokenization.py：手写 Unigram 完整训练 + 逐块搭建")

# ------------------------------------------------------------------------------
# 2.1 直觉：一个词可有多种切法，Unigram 用“概率乘积”挑最可能的那种
# ------------------------------------------------------------------------------
# 例："pug" 可切成 ["p","u","g"] 或 ["pu","g"]，各子词概率相乘，取乘积大的切法。
# 存 -log(概率) 当“代价”：概率大→代价小；乘积最大 ⇔ 代价之和最小(相加比连乘数值稳)。

# ------------------------------------------------------------------------------
# 2.2 手写 Unigram：建候选词表 + encode_word(维特比 DP 找最优切法)
# ------------------------------------------------------------------------------
print("\n[2.2] 手写 Unigram —— 建模 + DP 编码：")
uni_freqs = word_freqs_with(t5.backend_tokenizer.pre_tokenizer)   # SentencePiece 风格(▁)
char_freqs, sub_freqs = defaultdict(int), defaultdict(int)
for word, freq in uni_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        for j in range(i + 2, len(word) + 1):
            sub_freqs[word[i:j]] += freq                   # 所有长度≥2 的子串当候选
sorted_sub = sorted(sub_freqs.items(), key=lambda x: x[1], reverse=True)
# 候选词表 = 所有单字符 + 最高频的若干子词
token_freqs = dict(list(char_freqs.items()) + sorted_sub[:200])
total = sum(token_freqs.values())
uni_model = {t: -log(f / total) for t, f in token_freqs.items()}


def uni_encode_word(word, model):
    # best[i] = 把 word[:i] 切好的 (最小代价, 上一切点)；从左到右填表(维特比)
    best = [{"start": 0, "score": 0.0}] + [{"start": None, "score": None}] * len(word)
    for start in range(len(word)):
        base = best[start]["score"]
        if base is None:
            continue
        for end in range(start + 1, len(word) + 1):
            t = word[start:end]
            if t in model:
                sc = base + model[t]
                if best[end]["score"] is None or best[end]["score"] > sc:
                    best[end] = {"start": start, "score": sc}
    seg = best[-1]
    if seg["score"] is None:
        return ["<unk>"], None
    tokens, start, end = [], seg["start"], len(word)
    while start is not None and start != 0:
        tokens.insert(0, word[start:end])
        end, start = start, best[start]["start"]
    tokens.insert(0, word[0:end])
    return tokens, seg["score"]


print("  '▁Hopefully' ->", uni_encode_word("▁Hopefully", uni_model)[0])
print("  为什么用 DP：从左到右记住“到每个位置的最优切法”，避免枚举所有切法(指数级)。")

# ------------------------------------------------------------------------------
# 2.3 手写 Unigram：compute_loss + compute_scores + 删词训练循环
# ------------------------------------------------------------------------------
# 训练核心：反复算“删掉某 token 后，整个语料损失涨多少”，删掉“删了也不心疼”的最低分。
print("\n[2.3] 手写 Unigram —— 损失/得分/删词训练：")


def uni_compute_loss(model):
    # 语料总损失 = Σ 词频 × 该词最优切法的代价。损失越小，模型越“贴合”语料。
    loss = 0.0
    for word, freq in uni_freqs.items():
        _, wl = uni_encode_word(word, model)
        loss += freq * wl
    return loss


def uni_compute_scores(model):
    # 每个 token 的得分 = 删掉它之后损失的“增加量”。得分越小 = 删了越不心疼。
    import copy
    scores, base = {}, uni_compute_loss(model)
    for token in model:
        if len(token) == 1:                # 单字符永远保留(否则有的词切不出来)
            continue
        m2 = copy.deepcopy(model)
        del m2[token]
        scores[token] = uni_compute_loss(m2) - base
    return scores


loss0 = uni_compute_loss(uni_model)
print(f"  初始语料损失 = {loss0:.1f}")
# 一轮“删 10% 最低分 token”的删词循环（原文件里会循环到目标词表大小，这里演示一轮）
scores = uni_compute_scores(uni_model)
sorted_scores = sorted(scores.items(), key=lambda x: x[1])   # 得分从低到高(最该删的在前)
to_remove = [t for t, _ in sorted_scores[:max(1, int(len(uni_model) * 0.1))]]
for t in to_remove:
    token_freqs.pop(t, None)
total = sum(token_freqs.values())
uni_model = {t: -log(f / total) for t, f in token_freqs.items()}
print(f"  删掉最低分的 {len(to_remove)} 个 token 后，词表 {len(uni_model)} 个；"
      f"新损失 = {uni_compute_loss(uni_model):.1f}(略升，属正常)")

# ------------------------------------------------------------------------------
# 2.4~2.6 逐块搭建：tokenizers 库的 6 大组件，搭 3 种真实分词器
# ------------------------------------------------------------------------------
# 组件：normalizer / pre_tokenizer / model / trainer / post_processor / decoder
from tokenizers import (Tokenizer, models, normalizers, pre_tokenizers,
                        processors, decoders, trainers, Regex)
from transformers import PreTrainedTokenizerFast

BIG = CORPUS * 50     # 重复几遍让频率更明显(真实里用大语料/生成器)

# ---- 2.4 WordPiece 分词器（BERT 式）----
print("\n[2.4] 逐块搭 WordPiece(BERT 式) 并训练：")
wp_tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
wp_tk.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
wp_tk.pre_tokenizer = pre_tokenizers.Whitespace()
wp_tk.train_from_iterator(BIG, trainer=trainers.WordPieceTrainer(
    vocab_size=200, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]))
cls, sep = wp_tk.token_to_id("[CLS]"), wp_tk.token_to_id("[SEP]")
wp_tk.post_processor = processors.TemplateProcessing(     # 后处理套 [CLS]..[SEP] 模板
    single="[CLS]:0 $A:0 [SEP]:0", pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls), ("[SEP]", sep)])
wp_tk.decoder = decoders.WordPiece(prefix="##")
print("  tokens =", wp_tk.encode("Let's test this tokenizer.").tokens)
# 包成 HF 分词器才能配合 AutoModel 用(特殊标记必须手动指定)：
wrapped = PreTrainedTokenizerFast(
    tokenizer_object=wp_tk, unk_token="[UNK]", pad_token="[PAD]",
    cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
print("  包装后可直接 wrapped('...')：", wrapped("Let's test this tokenizer.")["input_ids"])

# ---- 2.5 BPE 分词器（GPT-2 式：ByteLevel，无归一化器）----
print("\n[2.5] 逐块搭 BPE(GPT-2 式) 并训练：")
bpe_tk = Tokenizer(models.BPE())
bpe_tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe_tk.train_from_iterator(BIG, trainer=trainers.BpeTrainer(
    vocab_size=200, special_tokens=["<|endoftext|>"]))
bpe_tk.post_processor = processors.ByteLevel(trim_offsets=False)
bpe_tk.decoder = decoders.ByteLevel()
print("  tokens =", bpe_tk.encode("Let's test this tokenizer.").tokens)

# ---- 2.6 Unigram 分词器（XLNet 式：Metaspace + 一串归一化器）----
print("\n[2.6] 逐块搭 Unigram(XLNet 式) 并训练：")
uni_tk = Tokenizer(models.Unigram())
uni_tk.normalizer = normalizers.Sequence([
    normalizers.Replace("``", '"'), normalizers.Replace("''", '"'),
    normalizers.NFKD(), normalizers.StripAccents(),
    normalizers.Replace(Regex(" {2,}"), " ")])
uni_tk.pre_tokenizer = pre_tokenizers.Metaspace()
uni_tk.train_from_iterator(BIG, trainer=trainers.UnigramTrainer(
    vocab_size=200, special_tokens=["<cls>", "<sep>", "<unk>", "<pad>", "<mask>"],
    unk_token="<unk>"))
uni_tk.decoder = decoders.Metaspace()
print("  tokens =", uni_tk.encode("Let's test this tokenizer.").tokens)
print("  三套只是把 model/pre_tokenizer/decoder 换掉，搭法完全一致。")


# ##############################################################################
# 第三部分 ← new tokenizer.py
#   生成器语料/重训词表 → 快速分词器能力 → NER 手写后处理 → QA(掩码+打分+长文本)
# ##############################################################################
banner("第三部分 ← new tokenizer.py：语料生成器 / 快速分词器 / NER / QA")

# ------------------------------------------------------------------------------
# 3.1 构建语料库(生成器惰性加载) + train_new_from_iterator(在自己语料上重训词表)
# ------------------------------------------------------------------------------
print("\n[3.1] 生成器语料 + 重训词表：")
# 生成器：用到第 i 批时才取，处理完就丢 → 省内存。★但生成器“只能用一次”：
gen = (x for x in range(5))
print("  list(gen) 第一次：", list(gen), " 第二次：", list(gen), "(空! 已耗尽)")
# 所以要包成“返回生成器的函数”，每次调用都新建一个：
def get_training_corpus():
    for i in range(0, len(CORPUS), 2):
        yield CORPUS[i:i + 2]        # 真实里是 dataset[i:i+1000]["text"]
# 用老分词器的“算法结构”，在自己语料上重训一份新词表(这里用小语料/小词表，几秒)：
new_tok = gpt2.train_new_from_iterator(get_training_corpus(), vocab_size=300)
print("  gpt2 原分词:", gpt2.tokenize("def add(a, b): return a+b")[:8], "...")
print("  重训后分词:", new_tok.tokenize("def add(a, b): return a+b")[:8], "...")
# 真实场景：在 code_search_net 上重训，让代码的缩进/下划线命名切得更短更合理。
# train_new_from_iterator 只重学“词表”，不训练模型权重，跟微调是两码事。

# ------------------------------------------------------------------------------
# 3.2 快速分词器特殊能力：is_fast / tokens / word_ids / offset
# ------------------------------------------------------------------------------
print("\n[3.2] 快速分词器的对齐能力：")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
enc = bert(example, return_offsets_mapping=True)
print("  is_fast =", bert.is_fast)
print("  tokens[:9]   =", enc.tokens()[:9])
print("  word_ids[:9] =", enc.word_ids()[:9], "(S/##yl/##va/##in 都属于单词3)")
s, e = enc.word_to_chars(3)
print(f"  word_to_chars(3) = [{s},{e}) -> {example[s:e]!r}")
print("  前 6 个 offset：", enc["offset_mapping"][:6], "((0,0)=特殊标记 [CLS])")

# ------------------------------------------------------------------------------
# 3.3 NER：手写后处理(logits→概率→argmax→id2label→offset→合并实体)
# ------------------------------------------------------------------------------
# 真实里预测来自模型；这里写死标签演示后处理，分组算法与真实 NER 完全一致。
print("\n[3.3] NER：offset 定位 + 合并连续实体：")
import numpy as np

enc = bert(example, return_offsets_mapping=True)
tokens, offsets = enc.tokens(), enc["offset_mapping"]


def fake_label(t):
    if t in {"S", "##yl", "##va", "##in"}: return "I-PER"
    if t in {"Hu", "##gging", "Face"}:     return "I-ORG"
    if t == "Brooklyn":                    return "I-LOC"
    return "O"


labels = [fake_label(t) for t in tokens]
sc = [0.99 if l != "O" else 1.0 for l in labels]
results, idx = [], 0
while idx < len(labels):
    if labels[idx] != "O":
        etype = labels[idx][2:]                  # 去掉 "I-" 前缀
        start, _ = offsets[idx]
        buf = []
        while idx < len(labels) and labels[idx] == f"I-{etype}":   # 连续同类才合并
            buf.append(sc[idx])
            _, end = offsets[idx]
            idx += 1
        results.append({"type": etype, "word": example[start:end],  # offset 还原完整单词
                        "score": float(np.mean(buf))})
    else:
        idx += 1
for r in results:
    print(f"  {r['type']:4} {r['word']!r:16} score={r['score']:.3f}")

# ------------------------------------------------------------------------------
# 3.4 QA：掩码 + start×end 打分 + triu + offset（重点看“为什么这样算”）
# ------------------------------------------------------------------------------
print("\n[3.4] QA 打分逻辑：")
import torch

q, context = "Where do I work?", "I work at Hugging Face in Brooklyn."
enc = bert(q, context, return_offsets_mapping=True, return_tensors="pt")
seq_ids = enc.sequence_ids(0)                 # None=特殊标记, 0=问题, 1=上下文
n = enc["input_ids"].shape[1]
# ① 掩码：只允许答案落在 context(=1)。为什么：答案不可能在问题里或 [SEP]/[PAD] 上。
mask = torch.tensor([s != 1 for s in seq_ids])
mask[0] = False                               # 保留 [CLS](有些模型用它表示“答案不存在”)
start_logits, end_logits = torch.zeros(n), torch.zeros(n)   # 写死的假 logits(演示用)
ctx = [i for i, s in enumerate(seq_ids) if s == 1]
start_logits[ctx[3]] = 5.0                    # 让 'Hu' 当最可能起点
end_logits[ctx[5]] = 5.0                      # 让 'Face' 当最可能终点
# 为什么被掩码位置设 -10000：之后要 softmax，大负数 → 概率≈0，等于“禁止选这里”。
start_logits[mask] = -10000
end_logits[mask] = -10000
start_p, end_p = torch.softmax(start_logits, -1), torch.softmax(end_logits, -1)
# ② 打分 scores[i,j]=P(start=i)×P(end=j)。为什么相乘：假设起止独立的联合概率。
scores = start_p[:, None] * end_p[None, :]
# ③ 为什么 triu(上三角)：答案必须“先开始后结束”(i<=j)，下三角(j<i)非法要丢掉。
scores = torch.triu(scores)
flat = scores.argmax().item()
si, ei = flat // n, flat % n
print("  预测答案 token 区间 =", f"[{si},{ei}] ->",
      repr(bert.decode(enc["input_ids"][0][si:ei + 1])))

# ------------------------------------------------------------------------------
# 3.5 QA 长文本：滑动窗口 stride / overflow（context 太长放不下时）
# ------------------------------------------------------------------------------
# context 超过模型 max_length 时切成“重叠的块”，别把答案切断。
print("\n[3.5] 长文本滑窗(stride)：")
long_ctx = "This sentence is not too long but we are going to split it anyway."
chunks = bert(long_ctx, truncation=True, max_length=8, stride=2,
              return_overflowing_tokens=True)
for ids in chunks["input_ids"]:
    print("   ", bert.decode(ids))
print("  为什么重叠(stride)：避免答案正好被切在两块边界而两块都不完整。")
print("  overflow_to_sample_mapping 记录每块来自原来第几条样本(批量时对得上号)。")


# ##############################################################################
# 附：常见报错 & 面试速答（纯文字）
# ##############################################################################
# 报错/坑：
#   · 'BertTokenizer' 没有 backend_tokenizer → 加载到“慢速”分词器；用 use_fast=True(默认)。
#   · offset 全是 (0,0) → 特殊标记或慢速分词器；只有快速分词器才有真 offset。
#   · 生成器只能用一次 → 包成“返回生成器的函数”(见 3.1)。
#   · 中文/多语言大量 [UNK] → 分词器与语言不匹配；换多语言分词器或自己重训词表。
#   · tokenizer.json 无法直接给 AutoModel → 先 PreTrainedTokenizerFast 包一层(见 2.4)。
#
# 面试速答：
#   Q BPE/WordPiece/Unigram 区别？
#   A 都是子词。BPE 合并“最高频”对；WordPiece 合并“得分最高(最黏)”对；
#     Unigram 从大词表删“对损失影响最小”的 token。BERT=WordPiece、GPT-2=BPE、T5=Unigram。
#   Q 为什么不用整词或纯字符？
#   A 整词→词表爆炸且生僻词全 [UNK]；纯字符→序列太长丢语义。子词两头折中。
#   Q fast tokenizer 除了快还有什么用？
#   A 提供 offset/word_ids 对齐，NER 贴回原文、QA 还原答案子串，没它这两类任务没法做。
#   Q 领域词汇(医疗/代码)被切碎怎么办？
#   A train_new_from_iterator 或 tokenizers 库在领域语料上重训词表(见 3.1 / 2.4~2.6)。

print("\n✅ 全部跑完：第一部分↔pre-tokenization.py，第二部分↔Unigram tokenization.py，"
      "第三部分↔new tokenizer.py。")
