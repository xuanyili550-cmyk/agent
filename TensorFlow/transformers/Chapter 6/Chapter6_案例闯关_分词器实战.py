"""
================================================================================
 Chapter 6 案例闯关 · 分词器实战（6 关，全部能在 Mac 上跑）
================================================================================
 用法：
   python3 Chapter6_案例闯关_分词器实战.py          # 看菜单
   python3 Chapter6_案例闯关_分词器实战.py 1         # 只跑第 1 关
   python3 Chapter6_案例闯关_分词器实战.py all        # 全部
 也可在 PyCharm 直接点运行 → 按提示输入关号。

 关卡地图：
   1  手写 BPE          纯 Python，无需下载：统计词频→合并最高频对→编码新词
   2  手写 WordPiece    纯 Python：用“得分”选合并对 + 贪心最长前缀编码
   3  手写 Unigram      纯 Python：维特比 DP 找“概率最大”的切法
   4  快速分词器超能力   tokens()/word_ids()/offset 映射（下载 bert 分词器·很小）
   5  逐块搭 mini 分词器 用 tokenizers 库在本地小语料上训练一个 WordPiece
   6  offset 实战·NER    用真 offset 把预测标签贴回原文，并合并连续实体

 说明：
   · 第 1~3 关是纯算法，零依赖、瞬间跑完，最适合彻底看懂三大子词算法。
   · 第 4、6 关只下载“分词器文件”（几 MB），不下模型权重，很快。
   · 第 5 关需要 `pip install tokenizers`（装 transformers 时通常已自带）。
================================================================================
"""

import sys
from collections import defaultdict


def title(n, text):
    print("\n" + "=" * 72)
    print(f"  第 {n} 关：{text}")
    print("=" * 72)


# 三大算法共用的一个迷你语料（够小、跑得快，又能看出效果）
CORPUS = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]


def simple_pre_tokenize(text):
    """一个不依赖任何库的极简预分词：按空格切词、把标点单独拆出来。
    返回词列表。真实分词器规则更复杂(见学习笔记第 3 部分)，这里只为让算法自洽可跑。
    """
    words = []
    for chunk in text.split():
        cur = ""
        for ch in chunk:
            if ch.isalnum():
                cur += ch
            else:                       # 标点：先把前面攒的词吐出，再把标点单独成词
                if cur:
                    words.append(cur)
                    cur = ""
                words.append(ch)
        if cur:
            words.append(cur)
    return words


def build_word_freqs():
    """统计语料里每个“词”的出现次数（三关都要用）。"""
    word_freqs = defaultdict(int)
    for text in CORPUS:
        for w in simple_pre_tokenize(text):
            word_freqs[w] += 1
    return word_freqs


# ==============================================================================
# 第 1 关：手写 BPE —— 合并“出现次数最多”的相邻字符对
# ==============================================================================
def case1_bpe():
    title(1, "手写 BPE（合并最高频的相邻对）")
    word_freqs = build_word_freqs()

    # 步骤①：把每个词拆成单字符，作为初始切法
    splits = {word: list(word) for word in word_freqs}

    def compute_pair_freqs(splits):
        # 统计相邻字符对(a,b)在整个语料里“加权(乘词频)”出现多少次
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            s = splits[word]
            for i in range(len(s) - 1):
                pair_freqs[(s[i], s[i + 1])] += freq
        return pair_freqs

    def merge_pair(a, b, splits):
        # 把所有词里相邻的 (a,b) 合并成一个新 token "a+b"
        for word in word_freqs:
            s = splits[word]
            i = 0
            while i < len(s) - 1:
                if s[i] == a and s[i + 1] == b:
                    s = s[:i] + [a + b] + s[i + 2:]
                else:
                    i += 1
            splits[word] = s
        return splits

    # 步骤②：反复找“最高频对”并合并，记录合并规则，直到词表到目标大小
    vocab_size = 45
    alphabet = sorted({c for w in word_freqs for c in w})
    vocab = list(alphabet)
    merges = {}
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits)
        if not pair_freqs:
            break
        best = max(pair_freqs, key=pair_freqs.get)   # 频率最高的一对
        splits = merge_pair(*best, splits)
        merges[best] = best[0] + best[1]
        vocab.append(best[0] + best[1])

    print(f"初始字母表 {len(alphabet)} 个，训练到词表 {len(vocab)} 个")
    print("学到的前 8 条合并规则：")
    for i, (pair, tok) in enumerate(merges.items()):
        print(f"   {pair[0]!r} + {pair[1]!r}  ->  {tok!r}")
        if i >= 7:
            break

    # 步骤③：编码新词 = 拆字符后，按“学到的规则、原顺序”逐条套用
    def tokenize(text):
        out = []
        for word in simple_pre_tokenize(text):
            s = list(word)
            for pair, merge in merges.items():
                i = 0
                while i < len(s) - 1:
                    if s[i] == pair[0] and s[i + 1] == pair[1]:
                        s = s[:i] + [merge] + s[i + 2:]
                    else:
                        i += 1
            out += s
        return out

    demo = "This is not a token."
    print(f"\n给新句子分词：{demo!r}")
    print("  ->", tokenize(demo))
    print("要点：BPE 编码时严格按训练时学到的合并顺序，把认识的零件一点点拼大。")


# ==============================================================================
# 第 2 关：手写 WordPiece —— 用“得分”选合并对（BERT 的算法）
# ==============================================================================
def case2_wordpiece():
    title(2, "手写 WordPiece（得分 = 对频 /(左频×右频)）")
    word_freqs = build_word_freqs()

    # WordPiece：非首字符加 ## 前缀，表示“接在前一个子词后面”
    splits = {
        word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
        for word in word_freqs
    }

    def compute_pair_scores(splits):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            s = splits[word]
            if len(s) == 1:
                letter_freqs[s[0]] += freq
                continue
            for i in range(len(s) - 1):
                letter_freqs[s[i]] += freq
                pair_freqs[(s[i], s[i + 1])] += freq
            letter_freqs[s[-1]] += freq
        # 核心公式：分母压制“各自都常见”的对，偏向“几乎只成对出现”的对
        return {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }

    def merge_pair(a, b, splits):
        for word in word_freqs:
            s = splits[word]
            i = 0
            while i < len(s) - 1:
                if s[i] == a and s[i + 1] == b:
                    # 合并时去掉后一个子词的 ## 前缀
                    merged = a + b[2:] if b.startswith("##") else a + b
                    s = s[:i] + [merged] + s[i + 2:]
                else:
                    i += 1
            splits[word] = s
        return splits

    alphabet = set()
    for word in word_freqs:
        alphabet.add(word[0])
        for c in word[1:]:
            alphabet.add(f"##{c}")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(alphabet)

    vocab_size = 70
    while len(vocab) < vocab_size:
        scores = compute_pair_scores(splits)
        if not scores:
            break
        best = max(scores, key=scores.get)          # 得分最高的一对
        splits = merge_pair(*best, splits)
        new_tok = best[0] + best[1][2:] if best[1].startswith("##") else best[0] + best[1]
        vocab.append(new_tok)

    print(f"词表大小 {len(vocab)}（含 5 个特殊标记）")

    # 编码：从词首贪心找“词表里最长的前缀”，剩下的加 ## 继续；切不动 → [UNK]
    vocab_set = set(vocab)

    def encode_word(word):
        tokens = []
        while word:
            i = len(word)
            while i > 0 and word[:i] not in vocab_set:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if word:
                word = f"##{word}"
        return tokens

    print("对已知词 'Hugging' 编码 ->", encode_word("Hugging"))
    print("对生僻词 'HOgging' 编码 ->", encode_word("HOgging"), "（含未知字符→[UNK]）")

    def tokenize(text):
        out = []
        for w in simple_pre_tokenize(text):
            out += encode_word(w)
        return out

    print("\n给句子分词 'This is the Hugging Face course!'：")
    print("  ->", tokenize("This is the Hugging Face course!"))
    print("要点：WordPiece 编码是“词首开始的最长前缀匹配”，和 BPE 的合并顺序不同。")


# ==============================================================================
# 第 3 关：手写 Unigram —— 维特比 DP 找“概率最大”的切法
# ==============================================================================
def case3_unigram():
    title(3, "手写 Unigram（动态规划挑最优切法）")
    from math import log

    word_freqs = build_word_freqs()

    # 建一个候选词表：所有单字符 + 所有长度≥2 的子串，按频率给分（存 -log 概率）
    char_freqs = defaultdict(int)
    subwords_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word)):
            char_freqs[word[i]] += freq
            for j in range(i + 2, len(word) + 1):
                subwords_freqs[word[i:j]] += freq
    sorted_sub = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
    token_freqs = dict(list(char_freqs.items()) + sorted_sub[:200])
    total = sum(token_freqs.values())
    # 概率越大 → -log 越小 → 我们要找“总分(负对数和)最小”的切法
    model = {tok: -log(freq / total) for tok, freq in token_freqs.items()}
    print(f"候选词表 {len(model)} 个 token，每个 token 存 -log(概率) 作为代价")

    def encode_word(word, model):
        # best[i] = 把 word[:i] 切好的最优(分数, 上一切点)
        best = [{"start": 0, "score": 0.0}] + [
            {"start": None, "score": None} for _ in range(len(word))
        ]
        for start in range(len(word)):
            base = best[start]["score"]
            if base is None:
                continue
            for end in range(start + 1, len(word) + 1):
                tok = word[start:end]
                if tok in model:
                    score = base + model[tok]
                    if best[end]["score"] is None or best[end]["score"] > score:
                        best[end] = {"start": start, "score": score}
        # 从末尾回溯出切法
        seg = best[-1]
        if seg["score"] is None:
            return ["<unk>"], None
        tokens, start, end = [], seg["start"], len(word)
        while start is not None and start != 0:
            tokens.insert(0, word[start:end])
            end, start = start, best[start]["start"]
        tokens.insert(0, word[0:end])
        return tokens, seg["score"]

    for w in ["Hopefully", "This", "tokenization"]:
        toks, score = encode_word(w, model)
        print(f"  {w!r:16} -> {toks}   (代价={score:.2f})")
    print("要点：Unigram 允许一个词有多种切法，靠 DP 选“各子词概率乘积最大(负对数和最小)”的那种。")


# ==============================================================================
# 第 4 关：快速分词器的超能力（offset / word_ids）
# ==============================================================================
def case4_fast_abilities():
    title(4, "快速分词器超能力：tokens / word_ids / offset")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("需要 transformers：pip install transformers"); return

    tok = AutoTokenizer.from_pretrained("bert-base-cased")   # 只下分词器文件，很小
    example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    enc = tok(example)

    print("is_fast(是不是快速分词器) =", tok.is_fast)
    print("\ntokens() 切出来的 token：")
    print("  ", enc.tokens())
    print("\nword_ids() 每个 token 属于第几个单词（子词共享同一个单词号）：")
    print("  ", enc.word_ids())

    # word_to_chars：第 3 个单词是 'Sylvain' 被切成 S/##yl/##va/##in，看它在原文哪一段
    start, end = enc.word_to_chars(3)
    print(f"\n第 3 个单词在原文字符区间 = [{start},{end}) -> {example[start:end]!r}")

    # 直接拿 offset_mapping：每个 token 对应原文 [start,end)
    enc2 = tok(example, return_offsets_mapping=True)
    print("\noffset_mapping 前 8 个（(0,0) 是特殊标记 [CLS]）：")
    for t, off in list(zip(enc2.tokens(), enc2["offset_mapping"]))[:8]:
        print(f"   {t:10} {off}")
    print("要点：有了 offset，就能把任意 token 的预测“对回原文的字符位置”——NER/QA 全靠它。")


# ==============================================================================
# 第 5 关：用 tokenizers 库逐块搭一个 mini 分词器（本地小语料训练）
# ==============================================================================
def case5_build_from_scratch():
    title(5, "逐块搭建一个 WordPiece 分词器（本地训练）")
    try:
        from tokenizers import (
            Tokenizer, models, normalizers, pre_tokenizers, processors,
            decoders, trainers,
        )
    except ImportError:
        print("需要 tokenizers：pip install tokenizers"); return

    # 训练语料：直接用本文件顶部的小语料，重复几遍让频率更明显
    train_corpus = CORPUS * 50

    # ① 空壳：选 WordPiece 模型
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    # ② 规范化：NFD 分解 + 小写 + 去重音
    tk.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    # ③ 预分词：按空格 + 标点切
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    # ④ 训练器：给个小词表就够演示
    trainer = trainers.WordPieceTrainer(
        vocab_size=200,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tk.train_from_iterator(train_corpus, trainer=trainer)

    # ⑤ 后处理：单句套 [CLS] ... [SEP] 模板
    cls_id, sep_id = tk.token_to_id("[CLS]"), tk.token_to_id("[SEP]")
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    # ⑥ 解码器：把 ## 拼回去
    tk.decoder = decoders.WordPiece(prefix="##")

    enc = tk.encode("Let's test this tokenizer.")
    print("训练完成，词表大小 =", tk.get_vocab_size())
    print("tokens =", enc.tokens)
    print("ids    =", enc.ids)
    print("解码回文本 =", repr(tk.decode(enc.ids)))

    # 包成 HF 分词器才能配合 AutoModel 使用
    try:
        from transformers import PreTrainedTokenizerFast
        wrapped = PreTrainedTokenizerFast(
            tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
            cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
        )
        print("\n已用 PreTrainedTokenizerFast 包装，可直接 wrapped('文本') 使用：")
        print("  ", wrapped("Let's test this tokenizer.")["input_ids"])
    except ImportError:
        print("(未装 transformers，跳过 PreTrainedTokenizerFast 包装演示)")
    print("要点：normalizer→pre_tokenizer→model→post_processor→decoder 五块可自由拼装。")


# ==============================================================================
# 第 6 关：offset 实战 —— 把 NER 预测贴回原文并合并连续实体
# ==============================================================================
def case6_ner_grouping():
    title(6, "offset 实战：把 token 预测贴回原文 + 合并实体")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("需要 transformers：pip install transformers"); return
    import numpy as np

    # 用真实分词器拿到真实 offset；预测标签这里用“手写死”的，避免下载 1.3GB 大模型，
    # 但分组逻辑与真实 NER 后处理完全一致(见 new tokenizer.py 第 4 节)。
    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    enc = tok(example, return_offsets_mapping=True)
    tokens = enc.tokens()
    offsets = enc["offset_mapping"]

    # 构造一份“看起来像模型输出”的标签：给 Sylvain / Hugging Face / Brooklyn 打标
    def fake_label(tok_text):
        person = {"S", "##yl", "##va", "##in"}
        org = {"Hu", "##gging", "Face"}
        loc = {"Brooklyn"}
        if tok_text in person: return "I-PER"
        if tok_text in org:    return "I-ORG"
        if tok_text in loc:    return "I-LOC"
        return "O"

    labels = [fake_label(t) for t in tokens]
    scores = [0.99 if l != "O" else 1.0 for l in labels]   # 假装的置信度

    # —— 核心：把连续同类的 I-XXX 合并成一个实体，用 offset 还原原文子串 ——
    results, idx = [], 0
    while idx < len(labels):
        label = labels[idx]
        if label != "O":
            etype = label[2:]                 # 去掉 "I-" 前缀
            start, _ = offsets[idx]
            all_scores = []
            while idx < len(labels) and labels[idx] == f"I-{etype}":
                all_scores.append(scores[idx])
                _, end = offsets[idx]
                idx += 1
            results.append({
                "entity_group": etype,
                "score": float(np.mean(all_scores)),
                "word": example[start:end],   # ← offset 把碎 token 还原成完整单词
                "start": start, "end": end,
            })
        else:
            idx += 1

    print(f"原文：{example}\n")
    print("识别出的实体（连续同类已合并）：")
    for r in results:
        print(f"   {r['entity_group']:4} {r['word']!r:16} "
              f"位置[{r['start']},{r['end']})  score={r['score']:.3f}")
    print("\n要点：模型是按 token 预测的，靠 offset 才能把 'S/##yl/##va/##in' 拼回 'Sylvain'。")


# ==============================================================================
# 菜单调度
# ==============================================================================
CASES = {
    1: case1_bpe,
    2: case2_wordpiece,
    3: case3_unigram,
    4: case4_fast_abilities,
    5: case5_build_from_scratch,
    6: case6_ner_grouping,
}


def menu():
    print(__doc__)
    print("请输入关号(1-6) / all，然后回车：", end="")
    try:
        choice = input().strip()
    except EOFError:
        choice = "all"
    return choice


def run(choice):
    if choice == "all":
        for n in sorted(CASES):
            CASES[n]()
    elif choice.isdigit() and int(choice) in CASES:
        CASES[int(choice)]()
    else:
        print(f"没有第 {choice} 关，可选：{sorted(CASES)} 或 all")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else menu()
    run(arg)
