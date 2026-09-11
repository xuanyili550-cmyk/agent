"""
================================================================================
 分章项目 · Ch6 进阶 · 手写三大子词算法（贴 HF Ch6 的硬核部分）
================================================================================
 Ch6_领域分词器.py 讲了“怎么用/重训分词器”；本文件补上 HF Ch6 最硬的部分——
 【从零手写三大子词算法】，看清 BPE/WordPiece/Unigram 到底怎么把词切成子词：
   ① 规范化 + 预分词 API：normalize_str / pre_tokenize_str（分词四步的前两步，用真 backend 演示）。
   ② 手写 BPE：统计相邻对频率 → 反复合并最高频对 → 得到合并规则 → 按规则编码（GPT/RoBERTa 用）。
   ③ 手写 WordPiece：合并打分 = 对频/(左频×右频)，子词加 ## 前缀 → 贪心最长前缀编码（BERT 用）。
   ④ 手写 Unigram：给一个带概率的子词表，用维特比 DP 找“整体概率最大”的切分（T5/XLNet 用）。
 全程纯 Python，无需下模型，秒出、带自检。完整章节材料见 ../../../Chapter 6/。
 跑：python3 Ch6_进阶_手写分词算法.py
================================================================================
"""
import math
from collections import defaultdict

CORPUS = [
    "hugging face course", "the tokenizer course", "tokenization is fun",
    "hugging face tokenizer", "learn tokenization course",
]


# ==============================================================================
# ① 规范化 + 预分词（分词四步的前两步，用真 backend 演示）
# ==============================================================================
def normalize_and_pretokenize():
    # print("=" * 70, "\n① 规范化 normalize_str + 预分词 pre_tokenize_str\n" + "=" * 70)
    from transformers import AutoTokenizer
    bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    text = "Héllo,  how aRE  U?"
    # 规范化：小写、去重音、清多余空格（BERT 的 normalizer）
    print("  原文        :", repr(text))
    print("  BERT 规范化 :", repr(bert.backend_tokenizer.normalizer.normalize_str(text)))
    # 预分词：按空格/标点粗切（不同模型符号不同：BERT 无前缀，GPT-2 用 Ġ 表空格）
    print("  BERT 预分词 :", [w for w, _ in bert.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are u?")])
    print("  GPT-2 预分词:", [w for w, _ in gpt2.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are u?")])
    # print("  Ġ = GPT-2 用来标记“词前有空格”；BERT 靠 ## 标记“词中续接”。")


def word_freqs(corpus):
    freqs = defaultdict(int)
    for line in corpus:
        for w in line.split():
            freqs[w] += 1
    return freqs


# ==============================================================================
# ② 手写 BPE：合并最高频相邻对
# ==============================================================================
def bpe_train(corpus, vocab_size=40):
    # print("\n" + "=" * 70, "\n② 手写 BPE(反复合并最高频相邻对)\n" + "=" * 70)
    freqs = word_freqs(corpus)
    splits = {w: list(w) for w in freqs}                       # 每个词先拆成字符
    alphabet = sorted({c for w in freqs for c in w})
    vocab = list(alphabet)
    merges = {}

    def pair_freqs():
        pf = defaultdict(int)
        for w, f in freqs.items():
            s = splits[w]
            for i in range(len(s) - 1):
                pf[(s[i], s[i + 1])] += f
        return pf

    while len(vocab) < vocab_size:
        pf = pair_freqs()
        if not pf:
            break
        best = max(pf, key=pf.get)                             # 最高频的相邻对
        merges[best] = best[0] + best[1]
        vocab.append(best[0] + best[1])
        for w in freqs:                                        # 把该对在所有词里合并
            s, i, out = splits[w], 0, []
            while i < len(s):
                if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                    out.append(s[i] + s[i + 1]); i += 2
                else:
                    out.append(s[i]); i += 1
            splits[w] = out

    def encode(word):                                          # 按学到的合并规则顺序切词
        s = list(word)
        for (a, b), merged in merges.items():
            i = 0
            while i < len(s) - 1:
                if s[i] == a and s[i + 1] == b:
                    s[i:i + 2] = [merged]
                else:
                    i += 1
        return s

    print(f"  学到 {len(merges)} 条合并规则，前5条: {list(merges.items())[:5]}")
    for w in ["tokenizer", "course"]:
        print(f"  BPE 编码 {w!r} → {encode(w)}")
    assert "".join(encode("tokenizer")) == "tokenizer"         # 子词拼回原词
    return encode


# ==============================================================================
# ③ 手写 WordPiece：合并打分 = 对频/(左频×右频)，子词加 ##
# ==============================================================================
def wordpiece_train(corpus, vocab_size=50):
    # print("\n" + "=" * 70, "\n③ 手写 WordPiece(打分合并 + ## 前缀，BERT 用)\n" + "=" * 70)
    freqs = word_freqs(corpus)
    # 首字符正常，其余字符加 ## 前缀（WordPiece 的标记方式）
    splits = {w: [c if i == 0 else "##" + c for i, c in enumerate(w)] for w in freqs}
    vocab = set(c for s in splits.values() for c in s)

    def scores():
        letter_f, pair_f = defaultdict(int), defaultdict(int)
        for w, f in freqs.items():
            s = splits[w]
            for c in s:
                letter_f[c] += f
            for i in range(len(s) - 1):
                pair_f[(s[i], s[i + 1])] += f
        # 打分：对频 / (左频 × 右频) —— 稀有字母组成的对得分高，避免高频字母乱合并
        return {p: pf / (letter_f[p[0]] * letter_f[p[1]]) for p, pf in pair_f.items()}

    while len(vocab) < vocab_size:
        sc = scores()
        if not sc:
            break
        best = max(sc, key=sc.get)
        merged = best[0] + best[1][2:] if best[1].startswith("##") else best[0] + best[1]
        vocab.add(merged)
        for w in freqs:
            s, i, out = splits[w], 0, []
            while i < len(s):
                if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                    out.append(merged); i += 2
                else:
                    out.append(s[i]); i += 1
            splits[w] = out

    def encode(word):                                          # 贪心：每步找“能匹配的最长前缀子词”
        tokens, i = [], 0
        while i < len(word):
            j = len(word)
            while j > i:
                sub = word[i:j] if i == 0 else "##" + word[i:j]
                if sub in vocab:
                    break
                j -= 1
            if j == i:
                return ["[UNK]"]
            tokens.append(word[i:j] if i == 0 else "##" + word[i:j]); i = j
        return tokens

    print(f"  词表大小={len(vocab)}")
    for w in ["tokenizer", "course"]:
        print(f"  WordPiece 编码 {w!r} → {encode(w)}")
    return encode


# ==============================================================================
# ④ 手写 Unigram：维特比 DP 找“整体概率最大”的切分
# ==============================================================================
def unigram_encode():
    # print("\n" + "=" * 70, "\n④ 手写 Unigram(维特比 DP 选最优切分，T5/XLNet 用)\n" + "=" * 70)
    # 给一个带频次的子词表（真实 Unigram 会用 EM 训练删词得到；这里直接给出便于看维特比）
    counts = {"to": 30, "ken": 20, "token": 8, "iz": 15, "ization": 6, "er": 25,
              "i": 40, "z": 5, "a": 30, "t": 50, "o": 40, "n": 45, "k": 12, "e": 55, "r": 40}
    total = sum(counts.values())
    scores = {tok: -math.log(c / total) for tok, c in counts.items()}   # 分数 = -log 概率(越小越好)

    def viterbi(word):
        n = len(word)
        best = [(0.0, None)] + [(math.inf, None)] * n           # best[i]=切到位置i的最优(总分,回溯)
        for end in range(1, n + 1):
            for start in range(end):
                sub = word[start:end]
                if sub in scores and best[start][0] + scores[sub] < best[end][0]:
                    best[end] = (best[start][0] + scores[sub], start)
        # 回溯出切分
        toks, i = [], n
        while i > 0:
            start = best[i][1]
            if start is None:
                return ["[UNK]"], math.inf
            toks.insert(0, word[start:i]); i = start
        return toks, best[n][0]

    for w in ["tokenization", "tokenizer"]:
        toks, loss = viterbi(w)
        print(f"  Unigram 切分 {w!r} → {toks}  (总-log概率={loss:.2f}，越小=整体越可能)")
    toks, _ = viterbi("tokenization")
    assert "".join(toks) == "tokenization"
    # print("  维特比在所有可能切分里挑“子词概率乘积最大(即 -log 和最小)”的那一种。")


if __name__ == "__main__":
    normalize_and_pretokenize()
    bpe_train(CORPUS)
    wordpiece_train(CORPUS)
    unigram_encode()
    print("\n✅ Ch6 进阶跑通：规范化/预分词 API → 手写 BPE → 手写 WordPiece → 手写 Unigram。")
    # print("面试：Q BPE/WordPiece/Unigram 训练与编码分别怎么做? Q 为什么 WordPiece 打分要除以左右频?"
          # " Q Unigram 为什么用维特比? (见 ../面试高频题库.py 三)")
