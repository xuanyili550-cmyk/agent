"""
================================================================================
 Chapter 6 · 规范化 / 预分词 + 手写 BPE 和 WordPiece（原始学习文件·带详细注释）
================================================================================
 对应课程「Normalization and pre-tokenization / BPE / WordPiece」三节。
 这份文件的价值在于：不调库、纯 Python 手写两大子词算法，彻底看懂它们怎么工作。

 一段文本进入分词器要经过 4 步流水线：
   规范化(normalize) → 预分词(pre-tokenize) → 模型(model:切子词) → 后处理(post-process)
 本文件覆盖前两步的观察，以及“模型”这一步里 BPE / WordPiece 的完整手写实现。

 本文件脉络：
   1) 规范化：normalizer.normalize_str 看大小写/去重音等清洗
   2) 预分词：不同模型规则不同（BERT 按空格标点切；GPT-2 用 Ġ 表示空格；T5 用 ▁）
   3) 三大算法对比表（BPE / WordPiece / Unigram 的训练思路差异）
   4) ★手写 BPE：统计词频 → 建字母表 → 反复合并「最高频的相邻对」→ 得到合并规则
   5) ★手写 WordPiece：与 BPE 类似，但合并依据是「得分」而非「频率」，子词带 ## 前缀

 一句话记住三者区别：
   BPE       —— 合并「最常一起出现」的一对
   WordPiece —— 合并「最像一个词」的一对（频率/两元素频率乘积）
   Unigram   —— 反过来，从大词表里「删掉损失最小」的词（见 Unigram tokenization.py）
================================================================================
"""

# ------------------------------------------------------------------------------
# 1) 规范化(Normalization)：清洗文本（小写、去重音、Unicode 规整等）
# ------------------------------------------------------------------------------
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
#normalizer该对象的属性有tokenizer一个normalize_str()方法，我们可以使用该方法来查看规范化
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
#快速分词器如何执行预分词，我们可以使用对象属性pre_tokenize_str()的方法：pre_tokenizertokenizer
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
#使用的是 BERT 分词器，预分词过程包括根据空格和标点符号进行分割。
# 其他分词器在这个步骤中可能有不同的规则。例如，如果我们使用 GPT-2 分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
# [('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)),
#  ('?', (19, 20))]
#基于 SentencePiece 算法的 T5 分词器
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
# [('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))]
#
# 模型	生物物理	WordPiece	Unigram
# 训练	从少量词汇开始，学习合并词元的规则。	从少量词汇开始，学习合并词元的规则。
# 从庞大的词汇量开始，学习删除标记的规则。
# 训练步骤	合并与最常见词对对应的词元	根据词对出现的频率，合并得分最高的词对对应的词元，
# 优先合并每个单独词元出现频率较低的词对。	移除词汇表中所有能使整个语料库损失最小化的词元。
# 学习	合并规则和词汇表	仅仅是词汇	词汇表，每个词条都有相应的分数。
# 编码	将单词拆分成字符，并应用训练期间学习到的合并规则。	首先找到词汇表中从开头开始的最长子词，
# 然后对单词的其余部分执行相同的操作。	利用训练过程中学习到的分数，找到最可能的词元分割方式


# ------------------------------------------------------------------------------
# 4) ★手写 BPE（Byte-Pair Encoding，字节对编码）
#    思路：把词拆成单字符，然后反复把「出现次数最多的相邻字符对」合并成一个新 token，
#    每合并一次就记一条“合并规则”，直到词表达到目标大小。
# ------------------------------------------------------------------------------
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
from collections import defaultdict
word_freqs=defaultdict(int)
for text in corpus:
    words_with_offsets=tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_word=[word for word,offset in words_with_offsets]
    for word in new_word:
        word_freqs[word]+=1

print(word_freqs)
#下一步是计算基础词汇表，它由语料库中使用的所有字符组成：
alphabet=[]
for word in word_freqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()
print(alphabet)

#我们还会在该词汇表的开头添加模型使用的特殊标记。对于 GPT-2 来说，唯一的特殊标记是"<|endoftext|>"：
vocab = ["<|endoftext|>"] + alphabet.copy()
#将每个单词拆分成单个字符，才能开始训练
splits={word:[c for c in word] for word in word_freqs.keys()}
#写一个函数来计算每对元素的频率
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs
pair_freqs=compute_pair_freqs(splits)
for i ,key in enumerate(pair_freqs.keys()):
    print(f" {key} : {pair_freqs[key]} ")
    if i >=5:
        break

best_pair = ""
max_freq = None
for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq<freq:
        best_pair=pair
        max_freq=freq
print(best_pair,max_freq)

merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits
splits = merge_pair("Ġ", "t", splits)
print(splits["Ġtrained"])

#50个词为目标词汇量
vocab_size = 50
while  len (vocab) < vocab_size:
    pair_freqs=compute_pair_freqs(splits)
    best_pair=""
    max_freq=None
    for pair,freq in pair_freqs.items():
        if max_freq is None or max_freq<freq:
            best_pair=pair
            max_freq=freq
    splits=merge_pair(*best_pair,splits)
    merges[best_pair]=best_pair[0]+best_pair[1]
    vocab.append(best_pair[0]+best_pair[1])
print(merges)
print(vocab)
#要对新文本进行分词，我们先对其进行预分词，然后将其分割，最后应用所有已学习到的合并规则：
def tokenize(text):
    pre_tokenize_result = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split

    return sum(splits, [])
tokenize("This is not a token.")


# ------------------------------------------------------------------------------
# 5) ★手写 WordPiece（BERT 用的算法）
# ------------------------------------------------------------------------------
# 与 BPE 类似，WordPiece 也学「合并规则」，主要区别在于「选哪一对来合并」：
# BPE 选频率最高的一对；WordPiece 选“得分”最高的一对，公式为：
#   score = pair_freq / (freq_of_first_element × freq_of_second_element)
# 直觉：分母压制了“单看都很常见、但未必总黏在一起”的对，
#       更偏向合并「几乎只成对出现」的字符对 → 更像一个真正的词。
# 另外 WordPiece 的子词带 ## 前缀，表示“它接在前一个子词后面”（如 Hugging→Hu ##gging）。


#分词算法
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
from collections import defaultdict
word_freqs=defaultdict(int)
for text in corpus:
    words_with_offsets=tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_word=[word for word, offset in words_with_offsets]
    for word in new_word:
        word_freqs[word]+=1
word_freqs
#
# defaultdict(
#     int, {'This': 3, 'is': 2, 'the': 1, 'Hugging': 1, 'Face': 1, 'Course': 1, '.': 4, 'chapter': 1, 'about': 1,
#     'tokenization': 1, 'section': 1, 'shows': 1, 'several': 1, 'tokenizer': 1, 'algorithms': 1, 'Hopefully': 1,
#     ',': 1, 'you': 1, 'will': 1, 'be': 1, 'able': 1, 'to': 1, 'understand': 1, 'how': 1, 'they': 1, 'are': 1,
#     'trained': 1, 'and': 1, 'generate': 1, 'tokens': 1})

alphabet = []
for word in word_freqs.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word[1:]:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")

alphabet.sort()
alphabet
print(alphabet)
#
# ['##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k', '##l', '##m', '##n', '##o', '##p', '##r', '##s',
#  '##t', '##u', '##v', '##w', '##y', '##z', ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'g', 'h', 'i', 's', 't', 'u',
#  'w', 'y']

#开头添加模型使用的特殊标记。以 BERT 为例，它是以下列表["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]：
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
    for word in word_freqs.keys()
}

def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores

pair_scores = compute_pair_scores(splits)
for i, key in enumerate(pair_scores.keys()):
    print(f"{key}: {pair_scores[key]}")
    if i >= 5:
        break


best_pair = ""
max_score = None
for pair, score in pair_scores.items():
    if max_score is None or max_score < score:
        best_pair = pair
        max_score = score

print(best_pair, max_score)

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

splits = merge_pair("a", "##b", splits)
splits["about"]

vocab_size = 70
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)
#进行预分词，然后将其分割，最后对每个词应用分词算法。也就是说，
# 我们从第一个词的开头开始寻找最长的子词并将其分割，然后对第二部分重复此过程，
# 依此类推，直至该词的剩余部分以及文本中的后续词：
def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens
print(encode_word("Hugging"))
print(encode_word("HOgging"))

def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word) for word in pre_tokenized_text]
    return sum(encoded_words, [])
tokenize("This is the Hugging Face course!")