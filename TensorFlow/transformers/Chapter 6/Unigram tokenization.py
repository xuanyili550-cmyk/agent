"""
================================================================================
 Chapter 6 · 手写 Unigram + 用 tokenizers 库从零逐块搭建分词器（原始学习文件）
================================================================================
 对应课程「Unigram tokenization / Building a tokenizer block by block」两节。

 本文件脉络：
   1) ★手写 Unigram：与 BPE/WordPiece 相反——先建「超大词表」，再用「损失」反复删词
        · encode_word 用「维特比式」动态规划，找一个词概率最大的切法
        · compute_loss/compute_scores 衡量“删掉某个 token 后，整个语料损失涨多少”
        · 每轮删掉“删了也不心疼(得分最低)”的 10% token，直到词表缩到目标大小
   2) 逐块搭建：Tokenizer = normalizer + pre_tokenizer + model + post_processor + decoder
        · 分别演示从零搭 WordPiece(BERT 式) / BPE(GPT-2 式) / Unigram(XLNet 式) 三套
        · 最后用 PreTrainedTokenizerFast 包一层，才能像普通 HF 分词器一样使用

 三大算法一句话对比（配合 pre-tokenization.py 里的 BPE/WordPiece）：
   BPE       从小词表出发，合并「最高频」的相邻对
   WordPiece 从小词表出发，合并「得分最高」的相邻对（BERT）
   Unigram   从大词表出发，删除「对损失影响最小」的 token（T5 / XLNet / 多语言模型）
================================================================================
"""

# ------------------------------------------------------------------------------
# 0) 直觉：一个词可能有多种切法，Unigram 用「概率乘积」挑最可能的那一种
# ------------------------------------------------------------------------------
#因此该概率就是每个词元概率的乘积。例如，单词“a”["p", "u", "g"]的分词"pug"概率为：
#P(["p","u","g"])=P("p")×P("u")×P("g")=210/5 ×210/36 ×210/20 =0.000389
#相比之下，分词["pu", "g"]具有以下概率：
#P(["pu","g"])=P("pu")×P("g")=210/5 ×210/20 =0.0022676
#实现 Unigram
#现在让我们用代码实现目前为止所学的所有内容。和 BPE 以及 WordPiece 一样，
# 这并不是 Unigram 算法的高效实现（恰恰相反），但它应该能帮助你更好地理解它。
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
from collections import defaultdict

word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

word_freqs

char_freqs = defaultdict(int)
subwords_freqs = defaultdict(int)
for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        # 遍历长度至少为 2 的子词f
        for j in range(i + 2, len(word) + 1):
            subwords_freqs[word[i:j]] += freq

# 按频率对子词进行排序
sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
sorted_subwords[:10]

token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
token_freqs = {token: freq for token, freq in token_freqs}

#我们计算所有频率的总和，将频率转换为概率。对于我们的模型，我们将存储概率的对数，
# 因为对数相加比小数相乘在数值上更稳定，这将简化模型损失的计算
from math import log

total_sum = sum([freq for token, freq in token_freqs.items()])
model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

#我们从末尾开始，从一个起始位置跳到下一个起始位置，同时记录标记，直到到达单词的开头：

def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        # 这应该由循环的前几个步骤正确填充
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # 如果我们找到了一个更好的以 end_idx 结尾的分割，则更新
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # 我们未找到单词的分词 -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score

print(encode_word("Hopefully", model))
print(encode_word("This", model))

#计算模型在语料库上的损失
def compute_loss(model):
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss
#计算每个标记的得分也不难；我们只需要计算删除每个标记后得到的模型的损失即可：

import copy
def compute_scores(model):
    scores = {}
    model_loss = compute_loss(model)
    for token, score in model.items():
        # 我们始终保留长度为 1 的 token
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token) - model_loss
    return scores
scores = compute_scores(model)
print(scores["ll"])
print(scores["his"])

#需要做的就是将模型使用的特殊标记添加到词汇表中，然后循环直到我们从词汇表中删减出足够的标记，达到我们所需的大小
percent_to_remove = 0.1
while len(model) > 100:
    scores = compute_scores(model)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    # 移除 percent_to_remove 个得分最低的token。
    for i in range(int(len(model) * percent_to_remove)):
        _ = token_freqs.pop(sorted_scores[i][0])

    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

#要对一些文本进行分词，我们只需要应用预分词，然后使用我们的encode_word()函数：

def tokenize(text, model):
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in words_with_offsets]
    encoded_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    return sum(encoded_words, [])


tokenize("This is the Hugging Face course.", model)

# ------------------------------------------------------------------------------
# 2) 逐块构建分词器（tokenizers 库）
# ------------------------------------------------------------------------------
#Tokenizer类构建的，其构建模块被重新分组到子模块中：

# normalizers包含您可以使用的所有可能类型Normalizer
# pre_tokenizers包含您可以使用的所有可能类型PreTokenizer
# modelsModel包含您可以使用的各种类型，例如BPE，WordPiece和Unigram
# trainersTrainer包含可用于在语料库上训练模型的所有不同类型的
# post_processors包含您可以使用的各种类型PostProcessor
# decodersDecoder包含可用于解码分词输出的各种类型

#获取语料库
from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

#分词器也可以直接使用文本文件进行训练。以下是如何生成一个包含
# WikiText-2 所有文本/输入的文本文件，以便在本地使用：
with open("wikitext-2.txt", "w", encoding="utf-8") as f:
    for i in range(len(dataset)):
        f.write(dataset[i]["text"] + "\n")

#从零开始构建 WordPiece 分词器
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
#该库提供了一个Lowercase归一化器和一个StripAccents归一化器，你可以使用以下方式组合多个归一化器Sequence：
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
#BertPreTokenizer我们可以使用预先构建好的模板
tokenizer.pre_tokenizer=pre_tokenizers.BertPreTokenizer()
tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
#Whitespace预分词器会根据空格和所有非字母、数字或下划线字符进行分割，
# 因此从技术上讲，它会根据空格和标点符号进行分割
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my presegmenter.")
pre_tokenizer=pre_tokenizers.WhitespaceSplit()
pre_tokenizer.pre_tokenize_str("Let's test my presegmenter.")
#可以使用 aSequence来组合多个预分词器
pre_tokenizer=pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(),pre_tokenizers.Punctuation()]
)
pre_tokenizer.pre_tokenize_str("Let's test my presegmenter.")
special_tokens = [ "[UNK]" , "[PAD]" , "[CLS]" , "[SEP]" , "[MASK] "]
trainer=trainers.WordPieceTrainer(vocab_size=25000,special_tokens=special_tokens)
tokenizer.train_from_iterator(get_training_corpus(),trainer=trainer)
tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
#['let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.']

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)
#经典的 BERT 模板定义
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)
tokenizer.decoder = decoders.WordPiece(prefix="##")
tokenizer.decode(encoding.ids)
#Tokenizer然后我们可以使用以下方法将该文件重新加载到对象中from_file()
tokenizer.save("tokenizer.json")
new_tokenizer = Tokenizer.from_file("tokenizer.json")

#分词器封装到 `<T>` 中PreTrainedTokenizerFast，我们可以将构建好的分词器作为
# `<T>` 传递tokenizer_object，也可以将保存的分词器文件作为 `<T>`
# 传递tokenizer_file。需要记住的关键一点是，我们必须手动设置所有特殊标记，
# 因为该类无法从tokenizer对象中推断出哪个标记是掩码标记、哪个[CLS]标记等等

from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json",
    # 您可以从分词器文件加载，或者
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
#是特定的分词器类（例如BertTokenizerFast）
from transformers import BertTokenizerFast
wrapped_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)

#从零开始构建 BPE 代币化器
tokenizer = Tokenizer(models.BPE())
#GPT-2 不使用归一化器，因此我们跳过该步骤，直接进行预分词
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!")
#接下来是模型，它需要进行训练。对于 GPT-2 来说，唯一特殊的标记是文本结束标记
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.model = models.BPE()
tokenizer.train(["wikitext-2.txt"], trainer=trainer)
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
#该trim_offsets = False选项指示后处理器，我们应该保留以“Ġ”开头的词元的偏移量不变：
# 这样，偏移量的起始位置将指向单词前的空格，而不是单词的第一个字符
# （因为空格在技术上也是词元的一部分）。让我们看一下刚刚编码的文本的结果，其中'Ġtest'索引为 4 的词元是
sentence = "Let's test this tokenizer."
encoding = tokenizer.encode(sentence)
start, end = encoding.offsets[4]
sentence[start:end]

tokenizer.decoder = decoders.ByteLevel()
tokenizer.decode(encoding.ids)

from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)
#or
from transformers import GPT2TokenizerFast

wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)

tokenizer = Tokenizer(models.Unigram())
from tokenizers import Regex

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)
#任何 SentencePiece 分词器都要使用的预分词器是Metaspace：
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

tokenizer.pre_tokenizer.pre_tokenize_str("Let's test the pre-tokenizer!")

#XLNet 包含相当多的特殊标记
special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(
    vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.model = models.Unigram()
tokenizer.train(["wikitext-2.txt"], trainer=trainer)

encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)

cls_token_id = tokenizer.token_to_id("<cls>")
sep_token_id = tokenizer.token_to_id("<sep>")
print(cls_token_id, sep_token_id)

tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
)

encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences!")
print(encoding.tokens)
print(encoding.type_ids)

tokenizer.decoder = decoders.Metaspace()

from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
    padding_side="left",
)

from transformers import XLNetTokenizerFast

wrapped_tokenizer = XLNetTokenizerFast(tokenizer_object=tokenizer)