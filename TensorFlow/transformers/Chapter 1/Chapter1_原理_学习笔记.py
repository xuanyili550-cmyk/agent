"""
================================================================================
 Behind the Pipeline —— pipeline 内部原理系统学习笔记
================================================================================
 配套 HuggingFace 课程「Chapter 1」章节。
 一句话主线：pipeline("sentiment-analysis") 看似一行，内部其实是三步流水线：

        原始文本  ──►  ①分词器(Tokenizer)  ──►  ②模型(Model)  ──►  ③后处理
        "I love it"     变成 input_ids 等张量      输出 logits       softmax→概率→标签

 本文件结构：
   第 1 部分  分词器：从文本到数字（input_ids / token_type_ids / attention_mask）
   第 2 部分  三个关键操作：填充 padding / 截断 truncation / 特殊标记
   第 3 部分  分词器的底层三步：tokenize → convert_tokens_to_ids → decode
   第 4 部分  批处理与 attention_mask 为什么至关重要
   第 5 部分  模型：AutoModel vs AutoModelForSequenceClassification，logits 与 softmax
   第 6 部分  保存与加载 save_pretrained / from_pretrained
   第 7 部分  把三步串起来 = 完整复现一个 pipeline

 术语速记：
   checkpoint（检查点）  一个训练好的模型的名字/路径，如 "bert-base-cased"
   token（词元）         文本被切成的最小单位；BERT 用的是 subword（子词）
   input_ids            每个 token 在词表里的编号（整数）
   logits               模型最后一层的原始分数，未归一化；softmax 后才是概率
================================================================================
"""

from transformers import (
    pipeline,
    AutoTokenizer,
    BertTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    BertModel,
)
import torch


# ==============================================================================
# 第 1 部分：分词器 —— 从文本到数字
# ==============================================================================
# 模型不认识文字，只认识数字。分词器(Tokenizer)负责把字符串翻译成模型能吃的张量。
# AutoTokenizer 会根据 checkpoint 名字自动挑对应的分词器类。

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded = tokenizer("Hello, I'm a single sentence!")
print(encoded)
# {'input_ids': [101, 8667, 117, ...102],
#  'token_type_ids': [0, 0, ...],
#  'attention_mask': [1, 1, ...]}
#
# 三个字段含义（务必记牢）：
#   input_ids       每个 token 的数字编号。101=[CLS] 起始，102=[SEP] 结束(BERT 专属)
#   token_type_ids  区分“句子A/句子B”。单句时全 0；做“句子对”任务(问答/蕴含)时才有 1
#   attention_mask  1=这个位置是真 token 要关注；0=这是填充(padding)要忽略

# 解码：把 input_ids 变回可读文本（会带上 [CLS]/[SEP] 特殊标记）
print(tokenizer.decode(encoded["input_ids"]))
# "[CLS] Hello, I'm a single sentence! [SEP]"

# ---- 句子对：传两个字符串，token_type_ids 会用 0/1 区分前后句 ----
pair = tokenizer("How are you?", "I'm fine, thank you!")
print(pair)

# ---- 直接返回 PyTorch 张量：return_tensors="pt"（"tf"=TF, "np"=NumPy）----
pair_pt = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
print(pair_pt)


# ==============================================================================
# 第 2 部分：填充 padding / 截断 truncation / 特殊标记
# ==============================================================================
# 一个 batch 里的句子长短不一，但张量必须是规整矩形。于是需要 padding + truncation。

# ---- padding：把短句补齐到统一长度 ----
enc = tokenizer(["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt")
print(enc)
# 短句尾部补 0（padding token），对应 attention_mask 也补 0 → 模型会忽略这些位置。
#
# padding 的几种取值：
#   padding=True 或 "longest"   补齐到“本 batch 里最长句”的长度（最常用、最省）
#   padding="max_length"        补齐到 max_length 指定长度（不指定则补到模型上限如512）
#   padding=False（默认）        不补齐

# ---- truncation：把超长句砍短（BERT 最长 512 token，超了会报错）----
long_text = "This is a very " + "very " * 60 + "long sentence."
enc = tokenizer(long_text, truncation=True)
print(enc["input_ids"])

# ---- padding + truncation + max_length 组合：精确控制到固定长度 ----
enc = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,          # 补齐
    truncation=True,       # 砍长
    max_length=5,          # 目标长度
    return_tensors="pt",
)
print(enc)

# ---- 特殊标记：[CLS] 句首、[SEP] 分隔/句尾，是 BERT 用来标句子边界的 ----
enc = tokenizer("How are you?")
print(enc["input_ids"])                    # [101, 1731, 1132, 1128, 136, 102]
print(tokenizer.decode(enc["input_ids"]))  # '[CLS] How are you? [SEP]'
#   add_special_tokens=False 可关掉自动加 [CLS]/[SEP]


# ==============================================================================
# 第 3 部分：分词器底层三步 tokenize → convert_tokens_to_ids → decode
# ==============================================================================
# tokenizer("文本") 是“一步到位”的封装。拆开看其实是这三步：

sequence = "Using a Transformer network is simple"

# 第①步 tokenize：把字符串切成 subword 词元
tokens = tokenizer.tokenize(sequence)
print(tokens)
# ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
#   注意 '##former'：## 表示它接在前一个词元后面（BERT 的 WordPiece 子词切法）

# 第②步 convert_tokens_to_ids：把词元换成词表编号
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)          # [7993, 170, 13809, 23763, 2443, 1110, 3014]
#   ★ 注意：这一步不会加 [CLS]/[SEP]，而 tokenizer("文本") 会加。这是二者的区别。

# 第③步 decode：编号 → 字符串（逆过程）
print(tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014]))

# BertTokenizer 是具体类，AutoTokenizer 会自动选到它；下面两行等价效果：
_ = BertTokenizer.from_pretrained("bert-base-cased")
_ = AutoTokenizer.from_pretrained("bert-base-cased")


# ==============================================================================
# 第 4 部分：批处理 与 attention_mask 为什么至关重要
# ==============================================================================
# 关键认知：padding 补进去的 0，如果不告诉模型“忽略它”，会污染计算结果！
# attention_mask 就是那个“告诉模型哪些要忽略”的开关。下面用实验证明。

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 构造：第二条句子尾部用 pad_token 补齐
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],   # 最后一个是填充
]

# (a) 不给 attention_mask —— 模型会把填充 token 也算进去 → 第二行 logits 是错的
print(model(torch.tensor(batched_ids)).logits)

# (b) 给了正确的 attention_mask（填充位置=0）—— 第二行 logits 才正确
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],   # 最后一个填充位置置 0，让模型忽略
]
out = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(out.logits)
# 结论：批处理时 padding 和 attention_mask 必须成对出现。
#       实际写代码时用 tokenizer(..., padding=True) 会自动帮你把两者都算好，
#       这里手动构造只是为了看清底层原理。


# ==============================================================================
# 第 5 部分：模型 —— AutoModel vs AutoModelForSequenceClassification，logits→softmax
# ==============================================================================
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# ---- AutoModel：只出“隐藏状态”(特征向量)，没有任务头 ----
base = AutoModel.from_pretrained(checkpoint)
outputs = base(**inputs)                       # **inputs 把字典按参数名展开传入
print(outputs.last_hidden_state.shape)         # torch.Size([2, 16, 768])
#   含义：[批大小=2, 序列长=16, 隐藏维度=768]。这是“高维特征”，还不能直接当答案。

# ---- AutoModelForSequenceClassification：带“分类头”，直接出每个标签的分数 ----
clf = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = clf(**inputs)
print(outputs.logits.shape)                    # torch.Size([2, 2])  → 2条样本 × 2个标签
print(outputs.logits)
#   这是 logits：原始分数，不是概率（可能有负数、加起来不为1）。

# ---- logits → 概率：softmax ----
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)     # 每行加起来=1，才是“正面/负面”的概率
#   为什么模型不直接输出概率？因为训练时损失函数(交叉熵)通常内置了 softmax，
#   所以模型本身只输出 logits，用的时候我们自己再 softmax。

# ---- 标签含义在 config 里 ----
print(clf.config.id2label)   # {0: 'NEGATIVE', 1: 'POSITIVE'}


# ==============================================================================
# 第 6 部分：保存与加载
# ==============================================================================
model = AutoModel.from_pretrained("bert-base-cased")

# 保存：会写出两个文件 —— config.json(架构配置) + model.safetensors(权重)
model.save_pretrained("directory_on_my_computer")

# 重新加载：路径既可以是本地目录，也可以是 Hub 上的 checkpoint 名
model = AutoModel.from_pretrained("directory_on_my_computer")

# BertModel 是具体类，与 AutoModel 加载同一 checkpoint 效果一致
_ = BertModel.from_pretrained("bert-base-cased")


# ==============================================================================
# 第 7 部分：把三步串起来 = 手动复现一个 pipeline
# ==============================================================================
# 这段展示“pipeline 一行”背后到底发生了什么：分词 → 模型 → softmax → 取标签。

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "I hate this!"]

# ① 分词
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
# ② 模型前向
output = model(**tokens)
# ③ 后处理：softmax + 取最大概率的标签
probs = torch.nn.functional.softmax(output.logits, dim=-1)
labels = [model.config.id2label[i] for i in probs.argmax(dim=-1).tolist()]
for text, prob, label in zip(sequences, probs, labels):
    print(f"{text!r:60s} -> {label} ({prob.max().item():.4f})")

# ---- 对比：pipeline 一行搞定同样的事（它内部就是上面三步）----
classifier = pipeline("sentiment-analysis")
print(classifier(sequences))
# [{'label': 'POSITIVE', 'score': 0.99...}, {'label': 'NEGATIVE', 'score': 0.99...}]
