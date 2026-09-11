"""
================================================================================
 Chapter 6 · 训练新分词器 + 快速分词器的特殊能力（原始学习文件·带详细注释）
================================================================================
 本文件对应 HuggingFace 课程「Training a new tokenizer / Fast tokenizers」两节。
 代码保持“可运行状态”（没有注释掉），但请注意下面两处会联网 / 吃资源：
   · load_dataset("code_search_net","python")  会下载数 GB 代码数据集
   · train_new_from_iterator(...)              会真的重新训练一个分词器（较慢）
 如果只想看效果、又不想下大数据，请看同目录：
   · Chapter6_分词器_学习笔记.py     ← 把本章原理系统讲一遍
   · Chapter6_案例闯关_分词器实战.py ← 全部用小语料，几秒跑完、无需大下载

 本文件脉络（从上到下）：
   1) 构建语料库：用「生成器」惰性加载，避免一次性把大数据读进内存
   2) train_new_from_iterator：拿一个老分词器的“配置”，在你自己的语料上重训词表
   3) 快速分词器(is_fast) 的特殊能力：tokens()/word_ids()/offset 映射
   4) Token 分类(NER)：从 pipeline 一路手写到「logits→概率→实体分组」
   5) 问答(QA)：offset 映射 + 长文本滑动窗口(stride) 定位答案在原文的位置
================================================================================
"""

# ------------------------------------------------------------------------------
# 1) 构建语料库（用生成器惰性加载，省内存）
# ------------------------------------------------------------------------------
from datasets import load_dataset

# 数据集：code_search_net 的 python 子集，每条样本含一个完整函数的源码字符串
raw_datasets=load_dataset("code_search_net","python")
raw_datasets["train"]
print(raw_datasets["train"][123456]["whole_func_string"])
# def handle_simple_responses(
#       self, timeout_ms=None, info_cb=DEFAULT_MESSAGE_CALLBACK):
#     """Accepts normal responses from the device.
#
#     Args:
#       timeout_ms: Timeout in milliseconds to wait for each response.
#       info_cb: Optional callback for text sent from the bootloader.
#
#     Returns:
#       OKAY packet's message.
#     """
#     return self._accept_responses('OKAY', info_cb, timeout_ms=timeout_ms)

# 除非你的数据集很小，否则不要取消注释以下行！
# training_corpus = [raw_datasets["train"][i: i + 1000]["whole_func_string"]
# for i in range(0, len(raw_datasets["train"]), 1000)]
#这行代码不会获取数据集中的任何元素；它只是创建一个可以在 Pythonfor循环中使用的对象。文本只有在需要时才会加载（即for循环执行到需要它们的步骤时），
# 并且每次只加载 1000 条文本。这样，即使处理庞大的数据集，也不会耗尽所有内存
training_corpus=(raw_datasets['train'][i:i+1000]["whole_func_string"] for i in  range(0,len(raw_datasets['train']),1000))
#生成器对象的缺点在于它只能使用一次
gen=(i for i in range(10))
print(list(gen))#[ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ]
print(list(gen))#[]

#我们定义了一个返回生成器的函数：
def get_training_corpus():
    return (raw_datasets['train'][i:i+1000]["whole_func_string"] for i in range(0,len(raw_datasets["train"]),1000))
training_corpus = get_training_corpus()
#for您还可以使用以下语句在循环内定义生成器yield：
#生成与之前完全相同的生成器，但允许你使用比列表推导式更复杂的逻辑。
def  get_training_corpus ():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx:start_idx + 1000]
        yield samples["whole_func_string"]
# ------------------------------------------------------------------------------
# 2) 训练一个“新”分词器（复用老分词器的算法，只在你的语料上重学词表）
# ------------------------------------------------------------------------------
# train_new_from_iterator：不是从零写算法，而是「借用 gpt2 的分词流水线结构」，
# 在你给的语料上重新统计、生成一份适配你领域(这里是 Python 代码)的新词表。
# 好处：对代码里的缩进、下划线命名等，新分词器切得更短更合理。
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
print(len(tokens))
print(len(old_tokenizer.tokenize(example)))

example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """
tokenizer.tokenize(example)
#
# ['class', 'ĠLinear', 'Layer', '():', 'ĊĠĠĠ', 'Ġdef', 'Ġ__', 'init', '__(', 'self', ',', 'Ġinput', '_', 'size', ',',
#  'Ġoutput', '_', 'size', '):', 'ĊĠĠĠĠĠĠĠ', 'Ġself', '.', 'weight', 'Ġ=', 'Ġtorch', '.', 'randn', '(', 'input', '_',
#  'size', ',', 'Ġoutput', '_', 'size', ')', 'ĊĠĠĠĠĠĠĠ', 'Ġself', '.', 'bias', 'Ġ=', 'Ġtorch', '.', 'zeros', '(',
#  'output', '_', 'size', ')', 'ĊĊĠĠĠ', 'Ġdef', 'Ġ__', 'call', '__(', 'self', ',', 'Ġx', '):', 'ĊĠĠĠĠĠĠĠ',
#  'Ġreturn', 'Ġx', 'Ġ@', 'Ġself', '.', 'weights', 'Ġ+', 'Ġself', '.', 'bias', 'ĊĠĠĠĠ']

#保存分词器
tokenizer.save_pretrained("code-search-net-tokenizer")


# ------------------------------------------------------------------------------
# 3) 快速分词器(Fast Tokenizer) 的特殊能力
# ------------------------------------------------------------------------------
# “快速”分词器由 Rust 实现(tokenizers 库)，除了快，还多出关键能力：
#   · tokens()        看切出来的 token 文本
#   · word_ids()      每个 token 属于「原文第几个单词」（子词会指向同一个单词）
#   · offset 映射     每个 token 对应「原文的字符区间 [start,end)」——NER/QA 定位靠它
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))
#两种方法可以检查分词器是快速的还是慢速的。我们可以检查以下is_fast属性tokenizer
print(tokenizer.is_fast)
print(encoding.is_fast)
#直接访问令牌
print(encoding.tokens())
#word_ids()方法获取每个词元所属单词的索引
print(encoding.word_ids())
#word_to_chars()最后，我们可以使用`or`token_to_chars()和
# char_to_word()`or`方法将任何单词或标记映射到原文中的字符，
# 反之亦然char_to_token()。例如， ` word_ids()or` 方法告诉我们##yl索引为 3 的单词的一部分，
# 但它在句子中是哪个单词呢？我们可以这样找出答案：
start, end = encoding.word_to_chars(3)
print(example[start:end])

# ------------------------------------------------------------------------------
# 4) Token 分类（NER 命名实体识别）：从 pipeline 到手写
# ------------------------------------------------------------------------------
# 4.1 先用高层 pipeline 拿基础结果（一行搞定）
from transformers import pipeline

token_classifier = pipeline("token-classification")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

from transformers import pipeline
#aggregation_strategy将改变每个分组实体的得分计算方式。
# "simple"得分是给定实体中每个词元得分的平均值：
# 例如，“Sylvain”的得分是我们在上一个示例中看到的词元a S、##ylb、##vac和d得分的平均值##in。其他可用策略包括
# "first"其中，每个实体的得分是该实体的第一个标记的得分（因此对于“Sylvain”，得分为 0.993828，即该标记的得分S）。
# "max"其中，每个实体的得分是该实体中所有标记的最高得分（因此，“拥抱的脸”的得分是 0.98879766，即“脸”的得分）。
# "average"其中，每个实体的得分是组成该实体的单词得分的平均值（因此，“Sylvain”与策略没有区别"simple"，
# 但“Hugging Face”的得分为 0.9819，是“Hugging”得分 0.975 和“Face”得分 0.98879 的平均值）。
token_classifier = pipeline("token-classification", aggregation_strategy="simple")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

# 4.2 手写完整流程：从输入 → 模型 logits → 概率 → 预测标签
from transformers import AutoTokenizer, AutoModelForTokenClassification
model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)
model=AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "我的名字是Sylvain，我在布鲁克林的Hugging Face工作。"
inputs = tokenizer(example, return_tensors= "pt" )
outputs = model(**inputs)
print(inputs["input_ids"].shape)
print(outputs.logits.shape)

import torch
#模型有 9 个不同的标签，因此模型的输出形状为 1 x 19 x 9。
# 与文本分类流程类似，我们使用 softmax 函数将这些 logits 转换为概率，
# 并取 argmax 来获得预测结果（注意，我们可以对 logits 取 argmax，因为 softmax 不会改变顺序）
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)

#该model.config.id2label属性包含索引到标签的映射，我们可以利用这些映射来理解预测结果：
print(model.config.id2label)
results = []
inputs_with_offsets=tokenizer(example,return_offsets_mapping=True)
tokens=inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]
for idx, pred in enumerate (predictions):
    # id2label 是字典，所以使用 []
    label = model.config.id2label[pred]
    # 过滤掉 O 标签
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )
print(results)
# [{'entity': 'I-PER', 'score': 0.9993828, 'index': 4, 'word': 'S'},
#  {'entity': 'I-PER', 'score': 0.99815476, 'index': 5, 'word': '##yl'},
#  {'entity': 'I-PER', 'score': 0.99590725, 'index': 6, 'word': '##va'},
#  {'entity': 'I-PER', 'score': 0.9992327, 'index': 7, 'word': '##in'},
#  {'entity': 'I-ORG', 'score': 0.97389334, 'index': 12, 'word': 'Hu'},
#  {'entity': 'I-ORG', 'score': 0.976115, 'index': 13, 'word': '##gging'},
#  {'entity': 'I-ORG', 'score': 0.98879766, 'index': 14, 'word': 'Face'},
#  {'entity': 'I-LOC', 'score': 0.99321055, 'index': 16, 'word': 'Brooklyn'}]
#管道还提供了原始句子中每个实体的“ startand”信息end。
# 这就是偏移量映射发挥作用的地方。要获取偏移量，
# 我们只需return_offsets_mapping=True在将分词器应用于输入时进行设置即可：
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
inputs_with_offsets["offset_mapping"]

#对实体进行分组的同时对预测结果进行后处理的代码，
# 我们将连续且标记为 的实体分组在一起I-XXX，但第一个实体除外，
# 它可以标记为B-XXX或（因此，当我们得到、一种新的实体类型，
# 或者告诉我们同一类型的实体即将开始时，I-XXX我们就停止对实体进行分组）：OB-XXX
import numpy as np

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]
    if label != "O":
        # 移除 B- 或 I-
        label = label[2:]
        start, _ = offsets[idx]# 获取所有标记为 I-label 的 token
        all_scores = []
        while (
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] == f"I-{label}"
        ):
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1# 得分是该分组实体中所有词元得分的平均值

        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

print(results)
#
# [{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18},
#  {'entity_group': 'ORG', 'score': 0.97960204, 'word': 'Hugging Face', 'start': 33, 'end': 45},
#  {'entity_group': 'LOC', 'score': 0.99321055, 'word': 'Brooklyn', 'start': 49, 'end': 57}]

# ------------------------------------------------------------------------------
# 5) 问答(QA)流程中的快速分词器：offset 定位 + 长文本滑窗(stride)
# ------------------------------------------------------------------------------
# 5.1 先用 pipeline 感受 QA（从 context 里抽出答案片段）
from transformers import pipeline

question_answerer = pipeline("question-answering")
context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"
question_answerer(question=question, context=context)
#
# {'score': 0.97773,
#  'start': 78,
#  'end': 105,
#  'answer': 'Jax, PyTorch and TensorFlow'}

long_context = """
🤗 Transformers: State of the Art NLP

🤗 Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

🤗 Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internals are exposed as consistently as possible.
  - Model files can be used independently of the library for quick experiments.

🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question_answerer(question=question, context=long_context)
#
# {'score': 0.97149,
#  'start': 1892,
#  'end': 1919,
#  'answer': 'Jax, PyTorch and TensorFlow'}

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs=tokenizer(question,context,return_tensors="pt")
outputs=model(**inputs)
print(outputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)
#torch.Size([1, 66]) torch.Size([1, 66])

# 为了将这些logits转换为概率，我们将应用softmax函数——但在此之前，
# 我们需要确保屏蔽掉不属于上下文的索引。我们的输入是[CLS] question [SEP] context [SEP]，
# 因此我们需要屏蔽问题中的标记以及[SEP]标记。不过，我们将保留[CLS]标记，因为有些模型使用它来指示答案不在上下文中。
#
# 由于之后我们会应用 softmax 函数，所以只需要将需要掩码的 logits
# 值替换为一个较大的负数即可。这里，我们使用-10000：
import torch

sequence_ids = inputs.sequence_ids()
# 屏蔽除上下文标记之外的所有内容
mask = [i != 1 for i in sequence_ids]
# 取消屏蔽 [CLS] 标记
mask[0] = False
mask = torch.tensor(mask)[None]

start_logits[mask] = -10000
end_logits[mask] = -10000
#正确地屏蔽了对应于我们不想预测的位置的logits，我们可以应用softmax函数了
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]
#假设事件“答案从 开始start_index”和“答案结束于end_index”相互独立，
# 则答案从 开始start_index且结束于 的概率end_index为：
#start_probabilities [ start_index ]×end_probabilities [ end_index ]
scores = start_probabilities[:, None] * end_probabilities[None, :]
scores = torch.triu(scores)

max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
print(scores[start_index, end_index])
#答案的词元（token）start_index部分end_index，现在只需要将其转换为上下文中的字符索引。
# 这时偏移量就派上用场了。我们可以像在词元分类任务中那样获取并使用它们
inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]

start_char, _ = offsets[start_index]
_, end_char = offsets[end_index]
answer = context[start_char:end_char]
result = {
    "answer": answer,
    "start": start_char,
    "end": end_char,
    "score": scores[start_index, end_index],
}
print(result)

#question-answering如果我们尝试对之前用作示例的问题和长上下文进行分词，
# 我们将得到比管道中使用的最大长度（384）更高的词元数量：
inputs = tokenizer(question, long_context)
print(len(inputs["input_ids"]))

inputs = tokenizer(question, long_context, max_length=384, truncation="only_second")
print(tokenizer.decode(inputs["input_ids"]))

#这意味着模型很难选出正确答案。为了解决这个问题，该question-answering流程允许我们将上下文分割成更小的块，
# 并指定最大长度。为了确保我们不会因为分割位置错误而导致无法找到答案，流程还在块之间设置了一些重叠。

#我们可以通过添加参数让分词器（快速或慢速）帮我们完成这项工作return_overflowing_tokens=True，
# 并且我们可以使用参数指定所需的重叠范围stride。以下是一个使用较短句子的示例

sentence = "This sentence is not too long but we are going to split it anyway."
inputs = tokenizer(
    sentence, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)
# '[CLS] This sentence is not [SEP]'
# '[CLS] is not too long [SEP]'
# '[CLS] too long but we [SEP]'
# '[CLS] but we are going [SEP]'
# '[CLS] are going to split [SEP]'
# '[CLS] to split it anyway [SEP]'
# '[CLS] it anyway. [SEP]'
for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
#我们可以看到，句子被分成若干块，每块inputs["input_ids"]最多有 6 个标记
# （我们需要添加填充才能使最后一块与其他块大小相同），并且每块之间有 2 个标记的重叠。
print(inputs.keys())
print(inputs["overflow_to_sample_mapping"])

sentences = [
    "This sentence is not too long but we are going to split it anyway.",
    "This sentence is shorter but will still get split.",
]
inputs = tokenizer(
    sentences, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

print(inputs["overflow_to_sample_mapping"])
# | 参数                         | 意思    |
# | -------------------------- | ----- |
# | `truncation=False`         | 不截断   |
# | `truncation=True`          | 超长就截断 |
# | `truncation="only_first"`  | 只截第一段 |
# | `truncation="only_second"` | 只截第二段 |

# | `padding`      | 意思                    | 什么时候用         |
# | -------------- | --------------------- | ------------- |
# | `False`        | 不补 `PAD`              | 默认，不需要对齐时     |
# | `True`         | 补到**当前 batch 最长**     | 最常用           |
# | `"longest"`    | 同样补到**当前 batch 最长**   | 和 `True` 基本等价 |
# | `"max_length"` | 补到 `max_length` 指定的长度 | 想固定长度         |
# | `"do_not_pad"` | 不补                    | 明确表示不 padding |

inputs = tokenizer(
    question,
    long_context,
    stride=128,
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")

inputs = inputs.convert_to_tensors("pt")
print(inputs["input_ids"].shape)

outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)

sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
# Mask all the [PAD] tokens
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))

start_logits[mask] = -10000
end_logits[mask] = -10000

candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = torch.triu(scores).argmax().item()

    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

print(candidates)

for candidate, offset in zip(candidates, offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = long_context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
    print(result)
#
# {'answer': '\n🤗 Transformers: State of the Art NLP', 'start': 0, 'end': 37, 'score': 0.33867}
# {'answer': 'Jax, PyTorch and TensorFlow', 'start': 1892, 'end': 1919, 'score': 0.97149}


