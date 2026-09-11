"""
================================================================================
 Chapter 7 · 摘要 + 从零训练因果语言模型 + 问答（原始学习文件·带详细注释）
================================================================================
 这个文件装了「三个任务」(课程后三节合在一处)：
   任务A(约   1~322 行)  文本摘要 Summarization —— 微调 mT5，把长影评压成短标题
   任务B(约 324~643 行)  从零训练因果语言模型 CLM —— 从头训一个 GPT-2 写 Python 代码
   任务C(约 645 行起)     抽取式问答 QA —— 微调 BERT，从 context 里定位答案片段
 代码保持可运行，但真训练都要 GPU + 联网 + push_to_hub。快速看懂请看同目录：
   · Chapter7_主要NLP任务_学习笔记.py / Chapter7_案例闯关_主要NLP任务实战.py

 ── 摘要 A ──  seq2seq；ROUGE 指标(比生成摘要和参考摘要的“词/子串重合度”)；
              先建“取前3句”的基线，再看微调能不能超过它。
 ── 因果LM B ──  用 AutoConfig 从头造一个小 GPT-2(不加载权重)；DataCollator 设 mlm=False；
              自定义按“关键 token 加权”的损失；梯度累积 gradient_accumulation。
 ── 问答 C ──  ★把“答案的字符位置”转成“start/end token 下标”(用 offset + stride 滑窗)；
              评估用 SQuAD 的 exact_match/f1；预测时在 n_best 个起止组合里挑最优。

 关键记忆点：
   摘要/翻译都是 seq2seq，标签是“目标文本”，评估要先 decode 再算 ROUGE/BLEU。
   QA 的难点是“把答案在原文的字符区间，映射成 token 下标”，长 context 用 stride 切块。
================================================================================
"""
# ------------------------------------------------------------------------------
# 任务A · 文本摘要：准备多语言语料库（亚马逊图书评论：review_body → review_title）
# ------------------------------------------------------------------------------
from datasets import load_dataset
spanish_dataset = load_dataset( "amazon_reviews_multi" , "es" )
english_dataset = load_dataset( "amazon_reviews_multi" , "en" )
print(english_dataset)
#从训练集中抽取随机样本：validationtestreview_bodyreview_title
def show_samples(dataset,num_samples=3,seed=42):
    samples=dataset['train'].shuffle(seed=seed).select(range(num_samples))
    for example in  samples:
        print(f"\n'>> Title: {example['review_title']} ' ")
        print(f"'>> Review: {example['review_body']} ' ")

show_samples(english_dataset)

english_dataset.set_format('pandas')
english_df=english_dataset["train"][:]
print(english_df['product_category'].value_counts()[:20])

#让我们分别筛选两种语言的数据集，只保留这些产品。` Dataset.filter()schedule`
# 函数可以非常高效地对数据集进行切片，因此我们可以定义一个简单的函数来实现这一点：
def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )
english_dataset.reset_format()
spanish_books = spanish_dataset.filter(filter_books)
english_books = english_dataset.filter(filter_books)
show_samples(english_books)

from datasets import concatenate_datasets,DatasetDict
books_dataset=DatasetDict()
for split in english_books.keys():
    books_dataset[split]=concatenate_datasets([english_books[split],spanish_books[split]])
    books_dataset[split]=books_dataset[split].shuffle(seed=42)
show_samples(books_dataset)
books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)
# 变压器模型	描述	多种语言
# GPT-2	尽管 GPT-2 被训练成自回归语言模型，但您可以通过在输入文本末尾添加“TL;DR”来让它生成摘要。	❌
# 飞马座	它使用预训练目标来预测多句文本中的掩码句子。这种预训练目标比传统的语言建模更接近于文本摘要，并且在常用的基准测试中取得了很高的分数。	❌
# T5	一种通用的 Transformer 架构，它以文本到文本的框架来构建所有任务；例如，模型总结文档的输入格式为summarize: ARTICLE。	❌
# mT5	T5 的多语言版本，在涵盖 101 种语言的多语言 Common Crawl 语料库 (mC4) 上进行预训练。	✅
# 旧金山湾区捷运	一种新颖的 Transformer 架构，它同时包含编码器和解码器堆栈，经过训练可以重建损坏的输入，并结合了 BERT 和 GPT-2 的预训练方案。	❌
# mBART-50	BART 的多语言版本，已用 50 种语言进行预训练。	✅

#使用mt5-small该检查点
from transformers import AutoTokenizer
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
inputs = tokenizer("I loved reading the Hunger Games!")
print(inputs)
#正在处理的是哪种分词器：convert_ids_to_tokens()
print(tokenizer.convert_ids_to_tokens(inputs.input_ids))
#由于标签也是文本，它们的长度可能超过模型的最大上下文长度
#Transformer 中的分词器提供了一个巧妙的text_target参数处理标签和输入
max_input_length = 512
max_target_length = 30

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["review_title"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = books_dataset.map(preprocess_function, batched=True)
print(tokenized_datasets)
generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"

import evaluate
rouge_score = evaluate.load("rouge")
scores = rouge_score.compute(
    predictions=[generated_summary], references=[reference_summary]
)
print(scores)
print(scores["rouge1"].mid)
#Score(precision=0.86, recall=1.0, fmeasure=0.92)
#建立稳固的基础
import nltk
nltk.download("punkt")
#nltk并创建一个简单的函数来提取评论中的前三个句子。文本摘要的惯例是用换行符分隔每个摘要
from nltk.tokenize import sent_tokenize
def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])
print(three_sentence_summary(books_dataset["train"][1]["review_body"]))
#计算基线的 ROUGE 分数

def evaluate_baseline(dataset,metric):
    summaries=[three_sentence_summary(text) for text in dataset['review_body']]
    return metric.compute(predictions=summaries, references=dataset["review_title"])
import pandas as pd
score = evaluate_baseline(books_dataset["validation"], rouge_score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names)
print(rouge_dict)

#使用 Trainer API 微调 mT5
from transformers import AutoModelForSeq2SeqLM
model=AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
from transformers import Seq2SeqTrainingArguments

batch_size = 8
num_train_epochs = 8
# 显示每个 epoch 的训练损失
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)

#训练器提供一个compute_metrics()函数，以便在训练过程中评估我们的模型。对于摘要生成来说，
# 这比直接调用rouge_score.compute()模型的预测结果要复杂一些，
# 因为我们需要先将输出和标签解码成文本，然后才能计算 ROUGE 分数
import  numpy as np
def compute_metrics(eval_pred):
    # 将生成的摘要解码为文本
    predictions,labels=eval_pred
    decoded_preds=tokenizer.batch_decode(predictions,skip_special_tokens=True)
    # 将标签中的 -100 替换为其他值，因为我们无法解码它们
    labels=np.where(labels!=-100,labels,tokenizer.pad_token_id)
    # 将参考摘要解码为文本
    dacoded_labels=tokenizer.batch_decode(labels,skip_special_tokens=True)
    # ROUGE 要求每个句子后换行
    decoded_preds=['\n'.join(sent_tokenize(pred.stirp()))for pred in decoded_preds]
    decoded_labels=['\n'.join(sent_tokenize(label.stirp()))for label in dacoded_labels]
    # 计算 ROUGE 分数
    result=rouge_score.compute(predictions=decoded_preds,references=decoded_labels,use_stemmer=True)
    # 提取中位数
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}
#Transformers 提供了一个DataCollatorForSeq2Seq排序器，可以动态地为我们填充输入和标签
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
tokenized_datasets = tokenized_datasets.remove_columns(
    books_dataset["train"].column_names
)
features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)
#
# {'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 'input_ids': tensor([[  1494,    259,   8622,    390,    259,    262,   2316,   3435,    955,
#             772,    281,    772,   1617,    263,    305,  14701,    260,   1385,
#            3031,    259,  24146,    332,   1037,    259,  43906,    305,    336,
#             260,      1,      0,      0,      0,      0,      0,      0],
#         [   259,  27531,  13483,    259,   7505,    260, 112240,  15192,    305,
#           53198,    276,    259,  74060,    263,    260,    459,  25640,    776,
#            2119,    336,    259,   2220,    259,  18896,    288,   4906,    288,
#            1037,   3931,    260,   7083, 101476,   1143,    260,      1]]), 'labels': tensor([[ 7483,   259,  2364, 15695,     1,  -100],
#         [  259, 27531, 13483,   259,  7505,     1]]), 'decoder_input_ids': tensor([[    0,  7483,   259,  2364, 15695,     1],
#         [    0,   259, 27531, 13483,   259,  7505]])}

#实例化训练器
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
#训练
trainer.train()
#评估
trainer.evaluate()
#使用  Accelerate 对 mT5 进行微调
tokenized_datasets.set_format("torch")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
from torch.utils.data import DataLoader

batch_size = 8
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=batch_size
)
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE 要求每个句子后都必须有换行符
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

from huggingface_hub import get_full_repo_name

model_name = "test-bert-finetuned-squad-accelerate"
repo_name = get_full_repo_name(model_name)
from huggingface_hub import Repository

output_dir = "results-mt5-finetuned-squad-accelerate"
repo = Repository(output_dir, clone_from=repo_name)

from tqdm.auto import tqdm
import torch
import numpy as np

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            # 如果我们没有填充到最大长度，我们也需要填充标签
            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # 将标签中的 -100 替换为我们无法解码它们
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_labels
            )

            rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

    # 计算指标
    result = rouge_score.compute()
    # 提取 ROUGE 分数的中位数
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    print(f"Epoch {epoch}:", result)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )

from transformers import pipeline
hub_model_id = "huggingface-course/mt5-small-finetuned-amazon-en-es"
summarizer = pipeline("summarization", model=hub_model_id)
def print_summary(idx):
    review = books_dataset["test"][idx]["review_body"]
    title = books_dataset["test"][idx]["review_title"]
    summary = summarizer(books_dataset["test"][idx]["review_body"])[0]["summary_text"]
    print(f"'>>> Review: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> Summary: {summary}'")

# ==============================================================================
# ★★★ 任务B · 从零开始训练因果语言模型(CLM)：训一个会写 Python 的小 GPT-2 ★★★
# ==============================================================================
# 因果=只能看左边、预测下一个词(GPT 家族)。这里不加载预训练权重，用 AutoConfig 从头造。
def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
example_1 = "import numpy as np"
example_2 = "import pandas as pd"

print(
    any_keyword_in_string(example_1, filters), any_keyword_in_string(example_2, filters)
)
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset
def filter_streaming_dataset(dataset, filters):
    filtered_dict=defaultdict(list)
    total=0
    for sample in tqdm(iter(dataset)):
        total+=1
        if any_keyword_in_string(sample['content'],filters):
            for k ,v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)
# 此单元格执行时间会非常长，因此您应该跳过它并转到
#下一个单元格！此函数应用于流式数据集
from datasets import load_dataset

split = "train"  # "valid"
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
filtered_data = filter_streaming_dataset(data, filters)
#Hub 上提供筛选后的数据集供您下载
from datasets import load_dataset, DatasetDict
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)
print(raw_datasets)
for key in raw_datasets["train"][0]:
    print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

from transformers import AutoTokenizer

context_length = 128
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

outputs = tokenizer(
    raw_datasets["train"][:2]["content"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
print(tokenized_datasets)
#初始化新模型
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
#它DataCollatorForLanguageModeling同时支持掩码语言模型 (MLM) 和因果语言模型 (CLM)。
# 默认情况下，它会为 MLM 准备数据，但我们可以通过设置参数切换到 CLM mlm=False
from transformers import DataCollatorForLanguageModeling
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")

# input_ids shape: torch.Size([5, 128])
# attention_mask shape: torch.Size([5, 128])
# labels shape: torch.Size([5, 128])

#Accelerate 创建训练循环时看到它的实际应用
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()
#使用管道进行代码生成
import torch
from transformers import pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipe = pipeline(
    "text-generation", model="huggingface-course/codeparrot-ds", device=device
)
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
df = pd.DataFrame({'x': x, 'y': y})
df.insert(0,'x', x)
# txt = """\
# # dataframe with profession, income and name
# df = pd.DataFrame({'profession': x, 'income':y, 'name': z})
#
# # calculate the mean income per profession
# """
#
# print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
# # dataframe with profession, income and name
# df = pd.DataFrame({'profession': x, 'income':y, 'name': z})
#
# # calculate the mean income per profession
# profession = df.groupby(['profession']).mean()

# import random forest regressor from scikit-learn
keytoken_ids = []
for keyword in [
    "plt",
    "pd",
    "sk",
    "fit",
    "predict",
    " plt",
    " pd",
    " sk",
    " fit",
    " predict",
    "testtest",
]:
    ids = tokenizer([keyword]).input_ids[0]
    if len(ids) == 1:
        keytoken_ids.append(ids[0])
    else:
        print(f"Keyword has not single token: {keyword}")
from torch.nn import CrossEntropyLoss
import torch


def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    # 调整标签数量，使 tokens < n 预测 n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # 计算每个 token 的损失
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # 调整每个样本的大小并计算平均损失
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    # 计算并调整权重
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
        axis=[0, 2]
    )
    weights = alpha * (1.0 + weights)
    # 计算加权平均值
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss
from torch.utils.data.dataloader import DataLoader

tokenized_datasets.set_format("torch")
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, shuffle=True)
eval_dataloader = DataLoader(tokenized_datasets["valid"], batch_size=32)

weight_decay = 0.1
def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

model = GPT2LMHeadModel(config)
from torch.optim import AdamW

optimizer = AdamW(get_grouped_params(model), lr=5e-4)
from accelerate import Accelerator

accelerator = Accelerator(fp16=True)
samples_per_step = accelerator.state.num_processes * 32  # 每步样本数 = 进程数 × batch_size(32)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
from transformers import get_scheduler

num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)
evaluate()
from tqdm.notebook import tqdm

gradient_accumulation_steps = 8
eval_steps = 5_000

model.train()
completed_steps = 0
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        logits = model(batch["input_ids"]).logits
        loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
        if step % 100 == 0:
            accelerator.print(
                {
                    "samples": step * samples_per_step,
                    "steps": completed_steps,
                    "loss/train": loss.item() * gradient_accumulation_steps,
                }
            )
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress step {step}", blocking=False
                )


# ==============================================================================
# ★★★ 任务C · 抽取式问答(QA)：微调 BERT，从 context 里“抠出”答案片段 ★★★
# ==============================================================================
# 抽取式=答案一定是 context 的一个连续子串；模型预测“起始 token”和“结束 token”。
# 核心难点：把答案在原文的“字符区间”，映射成 token 的 start/end 下标(靠 offset)。
from datasets import load_dataset
raw_datasets = load_dataset("squad")

print("Context: ", raw_datasets["train"][0]["context"])
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])

print(raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1))
print(raw_datasets["validation"][0]["answers"])
print(raw_datasets["validation"][2]["answers"])
print(raw_datasets["validation"][2]["context"])
print(raw_datasets["validation"][2]["question"])
from transformers import AutoTokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]
inputs = tokenizer(question, context)
tokenizer.decode(inputs["input_ids"])
inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))

inputs = tokenizer(
    question,
    context,
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
inputs.keys()
#dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])
#更多例子标记化
inputs = tokenizer(
    raw_datasets["train"][2:6]["question"],
    raw_datasets["train"][2:6]["context"],
    max_length=100,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")
#循环查找答案的第一个和最后一个标记
answers = raw_datasets["train"][2:6]["answers"]
start_positions = []
end_positions = []

for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)

    # 查找上下文的开始和结束
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # 如果答案不在上下文中，则标签为 (0, 0)，
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
    else:
        # 否则，它是起始和结束标记位置
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx - 1)

        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_positions.append(idx + 1)

start_positions, end_positions


idx = 0
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

start = start_positions[idx]
end = end_positions[idx]
labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start : end + 1])

print(f"Theoretical answer: {answer}, labels give: {labeled_answer}")
#'Theoretical answer: the Main Building, labels give: the Main Building'
idx = 4
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

decoded_example = tokenizer.decode(inputs["input_ids"][idx])
print(f"Theoretical answer: {answer}, decoded example: {decoded_example}")
#'Theoretical answer: a Marian place of prayer and reflection, decoded example: [CLS] What is the Grotto at Notre Dame? [SEP] Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend " Venite Ad Me Omnes ". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grot [SEP]'



max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # 查找上下文的开始和结束
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
len(raw_datasets["train"]), len(train_dataset)

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

validation_dataset = raw_datasets["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)
len(raw_datasets["validation"]), len(validation_dataset)

#使用 Trainer API 对模型进行微调
small_eval_set = raw_datasets["validation"].select(range(100))
trained_checkpoint = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#移除模型预期之外的列eval_set，用这些小的验证集构建一个批次，并将其输入模型。
# 如果有GPU可用，我们会使用它来加快速度
import torch
from transformers import AutoModelForQuestionAnswering

eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("torch")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(
    device
)

with torch.no_grad():
    outputs = trained_model(**batch)
start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()

import collections

example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"]].append(idx)

import numpy as np

n_best = 20
max_answer_length = 30
predicted_answers = []

for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []

    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_set["offset_mapping"][feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # 跳过未完全位于上下文中的答案
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # 跳过长度小于 0 或大于 max_answer_length 的答案。
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                answers.append(
                    {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )

    best_answer = max(answers, key=lambda x: x["logit_score"])
    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

import evaluate

metric = evaluate.load("squad")
theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]
metric.compute(predictions=predicted_answers, references=theoretical_answers)
from tqdm.auto import tqdm


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # 遍历与该示例关联的所有特征
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 跳过未完全位于上下文中的答案
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # 跳过长度小于 0 或大于 max_answer_length 的答案
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        #如果 答案数量大于0，则选择得分最高的答案：
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

compute_metrics(start_logits, end_logits, eval_set, small_eval_set)
#{'exact_match': 83.0, 'f1': 88.25}

from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)
trainer.train()

predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])

from torch.utils.data import DataLoader
from transformers import default_data_collator

train_dataset.set_format("torch")
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    validation_set, collate_fn=default_data_collator, batch_size=8
)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
from accelerate import Accelerator

accelerator = Accelerator(fp16=True)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

#训练循环的完整代码

from tqdm.auto import tqdm
import torch
progress_bar=tqdm(range(num_training_steps))
for epoch in range(num_train_epochs):
    model.train()
    for step,batch in enumerate(train_dataloader):
        outputs=model(**batch)
        loss=outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs=model(**batch)
        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())
    start_logits=np.concatenate(start_logits)
    end_logits=np.concatenate(end_logits)
    start_logits=start_logits[:len(validation_dataset)]
    end_logits=end_logits[:len(validation_dataset)]
    metrics = compute_metrics(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"]
    )
    print(f"epoch {epoch}:", metrics)
    accelerator.wait_for_everyone()
    unwrapped_model=accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir,save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/bert-finetuned-squad"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"
question_answerer(question=question, context=context)