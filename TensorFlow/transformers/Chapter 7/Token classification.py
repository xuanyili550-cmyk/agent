"""
================================================================================
 Chapter 7 · Token 分类 / 命名实体识别(NER)（原始学习文件·带详细注释）
================================================================================
 对应课程「Main NLP tasks → Token classification」。这是本章 6 大任务的第 1 个。
 任务目标：给句子里的每个词打标签(人名/地名/机构名/其它/非实体)。
 代码保持“可运行状态”（没有注释掉），但注意：真正的 trainer.train() 要 GPU + 联网，
 且 push_to_hub=True 会往你的 Hub 推模型。想快速看懂原理/跑小例子，请看同目录：
   · Chapter7_主要NLP任务_学习笔记.py      ← 把本章 6 个任务系统讲一遍(可运行)
   · Chapter7_案例闯关_主要NLP任务实战.py  ← 全用小模型/小数据，能在 Mac 上跑

 本文件脉络：
   1) 数据集 conll2003：tokens + ner_tags(数字标签) + label_names(标签名)
   2) ★标签对齐 align_labels_with_tokens：把“词级标签”对齐到“子词级 token”(核心难点)
   3) 批量预处理 tokenize_and_align_labels + Dataset.map
   4) DataCollatorForTokenClassification：动态填充，标签也补 -100
   5) 评估 seqeval：precision/recall/f1/accuracy(按实体算，不是按 token 算)
   6) Trainer 微调 / 自定义训练循环(accelerate) / 用微调好的模型做推理

 关键记忆点：
   -100 是交叉熵损失“忽略”的标签：特殊标记([CLS]/[SEP])和“子词的非首片”都设 -100，
   这样它们不参与 loss，也不参与评估。B-XXX(实体开头) 的子词续片要改成 I-XXX。
================================================================================
"""
import numpy as np
from datasets import load_dataset, ClassLabel, Sequence

# ------------------------------------------------------------------------------
# 1) 数据集：conll2003（经典 NER 数据集）
# ------------------------------------------------------------------------------
raw_dataset=load_dataset('conll2003')
# DatasetDict({
#     train: Dataset({
#         features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
#         num_rows: 14041
#     })
#     validation: Dataset({
#         features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
#         num_rows: 3250
#     })
#     test: Dataset({
#         features: ['chunk_tags', 'id', 'ner_tags', 'pos_tags', 'tokens'],
#         num_rows: 3453
#     })
# })

print(raw_dataset['train'][0]['tokens'])
print(raw_dataset['train'][0]['ner_tags'])
ner_feature=raw_dataset['train'].features['ner_tags']
print(ner_feature)
sequence_label=Sequence(feature=ClassLabel(num_classes=9, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], names_file=None, id=None), length=-1, id=None)
print(sequence_label)
#此列包含的元素是序列ClassLabel。序列中元素的类型位于feature此元素的属性中，
# 我们可以通过查看该元素的属性ner_feature来访问名称列表：namesfeature
label_names=ner_feature.feature.names
print(label_names)
#['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
# token-classification我们在深入研究管道时已经见过这些标签，但为了快速回顾一下：
# O意思是该词不对应任何实体。
# B-PER/I-PER表示该词对应于/位于人形实体内部。
# B-ORG/I-ORG表示该词对应于组织实体的开头/位于组织实体内部。
# B-LOC/I-LOC表示该词对应于位置实体的开头/位于位置实体内部。
# B-MISC/I-MISC表示该词对应于某个杂项实体的开头/位于该杂项实体内部。

#标签解码
words=raw_dataset['train'][0]['tokens']
labels=raw_dataset[ "train" ][ 0 ][ "ner_tags" ]
line1=''
line2=''
for word ,label in zip(words,labels):
    full_label=label_names[label]
    max_length=max(len(word),len(full_label))
    line1+=word+" "*(max_length-len(word)+1)
    line2+=full_label+" "* (max_length-len(full_label)+1)
print(line2)
print(line1)


# ------------------------------------------------------------------------------
# 2) ★标签对齐：把“词级标签”对齐到“子词级 token”（本任务核心难点）
# ------------------------------------------------------------------------------
# 数据里标签是“每个词一个”，但分词器会把词切成子词(la→la ##mb)，token 变多了。
# is_split_into_words=True 告诉分词器“输入已经是分好的词列表”，它会再切子词。
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

inputs = tokenizer(raw_dataset["train"][0]["tokens"], is_split_into_words=True)
inputs.tokens()
#['[CLS]', 'EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'la', '##mb', '.', '[SEP]']
print(inputs.word_ids())
#特殊词元的标签为 `<token>` -100。这是因为默认情况下， `<token> -100`
# 是一个在我们使用的损失函数（交叉熵损失函数）中会被忽略的索引。然后，
# 每个词元都获得与其所在单词开头词元相同的标签，因为它们属于同一个实体。
# 对于位于单词内部但不在开头的词元，我们将 `<token>` 替换为 ` B-<token> I-`（因为该词元并非实体的开头）
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # 一个新词的开头！
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # 特殊标记
            new_labels.append(-100)
        else:
            # 与前一个标记相同的词
            label = labels[word_id]
            # 如果标签是 B-XXX，则将其更改为 I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

labels = raw_dataset["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))

#为了预处理整个数据集，我们需要对所有输入进行分词，并应用于align_labels_with_tokens()所有标签。
# 为了充分利用快速分词器的速度，最好同时分词大量文本，因此我们将编写一个函数来处理示例列表
# ，并使用Dataset.map()带有相应选项的方法batched=True。与之前的示例唯一不同的是，
# word_ids()当分词器的输入是文本列表（或者在本例中是单词列表的列表）时，
# 该函数需要获取要提取词 ID 的示例的索引，因此我们也添加了该参数：
def tokenize_and_align_labels(examples):
    tokenizer_inputs=tokenizer(examples['tokens'],truncation=True,is_split_into_words=True)
    all_labels=examples['ner_tags']
    new_labels = []
    for i, labels in enumerate (all_labels):
        word_ids=tokenizer_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels,word_ids))
    tokenizer_inputs['labels']=new_labels
    return tokenizer_inputs
#一次性将所有这些预处理步骤应用到数据集的其他部分
tokenized_datasets=raw_dataset.map(
    tokenize_and_align_labels,batched=True,remove_columns=raw_dataset['train'].column_names
)

# ------------------------------------------------------------------------------
# 4) DataCollatorForTokenClassification：动态填充（输入补 [PAD]、标签补 -100）
# ------------------------------------------------------------------------------
# 为什么要专门的 collator：普通填充只补 input_ids，NER 还要把 labels 也补齐(补 -100)。
from transformers import  DataCollatorForTokenClassification
data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
#为了在少量样本上测试这个方法，我们可以直接在已分词训练集中的示例列表上调用它
batch=data_collator([tokenized_datasets['train'][i] for i in range(2)])
batch['labels']
# tensor([[-100,    3,    0,    7,    0,    0,    0,    7,    0,    0,    0, -100],
#         [-100,    1,    2, -100, -100, -100, -100, -100, -100, -100, -100, -100]])
for i in range(2):
    print(tokenized_datasets['train'][i]['labels'])
# [-100, 3, 0, 7, 0, 0, 0, 7, 0, 0, 0, -100]
# [-100, 1, 2, -100]
# ------------------------------------------------------------------------------
# 5) 评估：seqeval（按“实体”算 P/R/F1，而不是按单个 token 算）
# ------------------------------------------------------------------------------
# 为什么用 seqeval：NER 要求整个实体(如 B-PER I-PER)都对才算对，seqeval 懂 BIO 规则。
import  evaluate
metric=evaluate.load('seqeval')
labels=raw_dataset['train'][0]['ner_tags']
labels=[label_names[i] for i in labels]
print(labels)
#['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
#预测创建虚假预测
predictions = labels.copy()
predictions[ 2 ] = "O"
metric.compute(predictions=[predictions],references=[labels])

import numpy as np
def compute_metrics(eval_preds):
    logits,labels=eval_preds
    predictions=np.argmax(logits,axis=-1)
    true_labels = [[label_names[l] for l in label if l!=-100] for label in labels]
    true_predictions=[
        [label_names[p] for (p,l) in zip(predictions,label) if l!=-100]
        for predictions,label in zip(predictions,labels)
    ]
    all_metrics=metric.compute(predictions=true_predictions,references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
#定义模型
#id2label分别label2id包含从 ID 到标签以及反之亦然的映射关系
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
print(model.config.num_labels)

from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)
from transformers import Trainer
#将所有信息传递给系统Trainer并启动培训
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
trainer.train()

# ------------------------------------------------------------------------------
# 7) 自定义训练循环（accelerate）：把 Trainer 拆开，看清每一步
# ------------------------------------------------------------------------------
# 想完全掌控训练(自定义 loss、日志、多卡)时用它；Accelerator 帮你处理设备/分布式。
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)
#我们重新实例化模型，以确保我们不是继续之前的微调，而是再次从 BERT 预训练模型开始：

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
#需要一个优化器。我们将使用经典的 `std::vector` AdamW，
# 它类似于 ` Adamstd::vector`，但改进了权重衰减的应用方式
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)
from accelerate import Accelerator

accelerator = Accelerator()
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
from huggingface_hub import Repository, get_full_repo_name
model_name = "bert-finetuned-ner-accelerate"
repo_name = get_full_repo_name(model_name)
print(repo_name)
output_dir = "bert-finetuned-ner-accelerate"
repo = Repository(output_dir, clone_from=repo_name)
#训练循环
def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # 移除忽略的索引（特殊标记）并转换为标签
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions
#以下是训练循环的完整代码：
from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # 训练
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # 评估
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # 需要对预测和标签进行填充以便收集
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    # 保存并上传
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )


#使用微调模型
from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/bert-finetuned-ner"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
