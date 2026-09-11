"""
================================================================================
 Chapter 7 · 微调掩码语言模型(MLM) + 翻译(Translation)（原始学习文件·带详细注释）
================================================================================
 本文件其实装了「两个任务」(课程把它们放一块了)：
   前半段(约 1~315 行)  微调掩码语言模型 MLM —— 让 BERT/DistilBERT 更懂某个领域(IMDB影评)
   后半段(约 317 行起)  翻译 Translation      —— 微调 Marian(en→fr) 序列到序列模型
 代码保持可运行，但真训练要 GPU + 联网 + push_to_hub。想快速看懂请看同目录：
   · Chapter7_主要NLP任务_学习笔记.py / Chapter7_案例闯关_主要NLP任务实战.py

 ── MLM 部分脉络 ──
   1) fill-mask：给 [MASK] 填词，感受“掩码语言模型”在干嘛
   2) 数据预处理：分词 → group_texts 把多条文本拼起来再切成等长块(chunk)
   3) DataCollatorForLanguageModeling：动态随机掩码 15%；全词掩码(whole word masking)
   4) Trainer 微调 + 困惑度 Perplexity(=exp(loss)，越低越好) 评估
 ── 翻译部分脉络 ──
   5) translation pipeline + text_target 处理“目标语言标签”
   6) DataCollatorForSeq2Seq / SacreBLEU 指标 / Seq2SeqTrainer

 关键记忆点：
   MLM 的标签就是“被遮住的原词”；只在被 [MASK] 的位置算 loss，其余位置标签是 -100。
   困惑度 Perplexity = exp(交叉熵 loss)，直观理解为“模型每步平均在几个词里犹豫”。
================================================================================
"""
# ------------------------------------------------------------------------------
# 1) fill-mask：先感受“掩码语言模型”在干嘛（给 [MASK] 填词）
# ------------------------------------------------------------------------------
from transformers import AutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")

# DistilBERT 的分词器来生成模型的输入
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

import torch
text = "This is a great [MASK]."
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# 找到 [MASK] 的位置并提取其 logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# 选择具有最高 logits 值的 [MASK] 候选词
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

from datasets import load_dataset
imdb_datasets=load_dataset('imdb')
print(imdb_datasets)
sample=imdb_datasets['train'].shuffle(seed=42).select(range(3))

for row in sample:
    print(f"\n'>>> Review: {row['text']} '")
    print(f"'>>> Label: {row['label']} '")
def  tokenize_function ( examples ):
    result=tokenizer(examples['text'])
    if tokenizer.is_fast:
        result['word_ids']=[result.word_ids(i) for i in range(len(result['input_ids']))]
    return result
# 使用 batched=True 激活快速多线程！
tokenize_datasets=imdb_datasets.map(tokenize_function,batched=True,remove_columns=['text','label'])
print(tokenize_datasets)
print(tokenizer.model_max_length)
# 切片操作会为每个特征生成一个列表的列表
tokenized_samples=tokenize_datasets['train'][:3]
for idx, sample in  enumerate (tokenized_samples[ "input_ids" ]):
     print ( f"'>>> 查看{idx} 的长度：{ len (sample)} '" )
concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f"'>>> Concatenated reviews length: {total_length}'")

#我们遍历中的每个特征concatenated_examples，并使用列表推导式为每个特征创建切片
chunk_size = 128
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}

for chunk in chunks["input_ids"]:
    print(f"'>>> Chunk length: {len(chunk)}'")

#如果最后一块数据小于某个值，则将其丢弃chunk_size。
#将最后一个数据块填充到其长度等于chunk_size.
def  group_texts ( examples ):
    # 连接所有文本
    concatenated_examples={k:sum(examples[k],[]) for k in examples.keys()}
    # 计算连接后文本的总长度
    total_length=len(concatenated_examples[list(examples.keys())[0]])
    # 如果最后一个文本块小于 chunk_size，则丢弃该文本块
    total_length=(total_length//chunk_size)*chunk_size
    # 按 max_len 的块分割
    result={
        k:[t[i:i+chunk_size] for i in range(0,total_length,chunk_size)]
        for k ,t in concatenated_examples.items()
    }
    # 创建一个新的标签列
    result['labels']=result['input_ids'].copy()
    return  result

#让我们使用group_texts()我们可靠的Dataset.map()函数来处理分词后的数据集
lm_datasets=tokenize_datasets.map(group_texts,batched=True)
print(lm_datasets)
print(tokenizer.decode(lm_datasets['train'][1]['input_ids']))
print(tokenizer.decode(lm_datasets['train'][1]['labels']))
#使用 Trainer API 微调 DistilBERT
from  transformers import  DataCollatorForLanguageModeling
data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm_probability=0.15)
#由于它需要一个列表dict，其中每个元素dict代表一段连续的文本，
# 因此我们首先遍历数据集，然后再将批次数据提供给整理器。我们移除了"word_ids"该数据整理器不需要的键：
samples=[lm_datasets['train'][i] for i in range(2)]
for sample in samples:
    _=sample.pop('word_ids')
    for chunk in data_collator(samples)["input_ids"]:
        print(f"\n'>>> {tokenizer.decode(chunk)} '")

#这种方法称为全词掩码。如果我们想使用全词掩码，就需要自己构建一个数据整理器。
#数据整理器就是一个函数，它接收一个样本列表并将其转换为一个批次，现在我们就来构建它
import collections
import numpy as np
from transformers import default_data_collator
wwm_probability=0.2
def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids=feature.pop('word_ids')
        # 创建单词和对应标记索引之间的映射
        mapping=collections.defaultdict(list)
        current_word_index=-1
        current_word=None
        for idx,word_id in enumerate(word_ids):
            if word_id is not  None:
                if word_id!=current_word:
                    current_word_index+=1
                    current_word=word_id
                mapping[current_word_index].append(idx)
        # 随机掩码单词
        mask=np.random.binomial(1,wwm_probability,(len(mapping),))
        input_ids=feature['input_ids']
        labels=feature['labels']
        new_labels=[-100]*len(labels)
        for word_id in np.where(mask)[0]:
            word_id=word_id.item()
            for idx in mapping(word_id):
                new_labels[idx]=labels[idx]
                input_ids[idx]=tokenizer.mask_token_id
        feature['labels']=new_labels
    return default_data_collator(features)
samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")

#快速下采样数据集的方法
train_size=10_000
test_size=int(0.1* train_size)
downsampled_dataset=lm_datasets['train'].train_test_split(test_size=test_size, train_size=train_size,seed=42)
print(downsampled_dataset)


from transformers import TrainingArguments

batch_size = 64
# 显示每个 epoch 的训练损失
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
)

from transformers import  Trainer
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset['train'],
    eval_dataset=downsampled_dataset['test'],
    data_collator=data_collator,
    tokenizer=tokenizer
)


import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.train()
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

def insert_random_mask(bath):
    features=[dict(zip(bath,t)) for t in zip(*batch_size)]
    # 为数据集中的每一列创建一个新的“masked”列
    mask_inputs=data_collator(features)
    return {'masked_'+k:v.numpy() for k,v in mask_inputs.items()}

downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

from torch.utils.data import DataLoader
from transformers import default_data_collator
batch_size=64
train_dataloader=DataLoader(
    downsampled_dataset['train'],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator
)
eval_dataloader=DataLoader(
    eval_dataset,batch_size=batch_size,collate_fn=data_collator
)

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

#指定学习率调度器
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

from huggingface_hub import get_full_repo_name

model_name = "distilbert-base-uncased-finetuned-imdb-accelerate"
repo_name = get_full_repo_name(model_name)
repo_name

from huggingface_hub import Repository

output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)

from tqdm.auto import tqdm
import torch
import math

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

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

mask_filler = pipeline(
    "fill-mask", model="huggingface-course/distilbert-base-uncased-finetuned-imdb"
)


preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")

# ==============================================================================
# ★★★ 从这里开始是「翻译 Translation」任务（与上面 MLM 是两码事）★★★
# ==============================================================================
# 5) 翻译：微调 Marian(en→fr)。序列到序列(seq2seq)：输入一句英文，输出一句法文。
#    关键：用 text_target 处理“目标语言(法文)标签”，让它用正确的分词器编码。
from datasets import load_dataset
raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
split_datasets["validation"] = split_datasets.pop("test")
split_datasets["train"][1]["translation"]
from transformers import pipeline

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")
split_datasets['train'][172]['translation']
translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format."
)
from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")
en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]

inputs = tokenizer(en_sentence, text_target=fr_sentence)
print(inputs)

wrong_targets = tokenizer(fr_sentence)
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(inputs["labels"]))
max_length = 128
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs
#一次性将该预处理应用于数据集的所有分割结果
tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

#使用 Trainer API 对模型进行微调
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
print(batch.keys())
print(batch["labels"])
print(batch["decoder_input_ids"])
for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])
#BLEU 的一个缺点是它要求文本已经分词，这使得使用不同分词器的模型之间的分数难以比较。
# 因此，目前最常用的翻译模型基准测试指标是SacreBLEU
import evaluate
metric = evaluate.load("sacrebleu")
predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)

predictions = ["This This This This"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)
predictions = ["This plugin"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)

#为了将模型输出转换为指标可以使用的文本，我们将使用以下tokenizer.batch_decode()方法。
# 我们只需要清除-100标签中的所有空格（分词器会自动清除填充标记）：
import  numpy as np
def compute_metrics(eval_preds):
    preds,labels=eval_preds
    # 如果模型返回的 logits 超过预测值
    if isinstance(preds,tuple):
        preds=preds[0]
        # 替换标签中的 -100，因为我们无法解码它们
    decoded_preds=tokenizer.batch_decode(preds,skip_special_tokens=True)
    labels=np.where(labels!=-100,labels,tokenizer.pad_token_id)
    # 一些简单的后处理
    decoded_labels=tokenizer.batch_decode(labels,skip_special_tokens=True)
    decoded_preds=[pred.strip() for pred in decoded_preds]
    decoded_labels=[[label.strip()] for label in decoded_labels]
    result=metric.compute(predictions=decoded_preds,references=decoded_labels)
    return  {"bleu": result["score"]}
#就像之前一样Trainer，我们使用一个TrainingArguments包含更多字段的子类
from  transformers import Seq2SeqTrainingArguments
args=Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-fr",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

#我们只会在训练前和训练后对模型进行一次评估。
# 我们进行了设置fp16=True，这加快了在现代 GPU 上进行训练的速度。
# 我们predict_with_generate=True按照上文所述进行了设定。
# 我们过去常常push_to_hub=True在每个周期结束时将模型上传到 Hub。

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
trainer.evaluate(max_length=max_length)
trainer.train()
trainer.evaluate(max_length=max_length)
trainer.push_to_hub(tags="translation", commit_message="Training complete")

#自定义训练循环
from torch.utils.data import DataLoader
tokenize_datasets.set_format('torch')
train_dataloader=DataLoader(
    tokenize_datasets['train'],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader=DataLoader(
        tokenize_datasets['validation'],collate_fn=data_collator,batch_size=8
)
model=AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
from torch.optim import AdamW
optimizer=AdamW(model.parameters(),lr=2e-5)
#任何实例化对象的单元格Accelerator。
from accelerate import Accelerator
accelerator=Accelerator()
model,optimizer,train_dataloader,eval_dataloader=accelerator.prepare(model,optimizer,train_dataloader,eval_dataloader)
#将数据发送train_dataloader到目标位置accelerator.prepare()，
# 我们可以利用其长度来计算训练步数。请记住，我们应该始终在准备好数据加载器之后执行此操作，
# 因为该方法会改变目标位置的长度DataLoader。我们使用经典的线性学习率递增策略，从学习率逐渐减小到 0：
from transformers import get_scheduler
num_train_epochs=3
num_update_steps_per_epoch=len(train_dataloader)
num_training_steps=num_train_epochs* num_update_steps_per_epoch
lr_scheduler=get_scheduler(
    'linear',
    optimizer=optimizer,
    num_training_steps=num_training_steps,
    num_warmup_steps=0,
)

#训练循环
def  postprocess ( predictions, labels ):
    predictions=predictions.cpu().numpy()
    labels=labels.cpu().numpy()
    decoded_preds=tokenizer.batch_decode(predictions,skip_special_tokens=True)
    # 将标签中的 -100 替换为空，因为我们无法解码它们。
    labels=np.where(labels!=-100 ,labels,tokenizer.pad_token_id)
    decoded_labels=tokenizer.batch_decode(labels,skip_special_tokens=True)
    decoded_preds=[ [pred.strip()]for pred in decoded_preds]
    decoded_labels=[[label.strip()] for  label in decoded_labels]
    return decoded_preds,decoded_labels

#就像标记分类一样，两个进程可能对输入和标签进行了不同形状的填充，
# 因此我们accelerator.pad_across_processes()在调用方法之前需要将
# 预测结果和标签的形状统一起来gather()。如果不这样做，评估要么会出错，要么会一直挂起。
from tqdm import tqdm
import torch
progress_bar=tqdm(range(num_training_steps))
for epoch in range(num_train_epochs):
    model.train()
    for batch in train_dataloader:
        outputs=model(**batch)
        loss=outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens=accelerator.unwrap_model(model).generate(
                batch['input_ids'],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels=batch['labels']
        # 需要对预测和标签进行填充才能收集
        generated_tokens=accelerator.pad_across_processes(generated_tokens,dim=1,pad_index=tokenizer.pad_token_id)
        labels=accelerator.pad_across_processes(labels,dim=1,pad_index=100)
        predictions_gethered=accelerator.gather(generated_tokens)
        labels_gathered=accelerator.gather(labels)
        decoded_preds, decoded_labels =predictions(predictions_gethered,labels_gathered)
        metric.add_batch(predictions=decoded_preds,references=decoded_labels)
    results=metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")
    accelerator.wait_for_everyone()
    unwrapped_model=accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir,save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )



#使用微调模型
from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/marian-finetuned-kde4-en-to-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")
translator(
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format."
)