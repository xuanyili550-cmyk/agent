from datasets import load_dataset
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          AutoModelForSequenceClassification,TrainingArguments)
import numpy as np
import evaluate
raw_datasets=load_dataset('nyu-mll/glue','mrpc')
checkpoint='bert-base-uncased'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(example):
    return tokenizer(list(example['sentence1']),list(example['sentence2']),truncation=True)
tokenized_datasets=raw_datasets.map(tokenize_function,batched=True)
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
#在定义模型之前，首先需要Trainer定义一个TrainingArguments类，用于存放训练和评估过程中使用的所有超参数Trainer。
# 你只需要提供一个目录，用于保存训练好的模型以及训练过程中的检查点。其他所有参数都可以保留默认值，
# 这对于基本的微调来说已经足够好了
training_args = TrainingArguments("test-trainer")

model=AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)
#定义一个训练器——包括模型model、training_args训练集和验证集、
# 我们的数据结构data_collator以及我们的数据processing_class。processing_class参数是新增的，
# 它告诉训练器要使用哪个分词器进行处理
from transformers import Trainer
trainer=Trainer(model,training_args,train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                data_collator=data_collator,
                processing_class=tokenizer
                )
#为了在我们的数据集上微调模型，我们只需要调用train()我们的方法Trainer：
trainer.train()
predictions=trainer.predict(tokenized_datasets['validation'])
#输出predict()是另一个包含三个字段的命名元组：predictions`loss`、label_ids`time` 和
# ` metricstime_metrics`。`loss`metrics字段仅包含传入数据集上的损失值，
# 以及一些时间指标（预测所花费的总时间和平均时间）。当我们完成compute_metrics()函数
# 并将其传递给 `time_metrics` 时Trainer，`time_metrics` 字段还将包含 `time_metrics`
# 返回的指标compute_metrics()
print(predictions.predictions.shape, predictions.label_ids.shape)
#predictions这是一个形状为 408 x 2 的二维数组（408 是我们使用的数据集中的元素数量）
# 。这些是我们传递给模型的数据集中每个元素的 logits 值predict()
preds = np.argmax(predictions.predictions, axis= -1 )
#结果preds与标签进行比较。为了构建我们的compute_metric()函数
metric = evaluate.load( "glue" , "mrpc" )
metric.compute(predictions=preds, references=predictions.label_ids)
#{'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}

def  compute_metrics ( eval_preds ):
    metric= evaluate.load('glue', "mrpc" )
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#为了查看它如何用于在每个 epoch 结束时报告指标，以下是如何使用Trainer此compute_metrics()函数定义一个新的指标
training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
#混合精度训练fp16=True：在训练参数中使用混合精度训练可以加快训练速度并减少内存使用量
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    fp16=True,  # 启用混合精度
)
#梯度累积：当 GPU 内存有限时，可有效实现更大的批处理大小：
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 有效批次大小 = 4 * 4 = 16
)
#学习率调整：训练器默认使用线性衰减，但您可以自定义此设置：
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    learning_rate=2e-5,
    lr_scheduler_type="cosine", # 尝试不同的调度器
)


raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(list(example["sentence1"]), list(example["sentence2"]), truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#准备训练
# 在实际编写训练循环之前，我们需要定义一些对象。首先是用于遍历批次的 DataLoader。
# 但在定义这些 DataLoader 之前，我们需要对数据进行一些后处理tokenized_datasets，
# 以处理一些原本由程序Trainer自动完成的工作。具体来说，我们需要：
#
# 删除模型不希望出现的列（例如 ` sentence1and`sentence2列）。
# 将列重命名label为labels（因为模型期望参数被命名为labels）。
# 设置数据集的格式，使其返回 PyTorch 张量而不是列表。
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
#["attention_mask", "input_ids", "labels", "token_type_ids"]
#定义数据加载器
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
#为了快速检查数据处理过程中是否存在错误，我们可以像这样检查一批数据：
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
#
# {'attention_mask': torch.Size([8, 65]),
#  'input_ids': torch.Size([8, 65]),
#  'labels': torch.Size([8]),
#  'token_type_ids': torch.Size([8, 65])}
#
# shuffle=True请注意，由于我们为训练数据加载器进行了设置，并且我们在批次内填充到最大长度，因此您的实际形状可能会略有不同。
#
# 现在我们已经彻底完成了数据预处理（对于任何机器学习从业者来说，这都是一个令人满意却又难以企及的目标），接下来让我们看看模型。我们按照上一节中的方法实例化它：

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels= 2 )

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

#tensor(0.5441, grad_fn=<NllLossBackward>) torch.Size([8, 2])

from torch.optim import AdamW
#
# AdamW 带权重衰减：AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
# 8 位 Adam 优化器：bitsandbytes用于内存高效优化
# 不同的学习率：较低的学习率（1e-5 到 3e-5）通常对大型模型效果更好。
optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler
num_epochs = 3
num_training_steps=num_epochs*len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

import torch

# Mac 用 mps；原课程写死 cuda，你没 N卡会退到 cpu，建议按下面这样写
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
# device(type='cuda')   # ← 这行是课程在展示 device 的样子（repr），不是代码，注释掉；否则会 TypeError

#准备开始训练！为了大致了解训练何时结束，我们使用tqdm以下库在训练步骤数上方添加进度条：

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))
# 现代训练优化：为了让你的训练循环更加高效，请考虑以下几点：
#
# 渐变裁剪：torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)在……之前添加optimizer.step()
# 混合精度：torch.cuda.amp.autocast()用于GradScaler加快训练速度
# 梯度累积：累积多个批次的梯度以模拟更大的批次大小。
# 检查点机制：定期保存模型检查点，以便在训练中断后恢复训练。
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
#{'accuracy': 0.8431372549019608, 'f1': 0.8907849829351535}

#Accelerate 会自动处理分布式训练、混合精度和设备部署的复杂性。从创建训练和验证数据加载器开始

#大部分工作都在将数据加载器、模型和优化器发送到指定位置的代码行中完成accelerator.prepare()。
# 这将把这些对象包装在合适的容器中，以确保分布式训练按预期工作。
# 剩下的更改是删除将批次放在指定位置的代码行device
# （同样，如果您想保留它，可以将其更改为使用accelerator.device），
# 并将其替换loss.backward()为accelerator.backward(loss)。
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


#Accelerate 的完整训练循环示例

from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# 以下是一些在生产环境中使用时需要考虑的其他因素：
#
# 模型评估：评估模型时，务必使用多个指标，而不仅仅是准确率。使用 🤗 Evaluate 库进行全面评估。
#
# 超参数调优：考虑使用 Optuna 或 Ray Tune 等库进行系统性的超参数优化。
#
# 模型监控：在整个训练过程中跟踪训练指标、学习曲线和验证性能。
#
# 模型共享：训练完成后，将您的模型分享到 Hugging Face Hub，以便社区可以使用。
#
# 效率：对于大型模型，可考虑梯度检查点、参数高效的微调（LoRA、AdaLoRA）或量化方法等技术。


# 初始损失高：模型未经优化就启动，因此初始预测效果较差。
# 损失减少：随着训练的进行，损失通常会减少。
# 收敛：最终，损失值稳定在一个较低的水平，表明模型已经学习到了数据中的模式。

# Example of tracking loss during training with the Trainer
from transformers import Trainer, TrainingArguments
# import wandb   # ← wandb 未安装，注释掉；它是可选的实验追踪工具（需 pip install wandb + 登录）

# Initialize Weights & Biases for experiment tracking
# wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,  # Log metrics every 10 steps
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    report_to="none",  # 原为 "wandb"；没装 wandb 就用 "none"，否则会报错
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and automatically log metrics
trainer.train()
#
# 训练期间
# 在训练过程中（达到目标后trainer.train()），您可以监控以下关键指标：
#
# 损失收敛：损失是否仍在下降还是已经趋于平稳？
# 过拟合迹象：验证损失是否开始增加而训练损失却在减少？
# 学习率：曲线是否过于波动（学习率过高）或过于平坦（学习率过低）？
# 稳定性：是否存在突然的峰值或谷值下降，这表明可能存在问题？
# 训练后
# 训练过程完成后，您可以分析完整的曲线来了解模型的性能。
#
# 最终性能：该模型是否达到可接受的性能水平？
# 效率：能否用更少的训练轮数达到相同的性能？
# 泛化能力：训练性能和验证性能有多接近？
# 趋势：额外的培训是否有可能提高绩效？
#
# 过拟合症状：
#
# 训练损失持续下降，而验证损失则上升或趋于平稳。
# 训练准确率和验证准确率之间存在较大差距
# 训练准确率远高于验证准确率。
# 解决过拟合问题的方案：
#
# 正则化：添加 dropout、权重衰减或其他正则化技术
# 提前停止：当验证性能不再提升时停止训练。
# 数据增强：增加训练数据的多样性
# 降低模型复杂度：使用更小的模型或更少的参数。

# 使用提前停止检测过拟合的示例
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    num_train_epochs=10,   # 设置得高一些，但我们会提前停止
) # 添加提前停止以防止过拟合
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


# 课程用 diff 演示“怎么改参数”：-是旧值 +是新值。下面写成合法 Python，只留新值。
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,   # 原来是 5，这里演示改成 10
)


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,              # 原来 1e-5 → 改成 1e-4
    per_device_train_batch_size=32,  # 原来 16 → 改成 32
)

