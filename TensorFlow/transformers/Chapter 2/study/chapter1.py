from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
raw_datasets=load_dataset('nyu-mll/glue','mrpc')
checkpoint = "bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(list(example["sentence1"]),list(example["sentence2"]),truncation=True)

tokenized_datasets=raw_datasets.map(tokenize_function,batched=True)
data_collator=DataCollatorWithPadding(tokenizer)

from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
training_args = TrainingArguments("test-trainer")
model=AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=2)
trainer=Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    processing_class=tokenizer
)
import numpy as np
import evaluate
def compute_metrics(eval_preds):
    metric=evaluate.load("glue", "mrpc")
    logits, labels=eval_preds
    predictions=np.argmax(logits,axis=-1)
    return metric.compute(predictions=predictions,references=labels)

