import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    BertTokenizer,
    AutoModel,
    BertModel,
    AutoModelForSequenceClassification
)
tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
encoded = tokenizer("Hello, I'm a single sentence!")
print(encoded)

print(tokenizer.decode(encoded["input_ids"]))

pair = tokenizer("How are you?", "I'm fine, thank you!")
print(pair)
pair_pt = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
print(pair_pt)
enc=tokenizer("How are you?", "I'm fine, thank you!",padding=True, return_tensors="pt")
print(enc)

long_text = "This is a very " + "very " * 60 + "long sentence."
enc = tokenizer(long_text, truncation=True)
print(enc["input_ids"])

enc = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,          # 补齐
    truncation=True,       # 砍长
    max_length=5,          # 目标长度
    return_tensors="pt",
)
print(enc)
enc = tokenizer("How are you?")
print(enc["input_ids"])                    # [101, 1731, 1132, 1128, 136, 102]
print(tokenizer.decode(enc["input_ids"]))  # '[CLS] How are you? [SEP]'
#   add_special_tokens=False 可关掉自动加 [CLS]/[SEP]


sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
_ = BertTokenizer.from_pretrained("bert-base-cased")
_ = AutoTokenizer.from_pretrained("bert-base-cased")

# 第 4 部分：批处理 与 attention_mask 为什么至关重要

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],   # 最后一个是填充
]
print(model(torch.tensor(batched_ids)).logits)
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],   # 最后一个填充位置置 0，让模型忽略
]
out = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(out.logits)

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

base = AutoModel.from_pretrained(checkpoint)
outputs = base(**inputs)                       # **inputs 把字典按参数名展开传入
print(outputs.last_hidden_state.shape)

clf = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = clf(**inputs)
print(outputs.logits.shape)                    # torch.Size([2, 2])  → 2条样本 × 2个标签
print(outputs.logits)

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

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
sequences = ["I've been waiting for a HuggingFace course my whole life.", "I hate this!"]
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokens=tokenizer(sequences,padding=True, truncation=True, return_tensors="pt")
output=model(**tokens)
probs=torch.nn.functional.softmax(output.logits,dim=-1)
labels=[model.config.id2label[i] for i in probs.argmax(dim=-1).tolist()]
for text, prob, label in zip(sequences, probs, labels):
    print(f"{text!r:60s} -> {label} ({prob.max().item():.4f})")

classifier=pipeline("sentiment-analysis")
print(classifier(sequences))
