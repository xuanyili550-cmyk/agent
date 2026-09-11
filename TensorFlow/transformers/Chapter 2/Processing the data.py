import torch
from torch.optim import AdamW
from transformers import AutoTokenizer,AutoModelForSequenceClassification
#批次上训练序列分类器
checkpoint = "bert-base-uncased"
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]

tokenizer=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)

batch=tokenizer(sequences,padding=True,truncation=True,return_tensors='pt')  # 是 return_tensors（带 s）！少 s 会返回列表→报 'list' object has no attribute 'size'
batch['labels']=torch.tensor([1,1])
optimizer=AdamW(model.parameters())
loss = model(**batch).loss 
loss.backward()
optimizer.step()


#Datasets 库MRPC 数据集
from  datasets import load_dataset
# 新版 datasets 要求 namespace/name 格式，裸名 "glue" 已不支持；GLUE 现在在 nyu-mll/glue
raw_datasets = load_dataset( "nyu-mll/glue" , "mrpc" )
# DatasetDict({
#     train: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 3668
#     })
#     validation: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 408
#     })
#     test: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 1725
#     })
# })
raw_train_dataset=raw_datasets['train']
print(raw_train_dataset[0])
# {'sentence1': 'Amrozi accused his brother , whom he called " the witness " , '
#               'of deliberately distorting his evidence .',
#  'sentence2': 'Referring to him as only " the witness " , '
#               'Amrozi accused his brother of deliberately distorting his evidence .',
#  'label': 1, 'idx': 0}
#检查features数据表的列类型raw_train_dataset
#label类型为ClassLabel，整数到标签名称的映射存储在names文件夹中。0对应于not_equivalent，1对应于equivalent
print(raw_train_dataset.features)
# {'sentence1': Value('string'), 'sentence2': Value('string'),
#  'label': ClassLabel(names=['not_equivalent', 'equivalent']),
#  'idx': Value('int32')}

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 新版 datasets(5.x) 里 dataset["列名"] 返回 Column 惰性对象，不是 list；
# tokenizer 不认，需要用 list(...) 转成真正的 list[str] 再传入。
tokenized_sentences_1= tokenizer(list(raw_datasets[ "train" ][ "sentence1" ]))
tokenized_sentences_2= tokenizer(list(raw_datasets[ "train" ][ "sentence2" ]))

inputs = tokenizer("This is the first sentence.", "This is the second one.")
#{'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996,
# 2117, 2028, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
#['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]',
# 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']

tokenized_dataset = tokenizer(
    list(raw_datasets["train"]["sentence1"]),
    list(raw_datasets["train"]["sentence2"]),
    padding=True,
    truncation=True,
)

def tokenize_function(example):
    return tokenizer(list(example["sentence1"]), list(example["sentence2"]), truncation=True)
#我们batched=True在调用函数时使用了 `--tokenization` 参数map，这样该函数就能一次性应用于数据集中的多个元素
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

#DataCollatorWithPadding。实例化该函数时，它会接收一个分词器
# （用于确定要使用的填充标记，以及模型期望填充位于输入的左侧还是右侧）

from transformers import DataCollatorWithPadding
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
# DataCollatorWithPadding(tokenizer=BertTokenizer(name_or_path='bert-base-uncased', vocab_size=30522, model_max_length=512, padding_side='right',
# truncation_side='right', special_tokens={'unk_token': '[UNK]',
# 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]',
# 'mask_token': '[MASK]'}, added_tokens_decoder={
# 	0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# 	100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# 	101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# 	102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# 	103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# }), padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt')
print(data_collator)
samples=tokenized_datasets['train'][8:]
samples={k:v for k,v in samples.items() if k not in ["idx" , "sentence1" , "sentence2" ]}
[ len (x) for x in samples[ "input_ids" ]]
#[50, 59, 47, 67, 59, 50, 62, 32]
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
# {'attention_mask': torch.Size([8, 67]),
#  'input_ids': torch.Size([8, 67]),
#  'token_type_ids': torch.Size([8, 67]),
#  'labels': torch.Size([8])}

