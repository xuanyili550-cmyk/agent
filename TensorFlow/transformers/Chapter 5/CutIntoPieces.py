#对数据进行切片和切块分析
from datasets import load_dataset
import requests
import zipfile
import html
from pathlib import Path
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
zip_path = Path("drugsCom_raw.zip")
data_files = {
    "train": Path("drugsComTrain_raw.tsv"),
    "test": Path("drugsComTest_raw.tsv"),
}

def download():
    print("ZIP 文件不存在，开始下载...")
    response = requests.get(url)
    response.raise_for_status()
    zip_path.write_bytes(response.content)
    print("下载完成")

def extract():
    print("TSV 文件不存在，开始解压...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")
    print("解压完成")
def prepare_data():
    if not zip_path.exists():
        download()
    else:
        print("ZIP 已经存在，跳过下载")
    if not all(path.exists() for path in data_files.values()):
        extract()
    else:
        print("TSV 文件已经存在，跳过解压")

prepare_data()
data_files={"train": "drugsComTrain_raw.tsv","test":"drugsComTest_raw.tsv"}
drug_dataset=load_dataset("csv",data_files=data_files,delimiter="\t")
#抽取一个小的随机样本
drug_sample=drug_dataset["train"].shuffle(seed=42).select(range(1000))
print(drug_sample[:8])
#为了检验该Unnamed: 0列的患者 ID 假设，我们可以使用Dataset.unique()函数来验证每次拆分中 ID 的数量是否与行数匹配
for split in drug_dataset.keys():
    assert len(drug_dataset[split])==len(drug_dataset[split].unique("Unnamed: 0"))
#把Unnamed: 0列名改成更易于理解的名称。我们可以使用该DatasetDict.rename_column()函数
# 一次性重命名两个拆分数据集中的列
drug_dataset=drug_dataset.rename_column(original_column_name="Unnamed: 0",new_column_name="patient_id")
# DatasetDict({
#     train: Dataset({
#         features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
#         num_rows: 161297
#     })
#     test: Dataset({
#         features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
#         num_rows: 53766
#     })
# })
#condition使用归一化所有标签Dataset.map()该函数可以应用于每个拆分的所有行drug_dataset
def filter_nones(x):
    return x["condition"] is not None
drug_dataset = drug_dataset.filter(filter_nones)
def lowercase_condition(example):
    return {"condition":example['condition'].lower()}
drug_dataset.map(lowercase_condition)

#创建新列
def  compute_review_length ( example ):
    return {'review_length':len(example["review"].split())}
#函数不同lowercase_condition()，compute_review_length()它返回一个字典，
# 该字典的键与数据集中的任何列名都不对应
drug_dataset = drug_dataset.map (compute_review_length)
 # 检查第一个训练样本
drug_dataset[ "train" ][ 0 ]

#Dataset.sort()以查看极端值的情况
drug_dataset["train"].sort("review_length")[:3]
# {'patient_id': [103488, 23627, 20558],
#  'drugName': ['Loestrin 21 1 / 20', 'Chlorzoxazone', 'Nucynta'],
#  'condition': ['birth control', 'muscle spasm', 'pain'],
#  'review': ['"Excellent."', '"useless"', '"ok"'],
#  'rating': [10.0, 1.0, 6.0],
#  'date': ['November 4, 2008', 'March 24, 2017', 'August 20, 2016'],
#  'usefulCount': [5, 2, 10],
#  'review_length': [1, 1, 1]}
#该Dataset.filter()函数删除字数少于 30 个字的评论

def filter_max(x):
    return x["review_length"] >30
drug_dataset=drug_dataset.filter(filter_max)
print(drug_dataset.num_rows)
#Dataset.map()来对语料库中的所有 HTML 字符进行转义
def filter_unescape(x):
    return {
        "review": html.unescape(x["review"])
    }
drug_dataset = drug_dataset.map(filter_unescape)

#map() 方法的强大功能
#Dataset.map()方法接受一个batched参数，如果将其设置为 `true` True，
# 则会一次性向 `map` 函数发送一批示例（批次大小可配置，但默认值为 1000）。
# 例如，之前用于取消转义所有 HTML 的 `map` 函数运行时间较长（您可以从进度条中查看耗时）。
# 我们可以使用列表推导式同时处理多个元素来加快速度
def unescape_reviews(x):
    return {
        "review":[
            html.unescape(o) for o in x["review"]
        ]
    }
# 选项	快速分词器	慢速分词器
# batched=True	10.8秒	4分41秒
# batched=False	59.2秒	5分3秒
new_drug_dataset = drug_dataset.map (
     unescape_reviews, batched= True
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)

# 选项	快速分词器	慢速分词器
# batched=True	10.8秒	4分41秒
# batched=False	59.2秒	5分3秒
# batched=True，num_proc=8	6.52秒	41.3秒
# batched=False，num_proc=8	9.49秒	45.2秒
slow_tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-cased", use_fast=False)
def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)
tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)

#如何工作的
def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
#训练集中的第一个示例变成了两个特征，因为它被分词到了超过我们指定的最大词元数：
# 第一个特征长度为 128，第二个特征长度为 49。现在让我们对数据集中的所有元素执行此操作
result = tokenize_and_split(drug_dataset["train"][0])
[len(inp) for inp in result["input_ids"]]
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)

#ArrowInvalid: Column 1 named condition expected length 1463 but got length 1000

tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
len(tokenized_dataset["train"]), len(drug_dataset["train"])
#我们还可以通过使旧列与新列大小相同来解决长度不匹配的问题。
# 为此，我们需要用到overflow_to_sample_mapping分词器
# 在设置参数时返回的字段return_overflowing_tokens=True。
# 该字段提供了一个从新特征索引到其来源样本索引的映射。
# 利用这个映射，我们可以将原始数据集中的每个键与一个大小合适的值列表关联起来
# ，方法是重复每个样本的值，重复次数与它生成新特征的次数相同
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # 提取新旧索引之间的映射
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
# DatasetDict({
#     train: Dataset({
#         features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
#         num_rows: 206772
#     })
#     test: Dataset({
#         features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
#         num_rows: 68876
#     })
# })
#访问数据集中的元素时，我们得到的pandas.DataFrame是一个数组而不是字典
drug_dataset.set_format("pandas")
drug_dataset["train"][:3]
#想要的 Pandas 功能
train_df = drug_dataset["train"][:]
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "count": "frequency"})
)
frequencies.head()
#完成 Pandas 分析后，我们始终可以Dataset使用Dataset.from_pandas()以下函数创建一个新对象
from datasets import Dataset

freq_dataset = Dataset.from_pandas(frequencies)
# Dataset({
#     features: ['condition', 'frequency'],
#     num_rows: 819
# })
#将输出格式从drug_dataset重置"pandas"为"arrow"
drug_dataset.reset_format()
#Datasets 提供了一个Dataset.train_test_split()基于著名功能的函数scikit-learn。
# 让我们用它来将训练集拆分为train两个validation部分（我们设置了seed参数以确保结果可复现）
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# 将默认的“test”拆分重命名为“validation”
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# 将“test”数据集添加到我们的`DatasetDict`中
drug_dataset_clean["test"] = drug_dataset["test"]
# DatasetDict({
#     train: Dataset({
#         features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
#         num_rows: 110811
#     })
#     validation: Dataset({
#         features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
#         num_rows: 27703
#     })
#     test: Dataset({
#         features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'review_clean'],
#         num_rows: 46108
#     })
# # })
# Data format	Function
# Arrow	Dataset.save_to_disk()
# CSV	Dataset.to_csv()
# JSON	Dataset.to_json()
drug_dataset_clean.save_to_disk("drug-reviews")
#数据集保存后，我们可以使用load_from_disk()以下函数加载它
from datasets import load_from_disk
drug_dataset_reloaded = load_from_disk("drug-reviews")
#对于 CSV 和 JSON 格式，我们需要将每次拆分存储为单独的文件。
# 一种方法是遍历对象中的键和值DatasetDict
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")
#加载 JSON 文件
data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)
