from datasets import load_dataset,  Dataset
#加载和准备数据集
issues_dataset=load_dataset('lewtun/github-issues',split='train')

def issues_request(x):
    return (x["is_pull_request"] == False and len(x["comments"]) > 0)

issues_dataset = issues_dataset.filter(issues_request)
#其中大部分列对于构建搜索引擎来说并非必要。从搜索的角度来看，
# 信息量最大的列是 `<title>`、`<title>`title和body`<title> comments`，
# 而 ` <title>` 则html_url提供了指向源问题的链接。
# 让我们使用 ` Dataset.remove_columns()<function>` 函数删除其余列
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
issues_dataset.set_format("pandas")
df=issues_dataset[:]
df ["comments"][0].tolist()
comments_df = df.explode("comments", ignore_index=True)
comments_df.head(4)


comments_dataset=Dataset.from_pandas(comments_df)
# Dataset({
#     features: ['html_url', 'title', 'comments', 'body'],
#     num_rows: 2842
# })
def comment_length(x):
    return {"comment_length": len(x["comments"].split())}

comments_dataset = comments_dataset.map(comment_length)
def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_text)
#创建文本嵌入
from transformers import AutoTokenizer, AutoModel
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

import torch

# 原课程写死 cuda，Mac 没有 N卡会崩；自动选 mps(Apple GPU) 否则 cpu
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
#对模型的输出执行CLS 池化，其中我们只需收集特定词的最后一个隐藏状态[CLS]

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)
embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape
#torch.Size([1, 768])
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

#FAISS 的基本思想是创建一种称为索引的特殊数据结构，它允许我们找到与输入嵌入相似的嵌入。
# 在  Datasets 中创建 FAISS 索引很简单——我们使用Dataset.add_faiss_index()相应的函数并指定要索引的数据集列：
embeddings_dataset.add_faiss_index(column="embeddings")
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape
#torch.Size([1, 768])
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
#Dataset.get_nearest_examples()函数返回一个元组，其中包含对查询和文档重叠程度进行排名的分数，
# 以及一组相应的样本（此处为 5 个最佳匹配项）。让我们将它们收集起来，pandas.DataFrame以便于排序：
import pandas as pd

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()