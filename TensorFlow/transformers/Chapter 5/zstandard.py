from datasets import load_dataset

data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
print(pubmed_dataset[0])

import psutil
# Process.memory_info 以字节为单位，因此需要转换为兆字节
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
print(f"Dataset size in bytes: {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
print(f"Dataset size (cache file) : {size_gb:.2f} GB")

#诸如Dataset.map()并行处理之类的操作无需移动或复制数据集。
# 这些功能在底层都由Apache Arrow内存格式和pyarrow库实现，从而实现了极快的数据加载和处理速度。
import timeit
batch_size = 1000
def iterate_dataset():
    for idx in range(0, len(pubmed_dataset), batch_size):
        _ = pubmed_dataset[idx:idx + batch_size]
time = timeit.timeit(
    stmt=iterate_dataset,
    number=1
)
size_gb = 0.5  # 这里换成你的数据集实际大小，单位 GB
print(
    f"Iterated over {len(pubmed_dataset)} examples "
    f"(about {size_gb:.1f} GB) in "
    f"{time:.1f}s, i.e. {size_gb / time:.3f} GB/s"
)

#流式数据集
#要启用数据集流式传输，只需将streaming=True参数传递给load_dataset()函数即可。
# 例如，让我们再次加载 PubMed Abstracts 数据集
pubmed_dataset_streamed=load_dataset('json',data_files=data_files,split='train',streaming=True)
next(iter(pubmed_dataset_streamed))

from transformers import  AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset=pubmed_dataset_streamed.map(lambda x:tokenizer(x['text']))
next(iter(tokenized_dataset))
#为了加快流式标记化速度，您可以传递参数batched=True，
# 正如我们在上一节中看到的。它将分批处理示例；默认批次大小为 1000，
# 也可以使用参数指定batch_size。

#`shuffle` 来打乱流式数据集IterableDataset.shuffle()，
# 但与Dataset.shuffle()此不同的是，它只会打乱预定义数组中的元素buffer_size
shuffled_dataset =pubmed_dataset_streamed.shuffle(buffer_size=10_000,seed=42)
next(iter(shuffled_dataset))
#您还可以使用 `select`IterableDataset.take()和 ` IterableDataset.skip()select`
# 函数从流式数据集中选择元素，它们的作用与 `select` 类似Dataset.select()。
# 例如，要选择 PubMed Abstracts 数据集中的前 5 个示例

dataset_head=pubmed_dataset_streamed.take(5)
list (dataset_head)
# 跳过前 1000 个样本，并将剩余的样本包含在训练集中
train_dataset=shuffled_dataset.skip(1000)
# 取前 1000 个样本作为验证集
validation_dataset=shuffled_dataset.take(1000)

law_dataset_streamed = load_dataset(
    "json",
    data_files="https://the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
    split="train",
    streaming=True,
)
next(iter(law_dataset_streamed))


from itertools import islice
from datasets import interleave_datasets

combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])
list(islice(combined_dataset, 2))

base_url = "https://the-eye.eu/public/AI/pile/"
data_files = {
    "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
    "validation": base_url + "val.jsonl.zst",
    "test": base_url + "test.jsonl.zst",
}
pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
next(iter(pile_dataset["train"]))

