#
# CSV & TSV	csv	load_dataset("csv", data_files="my_file.csv")
# Text files	text	load_dataset("text", data_files="my_file.txt")
# JSON & JSON Lines	json	load_dataset("json", data_files="my_file.jsonl")
# Pickled DataFrames	pandas	load_dataset("pandas", data_files="my_dataframe.pkl")

from datasets import load_dataset
squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
squad_it_dataset["train"][0]

data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
squad_it_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['title', 'paragraphs'],
#         num_rows: 442
#     })
#     test: Dataset({
#         features: ['title', 'paragraphs'],
#         num_rows: 48
#     })
# })
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

#加载远程数据集
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")


