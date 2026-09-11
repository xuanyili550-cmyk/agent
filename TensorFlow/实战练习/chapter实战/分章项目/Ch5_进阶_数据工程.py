"""
================================================================================
 分章项目 · Ch5 进阶 · 数据工程（加载/清洗/切分/落盘/长文切块）
================================================================================
 Ch5_语义搜索引擎.py 讲了“嵌入+检索”；本文件补上 HF Ch5 的另一半——【Datasets 数据工程】：
   ① 多格式加载：csv/json/parquet + 远程 hf:// 直读。
   ② 清洗算子：过滤 filter、改列名、删列、去重、打乱+取子集、train/test 划分。
   ③ 长文切块：tokenizer 的 return_overflowing_tokens + stride 把超长文切成带重叠的块(RAG 前置)。
   ④ 落盘复用：处理好的数据存下来，下次直接读，别每次重算。
 ⚠ 本机说明：HF Ch5 原版用 datasets 库(load_dataset/filter/map/save_to_disk)；但本机 Python3.14
   下 datasets 的 map/pickle 坏了(见项目环境说明)，所以【可跑部分用 pandas 等价实现】(操作/思路
   完全一致)，datasets 原版 API 附在文末 DATASETS_REF 作参考。③ 的 tokenizer 切块是真跑。
 跑：python3 Ch5_进阶_数据工程.py
================================================================================
"""
import pandas as pd


# ==============================================================================
# ① 多格式加载（pandas 直读，等价 load_dataset 的各种 loader）
# ==============================================================================
def load_data():
    # print("=" * 70, "\n① 加载数据(pandas 直读 hf://，等价 load_dataset)\n" + "=" * 70)
    df = pd.read_parquet("hf://datasets/fancyzhx/ag_news/data/train-00000-of-00001.parquet")
    df = df.sample(n=200, random_state=42).reset_index(drop=True)
    print(f"  读入 ag_news：{len(df)} 行，列={list(df.columns)}")
    print(f"  首行：label={df['label'].iloc[0]}  text={df['text'].iloc[0][:50]}...")
    return df


# ==============================================================================
# ② 清洗算子：过滤/改名/删列/去重/打乱取子集/划分
# ==============================================================================
def clean_data(df):
    # print("\n" + "=" * 70, "\n② 清洗算子(filter/rename/drop/dedup/shuffle/split)\n" + "=" * 70)
    n0 = len(df)
    df = df[df["text"].str.len() >= 50].copy()                # filter：只留长度≥50 的
    print(f"  过滤(text≥50字符)：{n0} → {len(df)} 行")
    df = df.rename(columns={"label": "category"})             # 改列名
    df["length"] = df["text"].str.len()                       # 加派生列(等价 map 加字段)
    df = df.drop_duplicates(subset=["text"])                  # 去重
    print(f"  改列名 label→category、加 length 列、去重后：{len(df)} 行，列={list(df.columns)}")
    # 打乱 + train/test 划分(等价 shuffle + train_test_split)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]
    print(f"  打乱后按 8:2 划分：train={len(train)}  test={len(test)}")
    print(f"  各类别分布(train)：{train['category'].value_counts().sort_index().to_dict()}")
    return train, test


# ==============================================================================
# ③ 长文切块：tokenizer return_overflowing_tokens + stride（真跑）
# ==============================================================================
def chunk_long_text():
    # print("\n" + "=" * 70, "\n③ 长文切块(return_overflowing_tokens + stride，带重叠)\n" + "=" * 70)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    long_text = " ".join(f"sentence number {i} about transformers and data." for i in range(40))
    enc = tok(long_text, max_length=32, truncation=True,
              return_overflowing_tokens=True,          # ★超长部分不丢，切成多块
              stride=8)                                 # ★相邻块重叠 8 个 token(别把信息切断)
    chunks = enc["input_ids"]
    print(f"  原文 {len(tok(long_text)['input_ids'])} 个 token → 切成 {len(chunks)} 块(每块≤32，重叠8)")
    print(f"  第0块末尾: ...{tok.decode(chunks[0][-6:])}")
    print(f"  第1块开头(应与上一块重叠): {tok.decode(chunks[1][1:7])}...")
    print("  overflow_to_sample_mapping 记录每块来自原文第几条(切完还能归组)：",
          enc.get("overflow_to_sample_mapping", "N/A"))


# ==============================================================================
# ④ 落盘复用：处理好的数据存下来，下次直读
# ==============================================================================
def save_and_reload(train):
    # print("\n" + "=" * 70, "\n④ 落盘复用(等价 save_to_disk/load_from_disk)\n" + "=" * 70)
    path = "/tmp/ch5_train.parquet"
    train.to_parquet(path)                                    # 存(列式，比 csv 小且快)
    back = pd.read_parquet(path)                              # 下次直接读，不用重算
    print(f"  存到 {path} 再读回：{len(back)} 行，一致={len(back) == len(train)}")
    assert len(back) == len(train)


# ==============================================================================
# datasets 原版 API 参考（本机坏了跑不了，正常环境这样写）
# ==============================================================================
DATASETS_REF = '''
from datasets import load_dataset
ds = load_dataset("csv", data_files="data.csv")                 # 也支持 json/parquet/text
ds = load_dataset("fancyzhx/ag_news")                           # 直接从 Hub 加载
ds = ds.filter(lambda x: len(x["text"]) >= 50)                  # 过滤
ds = ds.rename_column("label", "category").remove_columns(["x"]) # 改名/删列
ds = ds.map(lambda x: {"length": len(x["text"])})              # 加派生列(批量)
split = ds["train"].shuffle(seed=42).train_test_split(test_size=0.2)  # 打乱+划分
ds.save_to_disk("./saved"); load_from_disk("./saved")          # 落盘/重载
stream = load_dataset("c4", "en", split="train", streaming=True)  # 流式(不下全量)
next(iter(stream))                                             # 边下边用，省内存/磁盘
'''

if __name__ == "__main__":
    df = load_data()
    train, test = clean_data(df)
    chunk_long_text()
    save_and_reload(train)
    print("\n" + "=" * 70, "\n datasets 原版 API 参考(本机坏，正常环境可用)：\n" + "=" * 70, DATASETS_REF)
    print("✅ Ch5 进阶跑通：加载 → 清洗(过滤/改名/去重/划分) → 长文切块(overflow+stride) → 落盘复用。")
    # print("面试：Q 长文超模型长度怎么办? Q stride 重叠为什么重要? Q 流式数据集解决什么? (见 ../面试高频题库.py 四)")
