"""
================================================================================
 Chapter 5 · 进阶补充（补齐审计发现的缺口）—— 可直接运行
================================================================================
 这是对 Chapter5_数据集与语义搜索_学习笔记.py 的补充，专门补之前没讲/讲浅的点：
   1) ★长文本 overflow 切分：一条长文切成多个特征 + 随之而来的 ArrowInvalid 报错 + 两种解法
   2) ★数据集落盘：save_to_disk/load_from_disk、to_csv、to_json(逐 split 存 jsonl 再读回)
   3) num_proc 多进程加速 map
   4) html.unescape 清洗 HTML 转义
   5) 下载+解压+幂等跳过 的工程流程(用本地 zip 演示，不联网)
   6) 多字段拼接成待嵌入文本(语义搜索前的关键预处理)
   7) batched=True 的机制(一次一批、配合列表推导)

 直接运行：python3 Chapter5_进阶补充_学习笔记.py
 只用小合成数据 + 一个小分词器(bert-base-uncased，已缓存)，几秒跑完、不下大数据。
================================================================================
"""

import html
import os
import tempfile

from datasets import Dataset, DatasetDict


def banner(t):
    print("\n" + "=" * 72 + f"\n {t}\n" + "=" * 72)


# ==============================================================================
# 1) ★长文本 overflow 切分 + ArrowInvalid 报错 + 两种解法
# ==============================================================================
banner("1) 长文本 overflow 切分：一条长文 → 多个特征")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
# 造一条“超长”文本 + 一个短的，故意让长的被切成多块
ds = Dataset.from_dict({
    "id": [0, 1],
    "text": ["word " * 60, "short text"],   # 第0条很长(60词)，第1条很短
})

# return_overflowing_tokens=True：超过 max_length 的部分不丢，而是另起一个“特征(块)”
def split_long(examples):
    out = tok(examples["text"], max_length=16, truncation=True,
              return_overflowing_tokens=True)
    return out

# ★直接 map 会报 ArrowInvalid！因为一条长文变成了多块，新特征行数(比如 6) 与旧列(id: 2 行)对不上。
try:
    ds.map(split_long, batched=True)
    print("  (未报错)")
except Exception as e:
    print(f"  直接 map 报错：{type(e).__name__} —— 长文被切成多块，行数和旧列 id 对不上")

# 解法一：remove_columns 把旧列删掉，只保留切好的新特征
split_a = ds.map(split_long, batched=True, remove_columns=ds.column_names)
print(f"  解法一(remove_columns)：2 条 → {len(split_a)} 个特征(块)")

# 解法二：用 overflow_to_sample_mapping 把旧列“按重复次数”对齐到新特征(保留 id 溯源)
def split_keep_id(examples):
    out = tok(examples["text"], max_length=16, truncation=True,
              return_overflowing_tokens=True)
    sample_map = out.pop("overflow_to_sample_mapping")   # 每个新块来自第几条原样本
    out["id"] = [examples["id"][i] for i in sample_map]  # 用它把 id 重复对齐
    return out

split_b = ds.map(split_keep_id, batched=True, remove_columns=ds.column_names)
print(f"  解法二(overflow_to_sample_mapping)：{len(split_b)} 个特征，各自的原始 id = {split_b['id']}")
print("  为什么会报错：一条样本被切成 N 块 → 新表行数变多 → 和没变的旧列长度不一致 → Arrow 拒绝。")


# ==============================================================================
# 2) ★数据集落盘：save_to_disk / load_from_disk / to_csv / to_json
# ==============================================================================
banner("2) 处理好的数据集怎么存起来复用")
work = tempfile.mkdtemp()
small = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 0]})

# 方式一：save_to_disk / load_from_disk —— 存的是 Arrow 格式，最快、最完整
small.save_to_disk(os.path.join(work, "ds_arrow"))
from datasets import load_from_disk
reloaded = load_from_disk(os.path.join(work, "ds_arrow"))
print("  save_to_disk/load_from_disk：读回", len(reloaded), "条")

# 方式二：导出 CSV / JSON(注意 DatasetDict 要逐个 split 存)
#   小坑：small.to_csv(...) 也可以，但 pandas 写 CSV 时会试探性 import zstandard，
#   而本目录里有个同名文件 zstandard.py 会“抢先被导入”(命名冲突)，所以这里只演示 to_json。
#   真实写法：small.to_csv("x.csv", index=False)
small.to_json(os.path.join(work, "small.jsonl"))     # 每行一个 json 对象(jsonl)
print("  to_json 已导出到:", work, " (to_csv 同理，见上方注释里的命名冲突小坑)")

# DatasetDict 逐 split 存 jsonl，再用 load_dataset 读回
dd = DatasetDict({"train": small, "test": small})
for split, d in dd.items():
    d.to_json(os.path.join(work, f"{split}.jsonl"))
from datasets import load_dataset
back = load_dataset("json", data_files={s: os.path.join(work, f"{s}.jsonl")
                                        for s in dd})
print("  逐 split 存 jsonl 再 load_dataset 读回：", {s: len(back[s]) for s in back})


# ==============================================================================
# 3) num_proc 多进程加速 map
# ==============================================================================
banner("3) num_proc：多进程并行处理(大数据集提速关键)")
big = Dataset.from_dict({"text": ["hello world"] * 1000})
# num_proc=4：把 map 切成 4 个进程并行跑。小数据看不出提速(反而有进程开销)，大数据才划算。
out = big.map(lambda x: {"n": len(x["text"].split())}, num_proc=2)
print(f"  num_proc=2 处理 {len(out)} 条完成(大数据集上能显著提速；小数据反而慢，因有进程开销)")
print("  经验：几十万条以上 + 纯 CPU 处理(分词/清洗) 才用 num_proc；配 batched=True 效果更好。")


# ==============================================================================
# 4) html.unescape 清洗 HTML 转义(真实文本常见)
# ==============================================================================
banner("4) html.unescape：把 &amp; &lt; &#39; 还原成 & < '")
raw = "Tom &amp; Jerry &lt;3 it&#39;s &quot;great&quot;"
print("  清洗前:", raw)
print("  清洗后:", html.unescape(raw))
# 常配合 map 用：ds = ds.map(lambda x: {"text": html.unescape(x["text"])})


# ==============================================================================
# 5) 下载 + 解压 + 幂等跳过 的工程流程(用本地 zip 演示，不联网)
# ==============================================================================
banner("5) 下载解压的工程套路(幂等：已存在就跳过)")
import zipfile
from pathlib import Path

data_dir = Path(work) / "downloaded"
data_dir.mkdir(exist_ok=True)
zip_path = data_dir / "demo.zip"
extracted = data_dir / "demo.txt"

# 造一个本地 zip 冒充“下载下来的压缩包”(真实里是 requests.get(url) 写文件)
if not zip_path.exists():                      # ← 幂等：文件在就不重复下
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("demo.txt", "hello from zip")
    print("  下载(此处为本地造)zip 完成")
else:
    print("  zip 已存在，跳过下载")

if not extracted.exists():                     # ← 幂等：解压过就不重复解
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    print("  解压完成 →", extracted.read_text())
else:
    print("  已解压，跳过")
print("  真实写法：requests.get(url,stream=True) 写盘 → zipfile 解压 → Path.exists() 判断幂等。")


# ==============================================================================
# 6) 多字段拼接成待嵌入文本(语义搜索前的关键预处理)
# ==============================================================================
banner("6) 拼接多字段 → 一段用于嵌入/检索的文本")
issues = Dataset.from_dict({
    "title": ["Login bug", "Slow query"],
    "body": ["cannot log in on mobile", "the report page takes 30s"],
    "comments": [["me too", "same here"], ["confirmed"]],   # 评论是列表
})

# 语义搜索前，要把 title+body+comments 拼成一段完整文本，模型才好嵌入这“整条 issue”
def concatenate_text(example):
    return {"text": example["title"] + " \n "
                    + example["body"] + " \n "
                    + " ".join(example["comments"])}

issues = issues.map(concatenate_text)
print("  拼接后第 0 条 text:")
print("   ", repr(issues[0]["text"]))
print("  为什么拼：嵌入的是“整条 issue 的语义”，标题/正文/评论都带信息，拼一起检索才全。")
print("  (评论是列表列，真实数据里常先用 pandas explode 把它展开成多行，再拼接)")


# ==============================================================================
# 7) batched=True 的机制(不只是“更快”)
# ==============================================================================
banner("7) batched=True 到底做了什么")
print("""
  · batched=False(默认)：map 一次给函数“一条”样本，函数处理单条。
  · batched=True：map 一次给函数“一批”(默认 1000 条)，函数收到的是“列的列表”，
      要用列表推导/一次性调用来处理整批 → 减少 Python 调用开销。
  · 对“快速分词器”尤其关键：它能一次并行分词一整批，比逐条快几十倍。
      官方对比(整本书量级)：快速分词器 batched≈10 秒 vs 慢速逐条≈4 分 41 秒。
  · 写法差异：
      batched=False:  def f(ex): return {"n": len(ex["text"].split())}       # ex 是一条
      batched=True :  def f(ex): return {"n": [len(t.split()) for t in ex["text"]]}  # ex["text"] 是一列
""")

print("✅ 全部跑完。这些是 Chapter 5 之前没覆盖到的“数据工程”实战点。")
