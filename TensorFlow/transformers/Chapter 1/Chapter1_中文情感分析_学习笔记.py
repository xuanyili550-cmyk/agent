"""
================================================================================
 中文情感分析实战 —— 系统学习笔记（可在你的 M4 Mac 上真跑）
================================================================================
 目标：加载一个中文情感分类模型，输入一句中文，输出「正面/负面/中性」+ 概率。
 这是第 7 部分「手动复现 pipeline」在中文任务上的落地版，完整走一遍：
        清洗文本 → 分词编码 → 模型前向(eval+no_grad) → softmax → argmax → 标签

 与原始笔记的两处修正（重要，手敲时别抄错）：
   ① 方法名：用公开的 tokenizer.encode_plus(...)，不是带下划线的 _encode_plus
              （_encode_plus 是内部私有方法，不该直接调用）
   ② 设备选择：原来的 'cuda' 在 Mac 上永远命中不到（Mac 没有 CUDA）。
              这里改成自动优先 Apple 的 'mps'(Metal GPU)，否则退回 'cpu'。

 术语速记：
   eval 模式        model.eval()：关闭 dropout 等训练专用行为，推理必须开
   no_grad          torch.no_grad()：不记录梯度，省显存、提速，推理必须用
   logits → softmax 原始分数 → 概率；argmax 取概率最大的那个类别的下标
================================================================================
"""

import sys
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax


# ------------------------------------------------------------------------------
# 1) 选设备：Apple Silicon 优先用 mps(Metal GPU)，没有再退 cpu
# ------------------------------------------------------------------------------
if torch.cuda.is_available():          # NVIDIA GPU（Linux/Win 上才有）
    device = "cuda"
elif torch.backends.mps.is_available():  # ★ Apple Silicon 的 GPU 加速后端
    device = "mps"
else:
    device = "cpu"
print(f"使用设备: {device}")


# ------------------------------------------------------------------------------
# 2) 载入预训练模型与分词器
#    这是一个已在“基金舆情/金融方向”中文语料上微调过的 BERT 情感分类模型。
#    BertForSequenceClassification = BERT 主干 + 一个分类头（输出每个标签的分数）。
# ------------------------------------------------------------------------------
pretrained = "sanshizhang/Chinese-Sentiment-Analysis-Fund-Direction"
model = BertForSequenceClassification.from_pretrained(pretrained)
tokenizer = BertTokenizer.from_pretrained(pretrained)

model = model.to(device)   # 把模型权重搬到目标设备
model.eval()               # ★ 切到评估模式：关闭 dropout，保证推理结果稳定


# ------------------------------------------------------------------------------
# 3) 预测函数：输入一句中文，返回 (各类别概率, 预测类别下标)
# ------------------------------------------------------------------------------
def predict_sentiment(text: str):
    # ---- 编码：把中文文本变成模型输入张量 ----
    # 现代写法：直接调用 tokenizer(...)（即 __call__），它支持下列全部参数。
    # 说明：老教程里的 tokenizer.encode_plus(...) 在 transformers 5.x 已被移除，
    #       原始笔记里的 _encode_plus 是私有方法，能跑但不推荐——统一用下面这种。
    # encode 的每个参数含义（你要的“逐个参数”）：
    encoding = tokenizer(
        text,
        max_length=512,               # 最大长度(token)；BERT 上限就是 512
        add_special_tokens=True,      # 自动加 [CLS] 句首、[SEP] 句尾
        return_token_type_ids=False,  # 单句分类用不到句子A/B区分，关掉省事
        padding="max_length",         # 补齐到 max_length（单句预测其实可用 padding=False 更省）
        truncation=True,              # 超过 512 就截断，防报错
        return_attention_mask=True,   # 返回 attention_mask，告诉模型哪些是真 token
        return_tensors="pt",          # 直接返回 PyTorch 张量
    )

    # 取出两个关键输入，并搬到与模型相同的设备
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # ---- 前向推理：不计算梯度 ----
    with torch.no_grad():                        # ★ 推理不需要梯度，省内存又更快
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = softmax(outputs.logits, dim=1)   # logits → 概率（每行加起来=1）

    # argmax 取概率最大的类别下标；.cpu() 搬回 CPU 才能转成 numpy/python 数
    pred_idx = torch.argmax(probs, dim=1).cpu().numpy()[0]
    return probs, pred_idx


# ------------------------------------------------------------------------------
# 4) 文本清洗：只保留中文、数字、字母、常见标点，去掉乱七八糟的特殊字符
#    正则说明：[^...] 表示“匹配不在这个集合里的字符”，然后统一替换成空。
#      一-鿿 是中文汉字的 Unicode 区间。
# ------------------------------------------------------------------------------
def clean_text(text: str) -> str:
    return re.sub(
        r"[^一-鿿\d.a-zA-Z%+\-。！？，、；：（）【】《》“”‘’]",
        "",
        text,
    )


# ------------------------------------------------------------------------------
# 5) 标签映射：模型输出的下标 → 人类可读标签
#    （下标含义取决于该模型训练时的定义，这里按原笔记：0负 1正 2中）
# ------------------------------------------------------------------------------
SENTIMENT_LABELS = {0: "negative", 1: "positive", 2: "neutral"}


def analyze(text: str) -> None:
    text = clean_text(text)
    probs, pred_idx = predict_sentiment(text)
    label = SENTIMENT_LABELS[pred_idx]
    confidence = probs[0][pred_idx].item()   # .item() 把单元素张量取成 python float
    print(f"文本: {text}")
    print(f"情感: {label}，置信度: {confidence:.4f}")


# ------------------------------------------------------------------------------
# 6) 两种用法
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]                       # 忽略脚本名，取命令行后面的词
    if args:
        # 用法A：命令行传入，如  python3 本文件.py 这家公司业绩大涨
        analyze(" ".join(args))
    else:
        # 用法B：直接运行文件，跑几条内置示例
        for s in [
            "这家公司业绩大涨，前景一片光明！",
            "基金又跌了，亏麻了，再也不买了。",
            "今天大盘震荡，走势不明朗。",
        ]:
            analyze(s)
            print("-" * 40)
