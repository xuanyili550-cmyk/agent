"""
================================================================================
 Chapter 3 · 使用预训练模型 (Using pretrained models) —— 系统学习笔记
================================================================================
 配套 HuggingFace 课程「Using pretrained models」章节。
 核心：不训练、不微调，直接从 Model Hub 拿别人训练好的模型来用。

 本章比 Ch1/Ch2 短，就两件事：
   ① 用 pipeline 一行调用 Hub 上的预训练模型（以 fill-mask 完形填空为例）
   ② 两种加载模型的方式：Auto 类（通用）vs 专用类（如 CamembertForMaskedLM）

 本文件结构：
   第 1 部分  Model Hub 和预训练模型：为什么能“拿来即用”
   第 2 部分  fill-mask 任务：BERT 预训练时到底学了什么
   第 3 部分  pipeline 用法 + 关键参数（top_k / targets / mask token）
   第 4 部分  Auto 类 vs 专用类：该用哪个（结论：优先 Auto）
   第 5 部分  AutoModelFor* 全家桶：不同任务对应不同的“头”
   第 6 部分  怎么挑一个合适的模型（任务 / 语言 / 大小）
   第 7 部分  ★常见报错速查（mask token 不统一、缺 sentencepiece 等）

 术语速记：
   Model Hub        HuggingFace 的模型仓库(huggingface.co/models)，几十万个模型
   checkpoint       一个具体模型的名字/路径，如 "camembert-base"、"bert-base-uncased"
   fill-mask        完形填空：把句子里某个词挖掉(<mask>)，让模型猜是什么
   MLM              Masked Language Modeling，掩码语言建模，BERT 的预训练方式
   masked/mask token 被挖掉的占位符。BERT 用 [MASK]，RoBERTa/Camembert 用 <mask>
================================================================================
"""

# ==============================================================================
# 第 1 部分：Model Hub 和预训练模型
# ==============================================================================
# 预训练模型 = 别人已经在海量语料上花大量算力训练好的模型，权重公开在 Model Hub。
# 你只要知道它的名字(checkpoint)，一行代码就能下载来用，无需自己训练。
#
# 例子里的 "camembert-base" 是一个法语版 BERT（camembert 是法国奶酪，法语社区起的名）。
# 常见 checkpoint：
#   bert-base-uncased        英文 BERT（不区分大小写）
#   bert-base-chinese        中文 BERT
#   camembert-base           法语 BERT
#   distilbert-base-uncased  英文 BERT 的精简版（更小更快）
#   roberta-base             英文 RoBERTa（BERT 的改进版）


# ==============================================================================
# 第 2 部分：fill-mask 任务 —— BERT 预训练时学了什么
# ==============================================================================
# BERT 这类模型的预训练方式就是“完形填空”(MLM)：把句子里 15% 的词随机挖掉，
# 让模型根据上下文猜被挖的词。猜多了，它就学会了语言规律和常识。
#
# 所以一个“裸”的预训练 BERT，天生就会 fill-mask（这是它的原生技能，不用微调）。
# 这也是为什么 Ch2 里 bert-base-uncased 加载成“分类模型”时要新建分类头——
# 分类不是它的原生技能，fill-mask 才是。

from transformers import pipeline

# fill-mask：把 <mask> 交给模型猜。camembert 用 <mask> 作为掩码符号。
fill_mask = pipeline("fill-mask", model="camembert-base")
results = fill_mask("Le camembert est <mask> :)")   # “这个卡门贝奶酪很___”
for r in results:
    print(f"  {r['token_str']:12s} 概率 {r['score']:.3f} → {r['sequence']}")
# 输出（法语）：délicieux(美味) 0.49 / excellent 0.11 / succulent 0.03 ...
# 模型靠上下文“Le camembert est ___ :)”猜出这里该填正面形容词。


# ==============================================================================
# 第 3 部分：pipeline 用法 + 关键参数
# ==============================================================================
# ---- top_k：返回概率最高的前 k 个候选（默认 5）----
# fill_mask("Le camembert est <mask> :)", top_k=3)   # 只看前 3 个
#
# ---- targets：只在指定候选词里选（限定答案范围）----
# fill_mask("This is a <mask> movie.", targets=["good", "bad"])   # 只在 good/bad 里挑
#
# ---- ★ mask token 因模型而异（最容易踩的坑，见第7部分）----
#   bert-base-uncased / bert-base-chinese  → 用 [MASK]
#   camembert / roberta                    → 用 <mask>
#   用错了会报错或结果乱。稳妥做法：用 tokenizer.mask_token 动态取，别写死。
#   例：mask = fill_mask.tokenizer.mask_token
#       fill_mask(f"The capital of France is {mask}.")


# ==============================================================================
# 第 4 部分：Auto 类 vs 专用类 —— 该用哪个
# ==============================================================================
# 加载同一个模型，有两种写法，效果完全一样：

# 写法A：专用类（名字写死了架构）
from transformers import CamembertTokenizer, CamembertForMaskedLM
tokenizer_a = CamembertTokenizer.from_pretrained("camembert-base")
model_a = CamembertForMaskedLM.from_pretrained("camembert-base")

# 写法B：Auto 类（自动根据 checkpoint 判断该用哪个架构）★推荐
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer_b = AutoTokenizer.from_pretrained("camembert-base")
model_b = AutoModelForMaskedLM.from_pretrained("camembert-base")

# 为什么优先用 Auto 类？
#   * 换模型时代码不用改：把 "camembert-base" 换成 "bert-base-uncased"，
#     AutoModelForMaskedLM 会自动加载 BERT 架构；而专用类 CamembertForMaskedLM
#     只能加载 camembert，换模型就得改类名。
#   * 更通用、更少出错。专用类只在你明确知道架构、且想显式表达时才用。
# 结论：99% 情况用 Auto 类。


# ==============================================================================
# 第 5 部分：AutoModelFor* 全家桶 —— 不同任务用不同的“头”
# ==============================================================================
# 同一个预训练主干(如 bert)，套不同的“任务头”，就能干不同的活。
# Auto 类按任务命名，选对了就自动装上对应的头：
#
#   AutoModel                          只出特征(隐藏状态)，无任务头（Ch1 学过）
#   AutoModelForMaskedLM               完形填空 fill-mask（本章）
#   AutoModelForSequenceClassification 文本分类 / 情感 / 句子对（Ch2 微调用的）
#   AutoModelForTokenClassification    命名实体识别 NER（逐 token 分类）
#   AutoModelForQuestionAnswering      抽取式问答
#   AutoModelForCausalLM               文本生成(GPT类，预测下一个词)
#   AutoModelForSeq2SeqLM              翻译 / 摘要(编码器-解码器)
#
# 记法：AutoModelFor + 任务名。任务变了就换这个类，checkpoint 常常不用变。


# ==============================================================================
# 第 6 部分：怎么挑一个合适的模型
# ==============================================================================
# 去 huggingface.co/models，按三个维度筛：
#   1) 任务(Task)   左侧筛选 fill-mask / text-classification / translation 等
#   2) 语言(Language) 中文任务别用纯英文模型；多语言选 multilingual/xlm-roberta
#   3) 大小/速度    base < large；要快/上端侧选 distil* / tiny / mobile 版
# 还要看：下载量(热门通常更靠谱)、模型卡(README)说明的用途和限制、许可证。
#
# 你机器上已下载可直接用的：
#   bert-base-uncased（英文）、bert-base-chinese（中文）、camembert-base（法语）、
#   distilgpt2（英文生成）、distilbert-...-sst-2（英文情感）


# ==============================================================================
# 第 7 部分：★常见报错速查
# ==============================================================================
# ┌────────────────────────────────────────────┬──────────────────────────────┐
# │ 现象 / 报错                                    │ 原因 & 解决                   │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ fill-mask 报错说找不到 mask，或结果全乱          │ mask token 用错了。            │
# │                                              │ bert 用 [MASK]，camembert/    │
# │                                              │ roberta 用 <mask>。            │
# │                                              │ → 用 tokenizer.mask_token 动态取│
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ ImportError/需要 sentencepiece                │ camembert/xlm 等用 SentencePiece│
# │                                              │ 分词器。→ pip install          │
# │                                              │ sentencepiece（你已装）        │
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ "Some weights were not initialized"          │ 正常提示：加载的任务头和         │
# │ 或 newly initialized                          │ checkpoint 不完全匹配。做       │
# │                                              │ fill-mask 用原生 MLM 模型就没这提示│
# ├────────────────────────────────────────────┼──────────────────────────────┤
# │ 中文/法语句子结果是乱码或不对                    │ 模型语言不匹配。中文用          │
# │                                              │ bert-base-chinese，别用英文模型 │
# └────────────────────────────────────────────┴──────────────────────────────┘
#
# 小结：本章最核心的一句话——
#   预训练模型自带“完形填空”技能(fill-mask)，用 pipeline 一行即可调用；
#   加载优先用 Auto 类；换任务换 AutoModelFor* 的后缀，换语言换 checkpoint。
