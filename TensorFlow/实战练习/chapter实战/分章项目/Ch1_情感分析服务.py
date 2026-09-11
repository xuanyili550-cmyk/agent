"""
================================================================================
 分章项目 · Ch1 · Transformer 模型入门（贴 HF 课程 Ch1：pipeline 全景 + 内部 + 架构 + 偏见）
================================================================================
 HF 课程第 1 章讲四件事，本文件逐一用【可运行真代码】复现：
   ① pipeline 一行解决多种任务(情感/零样本/生成/完形填空/NER)——先感受“能干什么”。
   ② pipeline 内部三步 = 分词器 + 模型 + 后处理——再看清“怎么做到的”。
   ③ 三大架构家族：Encoder(BERT) / Decoder(GPT) / Encoder-Decoder(T5)——各擅长什么任务。
   ④ 偏见与局限：预训练模型会从数据里学到偏见(必须知道的伦理点)。
   完整章节材料见 ../../../Chapter 1/；批量/生产版见 ../综合项目/项目6。
 注：transformers v5 已移除 summarization/translation/question-answering 这三个 pipeline，本文用还在的任务。
 跑：python3 Ch1_情感分析服务.py
================================================================================
"""
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, pipeline)

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# ① pipeline 全景：一行代码解决多种任务(HF Ch1 的“开胃菜”)
# ==============================================================================
def pipeline_tour():
    # ① pipeline 全景：一行调用多任务

    # 情感分析(encoder 分类)
    clf = pipeline("sentiment-analysis",
                   model="distilbert-base-uncased-finetuned-sst-2-english")
    print("  [情感]", clf("I've been waiting for this course my whole life.")[0])

    # 零样本分类：不用训练，自己给候选标签(NLI 模型判蕴含)
    zs = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    r = zs("This is a course about the Transformers library",
           candidate_labels=["education", "politics", "business"])
    print(f"  [零样本] {r['labels'][0]} ({r['scores'][0]:.2f})  候选={r['labels']}")

    # 文本生成(decoder)
    gen = pipeline("text-generation", model="distilgpt2")
    out = gen("In this course, we will teach you how to",
              max_new_tokens=20, do_sample=False, truncation=True)
    print("  [生成]", out[0]["generated_text"][:80], "...")

    # 完形填空(encoder 的预训练任务 MLM)
    mask = pipeline("fill-mask", model="distilbert-base-uncased")
    top = mask("This course will teach you all about [MASK] models.", top_k=3)
    print("  [填空] top3:", [t["token_str"] for t in top])

    # 命名实体识别(token 分类)
    ner = pipeline("token-classification", model="huggingface-course/bert-finetuned-ner",
                   aggregation_strategy="simple")
    print("  [NER]", [(e["entity_group"], e["word"]) for e in
                      ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")])


# ==============================================================================
# ② pipeline 内部三步：分词器 → 模型(logits) → 后处理(softmax)
# ==============================================================================
def pipeline_internals():
    # ② pipeline 内部三步(手写复现 sentiment-analysis)
    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEV).eval()

    texts = ["I absolutely love this!", "This is the worst thing ever.", "It's okay, nothing special."]
    # ⓐ 分词：文本 → input_ids/attention_mask（padding 对齐、truncation 防超长）
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt").to(DEV)
    # ⓑ 模型：前向输出 logits（未归一化的分数）
    with torch.no_grad():
        logits = model(**enc).logits
    # ⓒ 后处理：softmax→概率、argmax→标签（pipeline 帮你做的那一步）
    probs = torch.softmax(logits, dim=-1)
    for text, p in zip(texts, probs):
        i = int(p.argmax())
        print(f"  {text!r:38} → {model.config.id2label[i]} ({p[i]:.3f})")


# ==============================================================================
# ③ 三大架构家族：选对结构才能干对活
# ==============================================================================
def architectures():
    # ③ 三大 Transformer 架构家族
    table = [
        ("Encoder(自编码)", "BERT/DistilBERT/RoBERTa", "双向看全文", "分类/NER/抽取式QA/句向量"),
        ("Decoder(自回归)", "GPT/LLaMA/Qwen",         "单向predict下一个词", "文本生成/对话/补全"),
        ("Encoder-Decoder", "T5/BART/mT5",            "编码理解+解码生成", "翻译/摘要/seq2seq"),
    ]
    # print(f"  {'家族':<18}{'代表':<26}{'注意力':<20}{'擅长任务'}")
    # for fam, rep, attn, task in table:
    #     print(f"  {fam:<18}{rep:<26}{attn:<20}{task}")
    # 记忆：分类/理解找 Encoder；生成/对话找 Decoder；一进一出(翻译摘要)找 Encoder-Decoder。


# ==============================================================================
# ④ 偏见与局限：预训练模型会从语料里学到社会偏见(HF Ch1 的伦理提醒)
# ==============================================================================
def bias_demo():
    # ④ 偏见与局限(用 fill-mask 看性别职业偏见)
    mask = pipeline("fill-mask", model="distilbert-base-uncased")
    for sent in ["This man works as a [MASK].", "This woman works as a [MASK]."]:
        preds = [t["token_str"] for t in mask(sent, top_k=5)]
        print(f"  {sent:32} → {preds}")
    # 即便训练数据看似中立，模型仍可能给出带性别刻板印象的职业——上线前必须评估/缓解偏见。


if __name__ == "__main__":
    # print(f">>> 设备={DEV}\n")
    pipeline_tour()
    pipeline_internals()
    architectures()
    bias_demo()
    print("\n✅ Ch1 全景跑通：pipeline 多任务 → 内部三步 → 架构家族 → 偏见局限。")
    # print("面试：Q pipeline 内部三步? Q 为什么模型只输出 logits? Q Encoder/Decoder/Enc-Dec 各擅长啥?"
    #       " Q 零样本分类靠什么? (答案见 ../面试高频题库.py 一.基础)")
