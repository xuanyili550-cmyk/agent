"""
================================================================================
 分章项目 · Ch7 · NER 命名实体识别（贴 HF Ch7：token 分类全流程）
================================================================================
 HF 课程 Ch7“主要 NLP 任务”里最常用的 NER——识别人名/机构/地点。用可运行代码复现三层：
   ① pipeline 版：一行拿到实体级结果(aggregation 把子词合并成完整实体)。
   ② 手写版：模型对【每个 token】打 BIO 标签，看清 pipeline 内部到底做了什么。
   ③ 子词对齐难点：一个词被切成多个子词(S/##yl/##va/##in)，靠 offset 把它们拼回一个实体；
      训练时子词的非首片用 -100 忽略 loss(Ch6/7 的核心工程点)。
   完整章节材料见 ../../../Chapter 7/；中文 NER 见 ../中文版/中文_NER实体识别.py；生产用法见 ../综合项目/项目1、项目4。
 跑：python3 Ch7_NER实体识别.py     (下 NER 模型 ~430MB，缓存则秒开)
================================================================================
"""
import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification, pipeline)

CKPT = "huggingface-course/bert-finetuned-ner"
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
TEXT = "My name is Sylvain and I work at Hugging Face in Brooklyn."


# ==============================================================================
# ① pipeline 版：aggregation_strategy="simple" 直接给实体级结果
# ==============================================================================
def ner_pipeline():
    # print("=" * 70, "\n① pipeline 版(实体级，子词已合并)\n" + "=" * 70)
    ner = pipeline("token-classification", model=CKPT,
                   aggregation_strategy="simple", device=0 if DEV == "cuda" else -1)
    for text in [TEXT, "Tim Cook announced the new iPhone at Apple headquarters in Cupertino."]:
        print(f"\n  文本: {text}")
        for e in ner(text):
            print(f"     {e['entity_group']:5} {e['word']!r:16} (score={e['score']:.3f})")
    return ner


# ==============================================================================
# ② 手写版：模型对每个 token 打 BIO 标签(看清 pipeline 内部)
# ==============================================================================
def ner_manual():
    # print("\n" + "=" * 70, "\n② 手写版(每个 token 一个 BIO 标签)\n" + "=" * 70)
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForTokenClassification.from_pretrained(CKPT).to(DEV).eval()

    # return_offsets_mapping：拿到每个 token 在原文里的字符区间，用于拼词/对齐
    enc = tok(TEXT, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0]
    enc = {k: v.to(DEV) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0]                     # [seq_len, num_labels]
    pred_ids = logits.argmax(-1).tolist()
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])

    print(f"  {'token':<12}{'BIO标签':<12}{'字符区间'}")
    for t, pid, (a, b) in zip(tokens, pred_ids, offsets.tolist()):
        label = model.config.id2label[pid]
        if t in tok.all_special_tokens:                     # [CLS]/[SEP] 跳过
            continue
        mark = "  ← 实体" if label != "O" else ""
        print(f"  {t:<12}{label:<12}[{a},{b}]{mark}")
    # print("  B-XXX=实体开头, I-XXX=实体内部, O=非实体；pipeline 就是把连续的 B/I 按 offset 合并成一个实体。")


# ==============================================================================
# ③ 子词对齐 + -100：训练时怎么给子词打标签
# ==============================================================================
def subword_alignment():
    # print("\n" + "=" * 70, "\n③ 子词对齐与 -100(训练时的标签工程)\n" + "=" * 70)
    tok = AutoTokenizer.from_pretrained(CKPT)
    enc = tok("Sylvain", return_tensors="pt")
    pieces = tok.convert_ids_to_tokens(enc["input_ids"][0])
    print(f"  'Sylvain' 被切成子词: {pieces}")
    # print("  训练打标签规则：一个词的【首个子词】给真实标签(如 B-PER)，")
    # print("  其余子词(##yl/##va/##in)标 -100 → 计算 loss 时被忽略，避免同一个词内部标签打架。")
    # print("  推理时反过来：把连续同类子词按 offset 拼回完整实体(就是 aggregation 干的事)。")


if __name__ == "__main__":
    print(f">>> 设备={DEV}\n")
    ner_pipeline()
    ner_manual()
    subword_alignment()
    print("\n✅ Ch7 跑通：pipeline 实体级 → 手写 token 级 BIO → 子词对齐/-100。")
    # print("面试：Q aggregation_strategy 有什么用? Q NER 里 -100 标签是干嘛的? Q 子词怎么拼回实体?"
          # " (见 ../面试高频题库.py 三.分词器)")
