"""
================================================================================
 分章项目 · Ch7 进阶 · 四大 NLP 任务（MLM / 翻译 / 摘要 / 抽取式QA）+ 对应指标
================================================================================
 Ch7_NER实体识别.py 只覆盖了“6 大主要 NLP 任务”里的 NER；本文件补齐另外四个，
 每个都用【真模型 + 可运行代码 + 手写指标】（transformers v5 已删这些 pipeline，故用
 AutoModelForX + generate 的底层写法；seqeval/sacrebleu/rouge 未装，指标全部手写）：
   ① MLM 掩码语言模型：DataCollatorForLanguageModeling 随机遮词 → 训练 → 困惑度 Perplexity。
   ② 翻译：AutoModelForSeq2SeqLM(opus-mt) 英译中 → 手写 BLEU。
   ③ 摘要：T5 "summarize:" → 手写 ROUGE-L(基于最长公共子序列)。
   ④ 抽取式 QA：AutoModelForQuestionAnswering 取 start/end span → 手写 SQuAD EM/F1。
   完整章节材料见 ../../../Chapter 7/。
 跑：python3 Ch7_进阶_四大NLP任务.py     （首次下 opus-mt/t5-small/squad 模型，约几百MB）
================================================================================
"""
import math
import torch

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# ① MLM 掩码语言模型：随机遮词训练 + 困惑度
# ==============================================================================
def task_mlm():
    # print("=" * 70, "\n① MLM 掩码语言模型(DataCollatorForLanguageModeling + 困惑度)\n" + "=" * 70)
    from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                              DataCollatorForLanguageModeling)
    from torch.optim import AdamW
    ckpt = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForMaskedLM.from_pretrained(ckpt).to(DEV)

    corpus = ["the movie was absolutely fantastic and moving",
              "the food at this restaurant is delicious",
              "machine learning models need lots of data",
              "the weather today is sunny and warm"] * 6
    enc = tok(corpus, truncation=True, padding=True, max_length=32)
    data = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
            for i in range(len(corpus))]
    # ★DataCollatorForLanguageModeling：组 batch 时【随机把 15% token 换成 [MASK]】，标签=原词
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=8, shuffle=True, collate_fn=collator)

    def perplexity():
        model.eval(); losses = []
        for batch in loader:
            batch = {k: v.to(DEV) for k, v in batch.items()}
            with torch.no_grad():
                losses.append(model(**batch).loss.item())
        model.train()
        return math.exp(sum(losses) / len(losses))            # 困惑度 = exp(平均交叉熵)

    print(f"  训练前困惑度 = {perplexity():.1f}")
    opt = AdamW(model.parameters(), lr=5e-5)
    model.train()
    for _ in range(3):
        for batch in loader:
            batch = {k: v.to(DEV) for k, v in batch.items()}
            model(**batch).loss.backward(); opt.step(); opt.zero_grad()
    print(f"  训练后困惑度 = {perplexity():.1f}  (困惑度=模型对文本的“意外程度”，越低越好)")

    # fill-mask 看它学的领域词
    model.eval()
    s = "the movie was absolutely [MASK] ."
    e = tok(s, return_tensors="pt").to(DEV)
    with torch.no_grad():
        logits = model(**e).logits
    mpos = (e["input_ids"][0] == tok.mask_token_id).nonzero()[0]
    top = torch.topk(logits[0, mpos], 5)[1][0]
    print(f"  fill-mask {s!r} → {[tok.decode([t]) for t in top]}")


# ==============================================================================
# ② 翻译：opus-mt 英译中 + 手写 BLEU
# ==============================================================================
def bleu(pred, ref, max_n=4):
    """手写 BLEU(字符级，适合中文)：n-gram 修正精度的几何平均 × 简短惩罚。"""
    pred, ref = list(pred), list(ref)
    if not pred:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pg = [tuple(pred[i:i + n]) for i in range(len(pred) - n + 1)]
        rg = [tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)]
        if not pg:
            precisions.append(0.0); continue
        from collections import Counter
        rc = Counter(rg)
        match = sum(min(c, rc[g]) for g, c in Counter(pg).items())
        precisions.append(match / len(pg))
    if min(precisions) == 0:
        geo = 0.0
    else:
        geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / max(len(pred), 1))  # 简短惩罚
    return bp * geo


def task_translation():
    # print("\n" + "=" * 70, "\n② 翻译(opus-mt 英译中) + 手写 BLEU\n" + "=" * 70)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    m = "Helsinki-NLP/opus-mt-en-zh"
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForSeq2SeqLM.from_pretrained(m).to(DEV).eval()
    pairs = [("Machine learning is fun.", "机器学习很有趣。"),
             ("The weather is nice today.", "今天天气很好。")]
    enc = tok([s for s, _ in pairs], return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=40)
    preds = tok.batch_decode(out, skip_special_tokens=True)
    for (src, ref), pred in zip(pairs, preds):
        print(f"  {src!r}\n    译文={pred!r}  参考={ref!r}  BLEU={bleu(pred, ref):.2f}")
    # seq2seq 的关键：labels 用 text_target 编码(下面示意，不训练)
    # print("  训练时 labels 用 tok(text_target=中文) 编码；AutoModelForSeq2SeqLM 内部做 teacher forcing。")


# ==============================================================================
# ③ 摘要：T5 + 手写 ROUGE-L
# ==============================================================================
def rouge_l(pred, ref):
    """手写 ROUGE-L：基于最长公共子序列(LCS)的 F1(词级)。"""
    p, r = pred.split(), ref.split()
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if p[i - 1] == r[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(p)][len(r)]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def task_summarization():
    # print("\n" + "=" * 70, "\n③ 摘要(T5 'summarize:') + 手写 ROUGE-L\n" + "=" * 70)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(DEV).eval()
    doc = ("The transformer is a deep learning architecture based on attention. "
           "It has become the standard for natural language processing tasks and "
           "powers most large language models today.")
    ref = "The transformer is an attention-based architecture that powers large language models."
    enc = tok(["summarize: " + doc], return_tensors="pt").to(DEV)     # T5 靠任务前缀区分任务
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=40)
    pred = tok.batch_decode(out, skip_special_tokens=True)[0]
    print(f"  摘要={pred!r}")
    print(f"  参考={ref!r}\n  ROUGE-L(F1) = {rouge_l(pred, ref):.2f}  (LCS 越长=覆盖参考越多)")


# ==============================================================================
# ④ 抽取式 QA：取 start/end span + 手写 SQuAD EM/F1
# ==============================================================================
def squad_f1(pred, gold):
    """手写 SQuAD 指标：EM(完全匹配) + token 级 F1(词重叠)。"""
    p, g = pred.lower().split(), gold.lower().split()
    em = 1.0 if p == g else 0.0
    from collections import Counter
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return em, 0.0
    prec, rec = same / len(p), same / len(g)
    return em, 2 * prec * rec / (prec + rec)


def task_qa():
    # print("\n" + "=" * 70, "\n④ 抽取式 QA(start/end span) + 手写 SQuAD EM/F1\n" + "=" * 70)
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    m = "distilbert-base-cased-distilled-squad"
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForQuestionAnswering.from_pretrained(m).to(DEV).eval()
    context = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    qas = [("Where do I work?", "Hugging Face"), ("What is my name?", "Sylvain")]
    for q, gold in qas:
        enc = tok(q, context, return_tensors="pt").to(DEV)
        with torch.no_grad():
            o = model(**enc)
        # 抽取式 QA = 预测答案在 context 里的【起始 token】和【结束 token】下标
        s, e = int(o.start_logits.argmax()), int(o.end_logits.argmax())
        ans = tok.decode(enc["input_ids"][0][s:e + 1])
        em, f1 = squad_f1(ans, gold)
        print(f"  Q: {q}\n    预测={ans!r}  标准={gold!r}  EM={em:.0f} F1={f1:.2f}")


if __name__ == "__main__":
    print(f">>> 设备={DEV}\n")
    task_mlm()
    task_translation()
    task_summarization()
    task_qa()
    print("\n✅ Ch7 四大任务跑通：MLM(困惑度) + 翻译(BLEU) + 摘要(ROUGE-L) + 抽取式QA(EM/F1)。")
    # print("面试：Q 抽取式QA 预测什么? Q 困惑度是什么? Q BLEU/ROUGE 分别衡量什么? Q seq2seq 的 labels 怎么来?"
          # " (见 ../面试高频题库.py)")
