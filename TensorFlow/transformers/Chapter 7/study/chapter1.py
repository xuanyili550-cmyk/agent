RUN_PIPELINES = False

import numpy as np
def try_metric(name, *args):
    try:
        import evaluate
        return evaluate.load(name, *args)
    except Exception as e:
        return None

def align_labels_with_tokens(labels,word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id!=current_word:
            current_word=word_id
            new_labels.append(-100 if  word_id is not None else labels[word_id])
        elif word_id is None:
            new_labels.append(-100)
        else:
            label=labels[word_id]
            if label%2==1:
                label+=1
            new_labels.append(label)
    return new_labels
try:
    from transformers import AutoTokenizer
    ner_tok = AutoTokenizer.from_pretrained("bert-base-cased")
    words = ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."]
    tags = [3, 0, 7, 0, 0, 0, 7, 0, 0]         # B-ORG O B-MISC O O O B-MISC O O
    enc = ner_tok(words, is_split_into_words=True)
    aligned = align_labels_with_tokens(tags, enc.word_ids())
except Exception as e:
    print("  (需要 transformers 才能跑对齐演示)", e)

seqeval = try_metric("seqeval")
if seqeval:
    refs = [["B-ORG", "O", "B-MISC", "O", "O", "O", "B-MISC", "O", "O"]]
    preds = [["B-ORG", "O", "O", "O", "O", "O", "B-MISC", "O", "O"]]   # 故意错一个
    r = seqeval.compute(predictions=preds, references=refs)
if RUN_PIPELINES:
    from transformers import pipeline
    ner = pipeline("token-classification",
                   model="huggingface-course/bert-finetuned-ner",
                   aggregation_strategy="simple")
def group_texts(examples, chunk_size=128):
    # ① 把 batch 里所有样本的 token 首尾相接成一长条(丢掉句子边界，纯粹为切块)
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    # ② 丢掉结尾不足一个 chunk 的零头(保证每块都满 chunk_size)
    total_length = (total_length // chunk_size) * chunk_size
    # ③ 切成一块块
    result = {k: [t[i:i + chunk_size] for i in range(0, total_length, chunk_size)]
              for k, t in concatenated.items()}
    # ④ MLM 的标签=输入本身(collator 之后会把被遮位置以外的标签设 -100)
    result["labels"] = result["input_ids"].copy()
    return result

demo = {"input_ids": [list(range(0, 30)), list(range(100, 145))]}   # 两条“假 token”
chunks = group_texts(demo, chunk_size=20)

from transformers import AutoTokenizer,DataCollatorForLanguageModeling
mlm_tok=AutoTokenizer.from_pretrained("distilbert-base-uncased")
collator=DataCollatorForLanguageModeling(mlm=mlm_tok,mlm_probability=0.15)
sample = mlm_tok("the movie was absolutely fantastic and i loved every minute")
out = collator([sample])
masked = mlm_tok.decode(out["input_ids"][0])

loss_demo = 2.5

tr_tok=AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
inputs = tr_tok("Default to expanded threads",
                    text_target="Passer par défaut aux fils de discussion étendus")

bleu = try_metric("sacrebleu")
if bleu:
    ref = [["This plugin allows you to automatically translate web pages between several languages."]]
    good = ["This plugin lets you translate web pages between several languages automatically."]
    bad = ["This This This This"]
if RUN_PIPELINES:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _tk = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    _m = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr").eval()
    _o = _m.generate(**_tk("Default to expanded threads", return_tensors="pt"), max_length=64)

review = ("I really enjoyed this book. The characters were well developed and the plot "
          "kept me hooked. I would definitely recommend it to anyone. The ending was a "
          "bit rushed though. Overall a solid read.")
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize
baseline = "\n".join(sent_tokenize(review)[:3])
rouge = try_metric("rouge")
if rouge:
    r = rouge.compute(predictions=["I absolutely loved reading the Hunger Games"],
                      references=["I loved reading the Hunger Games"])

if RUN_PIPELINES:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _tk = AutoTokenizer.from_pretrained("t5-small")
    _m = AutoModelForSeq2SeqLM.from_pretrained("t5-small").eval()
    _o = _m.generate(**_tk("summarize: " + review, return_tensors="pt", truncation=True),
                     max_length=40, num_beams=4)

try:
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling
    clm_tok = AutoTokenizer.from_pretrained("distilgpt2")
    clm_tok.pad_token = clm_tok.eos_token
    clm_collator = DataCollatorForLanguageModeling(clm_tok, mlm=False)
    s = clm_tok("x = np.random.randn(100)")
    out = clm_collator([s])
    print("  mlm=False → labels 等于 input_ids：",
          bool((out["input_ids"][0] == out["labels"][0]).all().item()))
    print("  为什么不用手动错位：模型 forward 内部会 shift，用前 t 个预测第 t+1 个。")
except Exception as e:
    print("  (需要 transformers 才能演示 CLM 整理器)", e)

# 关键 token 加权损失：想让模型更重视某些 token(如 plt/pd/sk)，就给这些位置的 loss 乘更大权重。
keytokens = {"plt", "pd", "sk", "fit", "predict"}
seq = ["import", "pd", ";", "pd", ".", "read_csv", "(", "plt", ")"]
weights = [3.0 if t in keytokens else 1.0 for t in seq]     # 关键 token 权重更高
print("  token :", seq)
print("  权重  :", weights, " ← 关键 token 的损失被放大，逼模型更快学会写它们")

if RUN_PIPELINES:
    from transformers import pipeline
    gen = pipeline("text-generation", model="distilgpt2")
    print("  代码生成:", gen("import pandas as pd\ndf =", max_new_tokens=10)[0]["generated_text"])


# ##############################################################################
# 任务 6 · 抽取式问答(QA)  ← Summarization.py(任务C)
# ##############################################################################

# 抽取式 QA：答案一定是 context 的连续子串，模型预测“起始 token”和“结束 token”。
# ★难点：训练标签是“答案在原文的字符位置”，要用 offset 把它换算成 token 下标。
try:
    from transformers import AutoTokenizer
    qa_tok = AutoTokenizer.from_pretrained("bert-base-cased")
    context = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
               "in Paris, France. It was completed in 1889.")
    question = "When was the Eiffel Tower completed?"
    answer_text = "1889"
    start_char = context.index(answer_text)          # 答案在原文的起始字符
    end_char = start_char + len(answer_text)          # 结束字符

    enc = qa_tok(question, context, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    seq_ids = enc.sequence_ids()                      # None/0=问题, 1=context

    # 找到 context 在 token 序列里的起止范围
    idx = 0
    while seq_ids[idx] != 1:
        idx += 1
    ctx_start = idx
    while idx < len(seq_ids) and seq_ids[idx] == 1:
        idx += 1
    ctx_end = idx - 1

    # 若答案不在 context 内→标 (0,0)；否则用 offset 定位到 start/end token 下标
    if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
        start_pos = end_pos = 0
    else:
        i = ctx_start
        while i <= ctx_end and offsets[i][0] <= start_char:
            i += 1
        start_pos = i - 1
        i = ctx_end
        while i >= ctx_start and offsets[i][1] >= end_char:
            i -= 1
        end_pos = i + 1
    print(f"  答案字符区间 [{start_char},{end_char}) → token 下标 [{start_pos},{end_pos}]")
    print("  解码回来验证:", repr(qa_tok.decode(enc["input_ids"][start_pos:end_pos + 1])))
except Exception as e:
    print("  (需要 transformers 才能跑 QA 定位)", e)

# 推理时怎么从 logits 选答案：在 start/end 各取 n_best 个高分下标，两两组合，
# 过滤掉“end<start 或 太长 或 不在 context”的，剩下按 start_logit+end_logit 取最高。
def best_answer_from_logits(start_logits, end_logits, offsets, context,
                            n_best=5, max_len=30):
    starts = np.argsort(start_logits)[-1:-n_best - 1:-1]
    ends = np.argsort(end_logits)[-1:-n_best - 1:-1]
    best = None
    for s in starts:
        for e in ends:
            if offsets[s] is None or offsets[e] is None:
                continue
            if e < s or e - s + 1 > max_len:          # 非法/过长组合丢掉
                continue
            score = start_logits[s] + end_logits[e]
            if best is None or score > best["score"]:
                best = {"text": context[offsets[s][0]:offsets[e][1]], "score": score}
    return best


ctx = "Paris is the capital of France."
offs = [(0, 5), (6, 8), (9, 12), (13, 16), (17, 19), (20, 26), (26, 27)]   # 每 token 的字符区间
sl = np.array([5.0, 0, 0, 0, 0, 1.0, 0])          # 'Paris' 起点分最高
el = np.array([5.0, 0, 0, 0, 0, 0, 0])            # 'Paris' 也是终点
print("  n_best 挑答案:", best_answer_from_logits(sl, el, offs, ctx))

squad = try_metric("squad")
if squad:
    r = squad.compute(
        predictions=[{"id": "1", "prediction_text": "1889"}],
        references=[{"id": "1", "answers": {"text": ["1889"], "answer_start": [90]}}])
    print("  SQuAD 指标:", r)

if RUN_PIPELINES:
    # v5 移除了 question-answering pipeline，直接用 QA 模型取 start/end logits。
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    _ck = "distilbert-base-cased-distilled-squad"
    _tk = AutoTokenizer.from_pretrained(_ck)
    _m = AutoModelForQuestionAnswering.from_pretrained(_ck).eval()
    _ctx = "🤗 Transformers is backed by Jax, PyTorch and TensorFlow."
    _in = _tk("Which libraries back 🤗 Transformers?", _ctx,
              return_tensors="pt", return_offsets_mapping=True)
    _off = _in.pop("offset_mapping")[0]
    _out = _m(**_in)
    _s, _e = int(_out.start_logits.argmax()), int(_out.end_logits.argmax())
    print("  QA 推理:", repr(_ctx[_off[_s][0]:_off[_e][1]]))
