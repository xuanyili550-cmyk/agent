"""
================================================================================
 Chapter 7 · 主要 NLP 任务(Main NLP tasks) —— 系统学习笔记（可直接运行）
================================================================================
 配套 HuggingFace 课程「Main NLP tasks」整章，对应本目录 3 个原始文件里的 6 个任务：
   1 Token 分类/NER      ← Token classification.py
   2 掩码语言模型 MLM     ← Fine-tuning a masked language model.py(前半)
   3 翻译 Translation     ← Fine-tuning a masked language model.py(后半)
   4 摘要 Summarization   ← Summarization.py(任务A)
   5 因果语言模型 CLM      ← Summarization.py(任务B)
   6 抽取式问答 QA         ← Summarization.py(任务C)

 边跑边学：直接
     python3 Chapter7_主要NLP任务_学习笔记.py
 就能看到每个任务“最核心的那段逻辑”的真实 print 输出。

 ★ 设计说明（为什么这样组织）：
   本章 6 个任务的“真训练”都要 GPU + 联网 + 往 Hub 推模型，Mac 上不现实。
   但每个任务真正要理解的，是它**独有的那段预处理/标签/评估逻辑**，这些都能快速跑：
     · NER 的“词标签→子词标签”对齐          · MLM/CLM 的“拼接+切块+随机掩码”
     · 翻译/摘要的“目标文本当标签(text_target)” · QA 的“答案字符区间→start/end token 下标”
   这些纯逻辑默认就跑。需要下大模型的“真实推理 pipeline”统一由开关 RUN_PIPELINES 控制
   (默认 False，改成 True 会真的下模型跑推理)。评估指标(seqeval/sacrebleu/rouge/squad)
   若没装会打印“pip install”提示并跳过，不影响其余部分。
================================================================================
"""

# 把“会下大模型的真实推理”统一关掉；想看真效果改成 True(会联网下模型)。
RUN_PIPELINES = False

import numpy as np


def banner(text):
    print("\n" + "=" * 72 + f"\n {text}\n" + "=" * 72)


def try_metric(name, *args):  # 内部用新版 evaluate.load，名字避开已弃用的 datasets.load_metric
    """安全加载 evaluate 指标：没装依赖就返回 None 并提示，不让整份崩掉。"""
    try:
        import evaluate
        return evaluate.load(name, *args)
    except Exception as e:
        print(f"  (跳过 {name} 指标：{type(e).__name__}。装一下 → pip install {name})")
        return None


# ##############################################################################
# 任务 1 · Token 分类 / 命名实体识别(NER)  ← Token classification.py
# ##############################################################################
banner("任务1 · NER：把“词级标签”对齐到“子词级 token”(核心难点)")

# CoNLL-2003 的 9 个标签(BIO 体系)。B-=实体开头，I-=实体内部，O=非实体。
label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# ★核心：标签对齐。数据里标签是“每个词一个”，但分词器把词切成子词后 token 变多，
# 必须把标签也“摊开”对齐到每个 token。规则：
#   · 特殊标记([CLS]/[SEP])和“子词的非首片” → -100(交叉熵会忽略，不算 loss/不评估)
#   · 一个词的首片 → 用该词的标签
#   · 词内后续片 → 若该词是 B-XXX(实体开头)，续片要改成 I-XXX(实体内部)
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:            # 新词的第一个子词
            current_word = word_id
            new_labels.append(-100 if word_id is None else labels[word_id])
        elif word_id is None:                  # 特殊标记
            new_labels.append(-100)
        else:                                  # 同一个词的后续子词
            label = labels[word_id]
            if label % 2 == 1:                 # 奇数=B-XXX → 加 1 变成 I-XXX
                label += 1
            new_labels.append(label)
    return new_labels


# 用一个真实分词器演示(bert-base-cased 分词器很小)。'lamb' 会被切成 'la','##mb'。
try:
    from transformers import AutoTokenizer
    ner_tok = AutoTokenizer.from_pretrained("bert-base-cased")
    words = ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."]
    tags = [3, 0, 7, 0, 0, 0, 7, 0, 0]         # B-ORG O B-MISC O O O B-MISC O O
    enc = ner_tok(words, is_split_into_words=True)
    aligned = align_labels_with_tokens(tags, enc.word_ids())
    print("  tokens :", enc.tokens())
    print("  原词标签:", [label_names[t] for t in tags])
    print("  对齐后 :", [label_names[a] if a != -100 else "-100" for a in aligned])
    print("  看 'lamb'→'la','##mb'：首片保留标签，续片变 -100；特殊标记也是 -100。")
except Exception as e:
    print("  (需要 transformers 才能跑对齐演示)", e)

# 评估：seqeval 按“整个实体”算 P/R/F1(BIO 都对才算对)，不是按单 token。
seqeval = try_metric("seqeval")
if seqeval:
    refs = [["B-ORG", "O", "B-MISC", "O", "O", "O", "B-MISC", "O", "O"]]
    preds = [["B-ORG", "O", "O", "O", "O", "O", "B-MISC", "O", "O"]]   # 故意错一个
    r = seqeval.compute(predictions=preds, references=refs)
    print(f"  seqeval: precision={r['overall_precision']:.2f} "
          f"recall={r['overall_recall']:.2f} f1={r['overall_f1']:.2f}")

if RUN_PIPELINES:
    from transformers import pipeline
    ner = pipeline("token-classification",
                   model="huggingface-course/bert-finetuned-ner",
                   aggregation_strategy="simple")
    print("  NER 推理:", ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))


# ##############################################################################
# 任务 2 · 掩码语言模型(MLM)  ← Fine-tuning a masked language model.py(前半)
# ##############################################################################
banner("任务2 · MLM：拼接+切块(group_texts) + 随机掩码 15%")

# 领域自适应：拿通用 BERT，在你的领域文本(如影评)上继续做“完形填空”预训练，
# 让它更懂这个领域的用词。数据处理关键两步：拼接所有文本 → 切成等长 chunk。
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
print("  两条长度 30/45 → 拼成 75 → 丢零头到 60 → 切成",
      len(chunks["input_ids"]), "块，每块长度",
      [len(c) for c in chunks["input_ids"]])

# 随机掩码：DataCollatorForLanguageModeling 每个 batch 随机遮 15% 的 token 让模型猜。
try:
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling
    mlm_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collator = DataCollatorForLanguageModeling(tokenizer=mlm_tok, mlm_probability=0.15)
    sample = mlm_tok("the movie was absolutely fantastic and i loved every minute")
    out = collator([sample])
    masked = mlm_tok.decode(out["input_ids"][0])
    print("  随机掩码后:", masked)
    print("  被遮的位置标签=原词、其余=-100(只在被遮处算 loss)。")
except Exception as e:
    print("  (需要 transformers 才能跑掩码演示)", e)

# 评估用困惑度 Perplexity = exp(交叉熵 loss)：直观理解“模型每步平均在几个词里犹豫”，越低越好。
loss_demo = 2.5
print(f"  困惑度示例: loss={loss_demo} → perplexity=exp(loss)={np.exp(loss_demo):.1f}")


# ##############################################################################
# 任务 3 · 翻译(Translation)  ← Fine-tuning a masked language model.py(后半)
# ##############################################################################
banner("任务3 · 翻译：用 text_target 处理“目标语言标签” + SacreBLEU")

# seq2seq：输入英文，输出法文。★关键：目标句(法文)要用 text_target 编码成 labels，
# 这样它才用“解码器端的分词规则”，否则会被当成源语言切错。
try:
    from transformers import AutoTokenizer
    tr_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    inputs = tr_tok("Default to expanded threads",
                    text_target="Passer par défaut aux fils de discussion étendus")
    print("  input_ids(英文) 前8:", inputs["input_ids"][:8])
    print("  labels(法文)   前8:", inputs["labels"][:8], " ← 目标句被单独正确编码")
except Exception as e:
    print("  (翻译分词器需要 sentencepiece)", e)

# SacreBLEU：翻译最常用指标(自带分词，跨模型可比)。references 是“列表的列表”(可有多个参考译文)。
bleu = try_metric("sacrebleu")
if bleu:
    ref = [["This plugin allows you to automatically translate web pages between several languages."]]
    good = ["This plugin lets you translate web pages between several languages automatically."]
    bad = ["This This This This"]
    print(f"  好译文 BLEU={bleu.compute(predictions=good, references=ref)['score']:.1f}"
          f"   坏译文 BLEU={bleu.compute(predictions=bad, references=ref)['score']:.1f}")

if RUN_PIPELINES:
    # transformers v5 移除了 translation pipeline，直接用 seq2seq 模型 generate。
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _tk = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    _m = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr").eval()
    _o = _m.generate(**_tk("Default to expanded threads", return_tensors="pt"), max_length=64)
    print("  翻译推理:", _tk.decode(_o[0], skip_special_tokens=True))


# ##############################################################################
# 任务 4 · 摘要(Summarization)  ← Summarization.py(任务A)
# ##############################################################################
banner("任务4 · 摘要：先建“取前3句”基线 + ROUGE 指标")

# 摘要也是 seq2seq(长文→短文)。评估前先建一个“笨基线”：直接取正文前 3 句当摘要，
# 微调后的模型必须超过这个基线才算有价值。
review = ("I really enjoyed this book. The characters were well developed and the plot "
          "kept me hooked. I would definitely recommend it to anyone. The ending was a "
          "bit rushed though. Overall a solid read.")
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    baseline = "\n".join(sent_tokenize(review)[:3])
    print("  取前3句基线摘要：\n   ", baseline.replace("\n", " / "))
except Exception as e:
    print("  (需要 nltk+punkt 才能分句)", e)

# ROUGE：比“生成摘要”和“参考摘要”的 n-gram / 最长公共子序列 重合度(召回导向)。
rouge = try_metric("rouge")
if rouge:
    r = rouge.compute(predictions=["I absolutely loved reading the Hunger Games"],
                      references=["I loved reading the Hunger Games"])
    print("  ROUGE:", {k: round(float(v), 3) for k, v in r.items()})

if RUN_PIPELINES:
    # v5 移除了 summarization pipeline，用 t5-small + "summarize:" 前缀 generate。
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _tk = AutoTokenizer.from_pretrained("t5-small")
    _m = AutoModelForSeq2SeqLM.from_pretrained("t5-small").eval()
    _o = _m.generate(**_tk("summarize: " + review, return_tensors="pt", truncation=True),
                     max_length=40, num_beams=4)
    print("  摘要推理:", _tk.decode(_o[0], skip_special_tokens=True))


# ##############################################################################
# 任务 5 · 从零训练因果语言模型(CLM)  ← Summarization.py(任务B)
# ##############################################################################
banner("任务5 · 因果LM：mlm=False 的数据整理 + 关键token加权损失")

# 因果=只能看左边、预测下一个词(GPT 家族)。同一个 DataCollatorForLanguageModeling，
# 设 mlm=False 就切到 CLM 模式：labels 直接等于 input_ids(模型内部会自动“错一位”对齐，
# 用第 t 个 token 预测第 t+1 个)。
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
banner("任务6 · QA：答案“字符区间”→ start/end token 下标(本章最难)")

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


# ##############################################################################
# 六任务对比 & 面试速答（纯文字）
# ##############################################################################
banner("六任务对比 & 面试速答")
# ┌────────────┬─────────────┬──────────────────────┬───────────────┬──────────┐
# │ 任务       │ 模型类型    │ 标签是什么           │ 数据整理器    │ 指标     │
# ├────────────┼─────────────┼──────────────────────┼───────────────┼──────────┤
# │ NER        │ Encoder     │ 每个 token 一个类别  │ ForToken...   │ seqeval  │
# │ MLM        │ Encoder     │ 被遮住的原词         │ ForLangModel  │ 困惑度   │
# │ 翻译       │ Enc-Dec     │ 目标语言句子         │ ForSeq2Seq    │ SacreBLEU│
# │ 摘要       │ Enc-Dec     │ 摘要文本             │ ForSeq2Seq    │ ROUGE    │
# │ 因果LM     │ Decoder     │ input_ids(错一位)    │ ForLM(mlm=F)  │ 困惑度   │
# │ QA(抽取)   │ Encoder     │ start/end token 下标 │ default       │ SQuAD f1 │
# └────────────┴─────────────┴──────────────────────┴───────────────┴──────────┘
#
# 面试速答：
#   Q NER 里的 -100 是干嘛的？
#   A 交叉熵忽略的标签。特殊标记和“子词非首片”设 -100，不参与 loss/评估；B-XXX 续片改 I-XXX。
#   Q MLM 和 CLM 区别？
#   A MLM(BERT) 双向、遮词填空、标签是被遮词；CLM(GPT) 单向、预测下一个词、labels=input_ids。
#   Q 翻译/摘要为什么要 text_target / DataCollatorForSeq2Seq？
#   A 目标文本要用解码器端规则编码成 labels；seq2seq collator 会同时动态填充输入和标签(-100)。
#   Q 困惑度 / BLEU / ROUGE / SQuAD-f1 分别评什么？
#   A 困惑度=语言模型犹豫程度(越低越好)；BLEU=译文精确率导向；ROUGE=摘要召回导向；SQuAD-f1=答案词重合。
#   Q QA 最难的一步？
#   A 把“答案在原文的字符区间”映射成“start/end token 下标”，长 context 用 stride 滑窗切块。

print("\n✅ 全部跑完。想看真实模型推理，把文件顶部 RUN_PIPELINES 改成 True(会联网下模型)。")
