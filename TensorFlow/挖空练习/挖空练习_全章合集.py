"""
================================================================================
 挖空练习 · 全章合集（Ch1/2/3/5/6/7/11/12 + vLLM 部署）—— 一个文件搞定
================================================================================
 玩法：
   1) 运行本文件，按提示输入要练的章号（1/2/3/5/6/7/11/12/v），或命令行： python3 挖空练习_全章合集.py 1
   2) 先别看笔记，把对应练习函数里的每个 ______ 凭记忆填上
   3) 再运行该章：没填的地方报 NameError（告诉你漏哪行）；填错报错/结果不对
   4) 实在卡住 → 翻到文件最底部「答案区」（先尽力想，别马上看）

 建议：今天填一遍，过 2~3 天再来填一遍（忘了再想起来，记得最牢）。

 章节：
   1  Ch1 pipeline 原理   分词 → 模型 → softmax → 标签
   2  Ch2 微调全流程       数据 → Trainer → 评估（小子集，mps 几十秒）
   3  Ch3 使用预训练模型    fill-mask（pipeline 版 + 手动版）
   5  Ch5 数据集+语义搜索   嵌入 → FAISS 索引 → 检索
   6  Ch6 分词器           快速分词器 offset → 实体合并（NER 看家本领）
   7  Ch7 主要NLP任务       NER 标签对齐（词标签 → 子词 token 标签）
   11 Ch11 LoRA 微调         给 LLM 套 LoRA，只训 <1% 参数
   12 Ch12 GRPO 强化学习     组内优势归一化（RL for LLMs 的心脏）
   v  vLLM 优化推理部署     mlx-lm 等价实践（采样参数）
================================================================================
"""
import sys


def title(n, text):
    print("\n" + "=" * 72)
    print(f"  第 {n} 章挖空练习：{text}")
    print("=" * 72)


# ==============================================================================
# 第 1 章：pipeline 原理（分词 → 模型 → softmax → 标签）
# ==============================================================================
def practice_ch1():
    title(1, "分词 → 模型 → softmax → 标签")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"

    # 练习1：加载分词器（AutoTokenizer 的哪个方法？参数 ckpt）
    tokenizer = ______
    # 练习2：加载带分类头的模型
    model = ______

    texts = ["I love this movie!", "This is terrible."]

    # 练习3：编码——补齐、截断、返回 PyTorch 张量
    inputs = tokenizer(texts, padding=______, truncation=______, return_tensors=______)

    # 练习4：喂给模型，取 logits
    logits = model(**inputs).______

    # 练习5：logits → 概率（哪个函数？最后一维 dim=-1）
    probs = torch.nn.functional.______(logits, dim=______)

    # 练习6：每行取概率最大的下标
    pred_ids = probs.______(dim=-1)

    for text, pid, prob in zip(texts, pred_ids, probs):
        label = model.config.id2label[pid.item()]
        print(f"  {text!r:30s} -> {label} ({prob.max().item():.4f})")
    print("自检：应为 POSITIVE / NEGATIVE，概率都 0.99+")


# ==============================================================================
# 第 2 章：微调全流程（数据 → Trainer → 评估）
# ==============================================================================
def practice_ch2():
    title(2, "微调：数据 → Trainer → 评估（小子集）")
    import numpy as np
    from datasets import load_dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, TrainingArguments, Trainer)
    import evaluate

    ckpt = "bert-base-uncased"

    # 练习1：加载 MRPC（GLUE 在 nyu-mll/glue，子任务 mrpc）
    raw = load_dataset(______, ______)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    # 练习2：句子对一起编码、截断、不填充（列要用 list(...) 包）
    def tokenize_function(example):
        return tokenizer(list(example["sentence1"]), ______, truncation=______)

    # 练习3：批量应用分词函数
    tokenized = raw.______(tokenize_function, batched=______)
    # 练习4：动态填充器
    collator = ______

    train_ds = tokenized["train"].select(range(500))
    eval_ds = tokenized["validation"].select(range(200))

    # 练习5：加载模型，2 个标签
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=______)

    # 练习6：评估函数——logits argmax 后比对
    def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        preds = np.______(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # 练习7：训练参数（训 1 轮、每轮评估、不连 wandb）
    args = TrainingArguments(
        "test-trainer-practice",
        num_train_epochs=______,
        eval_strategy=______,
        report_to="none",
    )
    # 练习8：组装 Trainer
    trainer = Trainer(
        model, args,
        train_dataset=______,
        eval_dataset=______,
        data_collator=______,
        processing_class=______,
        compute_metrics=______,
    )
    # 练习9：开始训练
    trainer.______()
    print("✅ 训练完成，每个 epoch 会打印 eval_accuracy / eval_f1")


# ==============================================================================
# 第 3 章：使用预训练模型（fill-mask）
# ==============================================================================
def practice_ch3():
    title(3, "fill-mask：pipeline 版 + 手动版")
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

    ckpt = "bert-base-uncased"

    # ---- Part A：pipeline 版 ----
    # 练习1：创建 fill-mask 的 pipeline
    fm = pipeline(______, model=ckpt)
    # 练习2：动态取 mask 符号（别写死 [MASK]）
    mask = fm.tokenizer.______
    print("pipeline top1：", fm(f"The capital of France is {mask}.")[0]["token_str"])

    # ---- Part B：手动版 ----
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    # 练习3：加载掩码语言模型头
    model = ______
    model.eval()

    sentence = f"The weather today is very {tokenizer.mask_token}."
    # 练习4：分词，返回 PyTorch 张量
    inputs = tokenizer(sentence, return_tensors=______)
    # 练习5：找 [MASK] 位置（在 input_ids 里找等于 mask_token_id 的位置）
    mask_pos = (inputs["input_ids"][0] == tokenizer.______).nonzero(as_tuple=True)[0].item()
    # 练习6：前向，推理不算梯度
    with torch.______():
        logits = model(**inputs).logits
    # 练习7：mask 位置 → softmax（最后一维）→ top5
    probs = torch.softmax(logits[0, mask_pos], dim=______)
    top = torch.topk(probs, 5)
    print("手动 top5：", [tokenizer.decode([t]) for t in top.indices])


# ==============================================================================
# 第 5 章：数据集处理 + 语义搜索（FAISS）
# ==============================================================================
def practice_ch5():
    title(5, "数据集处理 + 语义搜索（FAISS）")
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModel

    corpus = [
        "You can load a dataset offline without internet.",
        "FAISS builds an index to find similar embeddings quickly.",
        "Semantic search finds documents by meaning, not keywords.",
        "Padding makes all sequences in a batch the same length.",
    ]
    ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt)
    model.eval()

    def embed(texts):
        # 练习1：编码——补齐、截断、返回 PyTorch 张量
        enc = tok(texts, padding=______, truncation=______, return_tensors=______)
        with torch.no_grad():
            out = model(**enc)
        # 练习2：CLS 池化——取每句第一个 token([CLS]) 的最后隐藏状态（下标填几？）
        return out.last_hidden_state[:, ______]

    # 练习3：从字典建 Dataset（方法名？）
    ds = Dataset.______({"text": corpus})
    # 练习4：map 出 embeddings 列——每条取 [?] 变成一维向量
    ds = ds.map(lambda x: {"embeddings": embed([x["text"]]).cpu().numpy()[______]})
    # 练习5：给 embeddings 列建 FAISS 索引（方法名？）
    ds.______(column="embeddings")

    query = "how to search text by meaning"
    q_emb = embed([query]).cpu().numpy()
    # 练习6：检索最近的 2 条（方法名？参数：列名, 查询向量, k=?）
    scores, samples = ds.______("embeddings", q_emb, k=______)
    print(f"问题：{query}")
    for s, t in zip(scores, samples["text"]):
        print(f"  [{s:.1f}] {t}")
    print("自检：top1 应命中 'Semantic search finds documents by meaning'")


# ==============================================================================
# 第 6 章：分词器 —— 快速分词器 offset + 实体合并（NER 看家本领）
# ==============================================================================
def practice_ch6():
    title(6, "分词器：offset 把 token 对回原文 + 合并实体")
    import numpy as np
    from transformers import AutoTokenizer

    ckpt = "bert-base-cased"
    # 练习1：加载分词器（AutoTokenizer 的哪个方法？参数 ckpt）
    tok = ______

    example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    # 练习2：编码并要求返回 offset 映射（参数名？值 True——只有快速分词器才有 offset）
    enc = tok(example, ______=True)
    # 练习3：拿 token 文本列表（方法名？）
    tokens = enc.______
    # 练习4：拿 offset 列表（encoding 的哪个键？）
    offsets = enc[______]

    # 模拟 NER 模型的预测标签（真实里来自模型；这里写死，专注练 offset+合并）
    def fake_label(t):
        if t in {"S", "##yl", "##va", "##in"}: return "I-PER"
        if t in {"Hu", "##gging", "Face"}:     return "I-ORG"
        if t == "Brooklyn":                    return "I-LOC"
        return "O"

    labels = [fake_label(t) for t in tokens]
    scores = [0.99 if l != "O" else 1.0 for l in labels]

    results, idx = [], 0
    while idx < len(labels):
        label = labels[idx]
        if label != "O":
            # 练习5：去掉 "I-" 前缀，只留实体类型（切片从第几位开始？）
            etype = label[______]
            # 练习6：取当前 token 的“起始字符”（offsets[idx] 是 (start, end)）
            start, _ = offsets[idx]
            all_scores = []
            # 练习7：只要下一个仍是同类 I-etype 就继续合并（比较什么？）
            while idx < len(labels) and labels[idx] == ______:
                all_scores.append(scores[idx])
                _, end = offsets[idx]
                idx += 1
            # 练习8：用 offset 把碎 token 还原成原文完整单词（对 example 切片）
            word = example[______]
            results.append({"entity_group": etype, "score": float(np.mean(all_scores)),
                            "word": word, "start": start, "end": end})
        else:
            idx += 1

    print(f"原文：{example}")
    for r in results:
        print(f"  {r['entity_group']:4} {r['word']!r:16} "
              f"[{r['start']},{r['end']}) score={r['score']:.3f}")
    print("自检：应得到 PER='Sylvain' / ORG='Hugging Face' / LOC='Brooklyn'")


# ==============================================================================
# 第 7 章：主要 NLP 任务 —— NER 标签对齐（词标签 → 子词 token 标签）
# ==============================================================================
def practice_ch7():
    title(7, "NER 标签对齐：把词级标签摊到子词级 token")
    from transformers import AutoTokenizer

    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
                  'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    # 练习1：加载 bert-base-cased 分词器
    tok = ______
    words = ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."]
    tags = [3, 0, 7, 0, 0, 0, 7, 0, 0]
    # 练习2：对“已分好词的列表”分词——要加 is_split_into_words=True
    inputs = tok(words, ______=True)

    def align(labels, word_ids):
        new_labels, current = [], None
        for wid in word_ids:
            if wid != current:
                current = wid
                # 练习3：特殊标记(None)填 -100，否则用该词标签
                new_labels.append(______)
            elif wid is None:
                new_labels.append(-100)
            else:
                label = labels[wid]
                # 练习4：B-XXX(奇数)要 +1 变 I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
        return new_labels

    # 练习5：拿每个 token 属于第几个单词
    word_ids = inputs.______
    aligned = align(tags, word_ids)
    print("tokens :", inputs.tokens())
    print("对齐后 :", [label_names[a] if a != -100 else "-100" for a in aligned])
    print("自检：'lamb'→'la','##mb'，首片留标签、续片 -100；[CLS]/[SEP] 也 -100。")


# ==============================================================================
# 第 11 章：LoRA 参数高效微调 —— 给 LLM 套 LoRA，只训 <1% 参数
# ==============================================================================
def practice_ch11():
    title(11, "LoRA：给小 LLM 套低秩适配器，只训 <1% 参数")
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    # 练习1：LoRA 配置(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")
    lora_config = LoraConfig(
        r=______, lora_alpha=______, lora_dropout=0.05,
        target_modules=______, task_type=______, bias="none",
    )
    # 练习2：把基座包成 LoRA 模型(peft 的哪个函数？)
    lora_model = ______(base, lora_config)
    # 练习3：打印可训练参数占比(方法名？)
    lora_model.______()

    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    print(f"可训练占比 = {trainable/total*100:.2f}%（应 <1%）")


# ==============================================================================
# 第 12 章：GRPO 强化学习 —— 组内优势归一化（GRPO 的心脏）
# ==============================================================================
def practice_ch12():
    title(12, "GRPO：把一组答案的奖励变成‘组内相对优势’")
    import torch
    rewards = torch.tensor([1., 0., 0., 1., 0., 0., 1., 1.])
    G = 4
    # 练习1：按组 reshape 成 (B, G)（第一维用 -1 自动推断）
    grouped = rewards.view(______, G)
    # 练习2：每组均值(dim=?)，再 repeat_interleave(G) 广播回每个成员
    mean = grouped.mean(dim=______).repeat_interleave(G)
    # 练习3：每组标准差(dim=?)，同样广播
    std = grouped.std(dim=______).repeat_interleave(G)
    # 练习4：组内优势 =（奖励 - 组均值）/（组标准差 + 1e-8）
    advantages = ______
    print("优势:", [round(x, 2) for x in advantages.tolist()])
    assert abs(advantages.view(-1, G).sum(dim=1)).max() < 1e-4
    print("自检通过：每组优势和≈0，正=强化答对的、负=抑制答错的。")


# ==============================================================================
# 附：vLLM 优化推理部署（vLLM 在 Mac 装不了，用 mlx-lm 实践等价概念）
# ==============================================================================
def practice_vllm():
    print("\n" + "=" * 72)
    print("  vLLM 优化推理部署（用 mlx-lm 实践：本地生成 + 采样参数）")
    print("=" * 72)
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    # 练习1：加载模型（函数名？返回 model, tokenizer）
    model, tok = ______(MODEL)
    # 练习2：造采样器——温度 0.7、top_p 0.95（对应 vLLM 的 SamplingParams）
    sampler = make_sampler(temp=______, top_p=______)

    prompt = "Explain what an LLM is in one sentence."
    mx.random.seed(0)
    # 练习3：生成——传 model, tok, prompt；最多 40 个 token；用上面的 sampler
    out = generate(model, tok, prompt, max_tokens=______, sampler=______)
    print("生成：", out.strip())
    print("👉 对应 vLLM：llm = LLM(model); llm.generate(prompt, SamplingParams(...))")


# ------------------------------------------------------------------------------
# 调度器
# ------------------------------------------------------------------------------
CHAPTERS = {1: practice_ch1, 2: practice_ch2, 3: practice_ch3,
            5: practice_ch5, 6: practice_ch6, 7: practice_ch7,
            11: practice_ch11, 12: practice_ch12}
EXTRA = {"v": practice_vllm, "vllm": practice_vllm}   # vLLM 部署用 v 触发

MENU = """\
用法：
  python3 挖空练习_全章合集.py <章号>   练某一章，如 1 / 2 / 3 / 5 / 6 / 7 / 11 / 12 / v
  python3 挖空练习_全章合集.py all       依次练全部

章节：
  1  Ch1 pipeline 原理（6 空）     2  Ch2 微调全流程（9 空）
  3  Ch3 fill-mask（7 空）         5  Ch5 数据集+语义搜索（6 空）
  6  Ch6 分词器 offset+实体（8 空） 7  Ch7 NER 标签对齐（5 空）
  11 Ch11 LoRA 配置（5 空）         12 Ch12 GRPO 组内优势（4 空）
  v  vLLM 优化推理部署（5 空，用 mlx-lm）
"""


def run_choice(choice: str):
    choice = choice.strip().lower()
    if choice == "all":
        for n in sorted(CHAPTERS):
            CHAPTERS[n]()
        practice_vllm()          # 数字章跑完，再跑 vLLM（避免 int/str 混排序报错）
        return
    if choice in EXTRA:          # v / vllm → vLLM 部署练习
        EXTRA[choice]()
        return
    # 只把“章号合不合法”放进 try，别把练习函数的执行也包进去，
    # 否则练习里真报错会被误吞成“没有第 N 章”，掩盖真正的 bug。
    try:
        n = int(choice)
    except ValueError:
        print(f"没有第 {choice} 章，请输入 1/2/3/5/6/7/11/12/v 或 all。")
        return
    if n not in CHAPTERS:
        print(f"没有第 {choice} 章，请输入 1/2/3/5/6/7/11/12/v 或 all。")
        return
    CHAPTERS[n]()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        run_choice(args[0])
    else:
        print(MENU)
        while True:
            choice = input("练哪一章？（1/2/3/5/6/7/11/12/v，或 all，回车/q 退出）：")
            if choice.strip().lower() in ("", "q", "quit", "exit"):
                print("已退出。")
                break
            run_choice(choice)
            print()


# ==============================================================================
#  答案区（卡住再看！先尽力想，回忆的挣扎才是记忆的关键）
# ==============================================================================
#  【Ch1】
#   1: AutoTokenizer.from_pretrained(ckpt)
#   2: AutoModelForSequenceClassification.from_pretrained(ckpt)
#   3: padding=True, truncation=True, return_tensors="pt"
#   4: model(**inputs).logits
#   5: torch.nn.functional.softmax(logits, dim=-1)
#   6: probs.argmax(dim=-1)
#
#  【Ch2】
#   1: load_dataset("nyu-mll/glue", "mrpc")
#   2: list(example["sentence2"]), truncation=True
#   3: raw.map(tokenize_function, batched=True)
#   4: DataCollatorWithPadding(tokenizer=tokenizer)
#   5: num_labels=2
#   6: np.argmax(logits, axis=-1)
#   7: num_train_epochs=1, eval_strategy="epoch"
#   8: train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator,
#      processing_class=tokenizer, compute_metrics=compute_metrics
#   9: trainer.train()
#
#  【Ch3】
#   1: pipeline("fill-mask", model=ckpt)
#   2: fm.tokenizer.mask_token
#   3: AutoModelForMaskedLM.from_pretrained(ckpt)
#   4: return_tensors="pt"
#   5: tokenizer.mask_token_id
#   6: with torch.no_grad():
#   7: dim=-1
#
#  【Ch5】
#   1: padding=True, truncation=True, return_tensors="pt"
#   2: out.last_hidden_state[:, 0]
#   3: Dataset.from_dict({"text": corpus})
#   4: embed([x["text"]]).cpu().numpy()[0]
#   5: ds.add_faiss_index(column="embeddings")
#   6: ds.get_nearest_examples("embeddings", q_emb, k=2)
#
#  【Ch6】
#   1: AutoTokenizer.from_pretrained(ckpt)
#   2: tok(example, return_offsets_mapping=True)
#   3: enc.tokens()
#   4: enc["offset_mapping"]
#   5: label[2:]                    # 去掉 "I-" 两个字符
#   6: (start, _ = offsets[idx] 已给)
#   7: labels[idx] == f"I-{etype}"  # 下一个仍是同类才继续合并
#   8: example[start:end]           # 用原文字符区间还原完整单词
#
#  【Ch7】
#   1: AutoTokenizer.from_pretrained("bert-base-cased")
#   2: tok(words, is_split_into_words=True)
#   3: -100 if wid is None else labels[wid]
#   4: if label % 2 == 1: label += 1     # B-XXX(奇数)→I-XXX
#   5: inputs.word_ids()
#
#  【Ch11】
#   1: r=8, lora_alpha=16, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM"
#   2: get_peft_model(base, lora_config)
#   3: lora_model.print_trainable_parameters()
#
#  【Ch12】
#   1: rewards.view(-1, G)
#   2: grouped.mean(dim=1).repeat_interleave(G)
#   3: grouped.std(dim=1).repeat_interleave(G)
#   4: (rewards - mean) / (std + 1e-8)
#
#  【vLLM 部署】
#   1: load(MODEL)
#   2: temp=0.7, top_p=0.95
#   3: max_tokens=40, sampler=sampler
# ==============================================================================
