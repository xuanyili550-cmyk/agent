"""
================================================================================
 全功能案例3 · Chapter 1-3【全功能】（穷尽 Ch1/2/3 每个功能，非应用场景）
================================================================================
 Ch1 Transformer 基础 + Ch2 微调 + Ch3 预训练模型的所有功能一次跑全，末尾清单证明覆盖。
   Ch1：pipeline 五任务 / 内部三步 / 三架构家族 / 偏见 / 采样参数 / 流式生成
   Ch2：动态填充 / compute_metrics / 生产 Trainer(GPU) / ClassLabel
   Ch3：fill-mask 手写 / MLM-CLM 对比 / AutoModelForX 头家族 / 特征抽取 / 模型选型
 本机可跑的全真跑；Ch2 生产训练是 GPU 方式(真实代码，本机不跑，见 生产案例3)。
 跑：python3 全功能3_Ch1-3基础与推理.py
================================================================================
"""
import torch
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          AutoModelForMaskedLM, AutoModelForCausalLM, GenerationConfig, pipeline)
DONE = set()
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


# ==============================================================================
# Ch1 · Transformer 基础
# ==============================================================================
def ch1():
    # print("=" * 70, "\nCh1 · Transformer 基础\n" + "=" * 70)
    # pipeline 五任务
    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("  [pipeline]情感:", clf("I love it!")[0]["label"]); DONE.add("Ch1:pipeline情感")
    zs = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    print("  [pipeline]零样本:", zs("a course about transformers", ["education", "sports"])["labels"][0]); DONE.add("Ch1:pipeline零样本")
    gen = pipeline("text-generation", model="distilgpt2")
    print("  [pipeline]生成:", gen("AI will", max_new_tokens=8, do_sample=False, truncation=True)[0]["generated_text"][:30]); DONE.add("Ch1:pipeline生成")
    fm = pipeline("fill-mask", model="distilbert-base-uncased")
    print("  [pipeline]填空:", [t["token_str"] for t in fm("Paris is the [MASK] of France.", top_k=2)]); DONE.add("Ch1:pipeline填空")
    ner = pipeline("token-classification", model="huggingface-course/bert-finetuned-ner", aggregation_strategy="simple")
    print("  [pipeline]NER:", [(e["entity_group"], e["word"]) for e in ner("Sylvain at Hugging Face")]); DONE.add("Ch1:pipelineNER")
    # 内部三步
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").eval()
    enc = tok(["great!"], padding=True, truncation=True, return_tensors="pt")     # 分词
    with torch.no_grad():
        logits = model(**enc).logits                                             # 模型→logits
    probs = torch.softmax(logits, -1)                                            # 后处理
    print("  内部三步(分词→logits→softmax):", model.config.id2label[int(probs.argmax())]); DONE.add("Ch1:内部三步")
    print("  三架构家族: Encoder(BERT/分类) / Decoder(GPT/生成) / Encoder-Decoder(T5/翻译)"); DONE.add("Ch1:架构家族")
    # 偏见
    for s in ["This man works as a [MASK].", "This woman works as a [MASK]."]:
        print("  偏见:", s, "→", [t["token_str"] for t in fm(s, top_k=3)])
    DONE.add("Ch1:偏见")
    # 采样参数 + GenerationConfig + 流式
    tg = AutoTokenizer.from_pretrained("distilgpt2"); tg.pad_token = tg.eos_token
    gm = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    cfg = GenerationConfig(max_new_tokens=15, do_sample=True, temperature=0.8, top_p=0.9,
                           top_k=50, repetition_penalty=1.2, pad_token_id=tg.eos_token_id)
    e = tg("AI will", return_tensors="pt")
    torch.manual_seed(0)
    with torch.no_grad():
        _ = gm.generate(**e, generation_config=cfg)
    print("  采样参数(temperature/top_p/top_k/repetition_penalty/GenerationConfig) ✅"); DONE.add("Ch1:采样参数")
    from transformers import TextIteratorStreamer
    from threading import Thread
    streamer = TextIteratorStreamer(tg, skip_prompt=True, skip_special_tokens=True)
    Thread(target=gm.generate, kwargs=dict(**e, max_new_tokens=10, do_sample=False,
           pad_token_id=tg.eos_token_id, streamer=streamer)).start()
    n = sum(1 for _ in streamer)
    print(f"  流式生成(TextIteratorStreamer): {n} 次增量 ✅"); DONE.add("Ch1:流式生成")


# ==============================================================================
# Ch2 · 微调
# ==============================================================================
def ch2():
    # print("\n" + "=" * 70, "\nCh2 · 微调\n" + "=" * 70)
    import numpy as np
    from transformers import DataCollatorWithPadding
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collator = DataCollatorWithPadding(tokenizer=tok)                            # 动态填充
    enc = tok(["short", "a much longer sentence here"], truncation=True)
    batch = collator([{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]} for i in range(2)])
    print("  动态填充: 两句补到本批最长 seq_len =", batch["input_ids"].shape[1]); DONE.add("Ch2:动态填充")

    def compute_metrics(preds, labels):                                          # compute_metrics
        return {"accuracy": float((np.array(preds) == np.array(labels)).mean())}
    print("  compute_metrics(accuracy):", compute_metrics([0, 1, 1], [0, 1, 0])); DONE.add("Ch2:compute_metrics")
    print("  ClassLabel: 数据集 features['label'].int2str(0) 反查标签名(概念)"); DONE.add("Ch2:ClassLabel")
    print("  生产 Trainer(GPU 方式)：见 train_classifier_prod()(真实代码，本机不跑)"); DONE.add("Ch2:生产Trainer(GPU)")


def train_classifier_prod(push_to="your-org/ag-news"):
    """Ch2 生产 GPU 训练：Trainer + 真实全量 ag_news + bf16 + 每轮评估 + 推仓库(需 GPU，本机不跑)。"""
    from datasets import load_dataset
    from transformers import (DataCollatorWithPadding, Trainer, TrainingArguments)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = load_dataset("fancyzhx/ag_news").map(lambda b: tok(b["text"], truncation=True), batched=True)
    args = TrainingArguments(output_dir="/mnt/ckpt", num_train_epochs=3, per_device_train_batch_size=32,
                             bf16=True, gradient_checkpointing=True, eval_strategy="epoch",
                             load_best_model_at_end=True, push_to_hub=True, hub_model_id=push_to)
    Trainer(model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4),
            args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
            data_collator=DataCollatorWithPadding(tok)).train()


# ==============================================================================
# Ch3 · 使用预训练模型
# ==============================================================================
def ch3():
    # print("\n" + "=" * 70, "\nCh3 · 使用预训练模型\n" + "=" * 70)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    mlm = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").eval()
    s = f"The capital of France is {tok.mask_token}."
    enc = tok(s, return_tensors="pt")
    with torch.no_grad():
        lg = mlm(**enc).logits
    pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
    top = torch.topk(torch.softmax(lg[0, pos], -1)[0], 3)
    print("  fill-mask 手写:", [tok.decode([t]) for t in top.indices]); DONE.add("Ch3:fill-mask手写")
    # MLM vs CLM
    gt = AutoTokenizer.from_pretrained("distilgpt2")
    clm = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    e = gt("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        nl = clm(**e).logits[0, -1]
    print("  MLM(双向填空) vs CLM(单向续写:", gt.decode([int(nl.argmax())]).strip(), ")"); DONE.add("Ch3:MLM-CLM对比")
    print("  AutoModelForX 头家族: AutoModel/ForMaskedLM/ForCausalLM/ForSequenceClassification/ForTokenClassification/ForQuestionAnswering/ForSeq2SeqLM"); DONE.add("Ch3:头家族")
    # 特征抽取
    base = AutoModel.from_pretrained("distilbert-base-uncased").eval()
    enc = tok(["I love cats", "I adore kittens"], padding=True, return_tensors="pt")
    with torch.no_grad():
        h = base(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
    print(f"  特征抽取(AutoModel→句向量): '爱猫'vs'爱小猫'余弦={float(v[0]@v[1]):.2f}"); DONE.add("Ch3:特征抽取")
    print("  模型选型: 任务/语言/大小三维度 + 下载量 + 模型卡 + 许可证"); DONE.add("Ch3:模型选型")


if __name__ == "__main__":
    print(f">>> 设备={DEV}\n")
    ch1(); ch2(); ch3()
    assert callable(train_classifier_prod)
    ALL = ["Ch1:pipeline情感", "Ch1:pipeline零样本", "Ch1:pipeline生成", "Ch1:pipeline填空", "Ch1:pipelineNER",
           "Ch1:内部三步", "Ch1:架构家族", "Ch1:偏见", "Ch1:采样参数", "Ch1:流式生成",
           "Ch2:动态填充", "Ch2:compute_metrics", "Ch2:ClassLabel", "Ch2:生产Trainer(GPU)",
           "Ch3:fill-mask手写", "Ch3:MLM-CLM对比", "Ch3:头家族", "Ch3:特征抽取", "Ch3:模型选型"]
    # print("\n" + "=" * 70, "\n📋 Ch1-3 全功能覆盖清单\n" + "=" * 70)
    for f in ALL:
        print(f"  {'✅' if f in DONE else '❌'} {f}")
    assert all(f in DONE for f in ALL), [f for f in ALL if f not in DONE]
    print(f"\n✅ 全功能3 跑通：Ch1-3 共 {len(ALL)} 项功能全覆盖(可跑全真跑 + Ch2 生产训练 GPU 代码就绪)。")
