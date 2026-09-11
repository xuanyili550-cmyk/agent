import sys
import torch
from torch.nn.functional import softmax


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
def title(n, text):
    print("\n" + "=" * 70)
    print(f"  第 {n} 关：{text}")
    print("=" * 70)

def case_01():
    title(1, "分词器三字段 input_ids / token_type_ids / attention_mask")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained("bert-base-cased")
    req="Hello, I'm a single sentence!"
    req1="你好啊，我是你的最爱"
    enc=tok(req)
    print("编码结果：", enc)
    print("input_ids     = token 的编号（101=[CLS]起始, 102=[SEP]结束）")
    print("token_type_ids = 句子A/B 区分，单句全 0")
    print("attention_mask = 1 关注 / 0 忽略")
    print("解码回文本：", tok.decode(enc["input_ids"]))

def case_02():
    title(2, "句子对：token_type_ids 区分前后句")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    enc = tok("How are you?", "I'm fine, thank you!")
    print(enc)
    print("👉 注意 token_type_ids：前一句是 0，后一句变成 1")
    print("   这类‘句子对’输入用于问答、句子关系判断等任务")

def case_03():
    title(3, "padding 填充：把短句补齐到统一长度")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    enc=tok([
        "How are you?", "I'm fine, thank you!"],
        padding = True,
        return_tensors='pt'   # 注意是 return_tensors（带 s），少个 s 就不返回张量了
    )
    print(enc)
    print("👉 短句尾部补了 0（padding），对应 attention_mask 也是 0")
    print("   padding=True/'longest' 补到本批最长；'max_length' 补到指定/模型上限")

def case_04():
    title(4, "truncation 截断：超长句砍短（BERT 上限 512）")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    long_text = "This is a very " + "very " * 60 + "long sentence."
    no_trunc = tok(long_text)
    trunc=tok(long_text,truncation=True,max_length=40)
    print("不截断长度：", len(no_trunc["input_ids"]))
    print("截断到 10  ：", len(trunc["input_ids"]), "->", trunc["input_ids"])
    print("👉 padding 管‘补短’，truncation 管‘砍长’，常一起用保证长度统一")

def case_05():
    title(5, "底层三步 tokenize → convert_tokens_to_ids → decode")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    s = "Using a Transformer network is simple"
    tokens=tok.tokenize(s)
    print("①切词  tokenize:", tokens, "  （##former 表示接在前词后）")
    ids=tok.convert_tokens_to_ids(tokens)
    print("②转ID  convert_tokens_to_ids:", ids, "  （不含 [CLS]/[SEP]）")
    print("③解码  decode:", tok.decode(ids))
    end=tok(s,add_special_tokens=False)
    end1=tok(s)
    print(tok.decode(end["input_ids"]),"不含 [CLS]/[SEP]）")
    print(tok.decode(end1["input_ids"]),"含 [CLS]/[SEP]）")
    print("👉 对比：tok('文本') 会自动加 [CLS]/[SEP]，这三步手动版不会")

def case_06():
    title(6, "attention_mask 对照实验：有 vs 无，结果差多少")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model=AutoModelForSequenceClassification.from_pretrained(ckpt)
    batched_ids = [
        [200, 200, 200],
        [200, 200, tok.pad_token_id],
    ]
    print("❌ 不给 mask（把填充也算进去，第二行 logits 是错的）：")
    print(model(torch.tensor(batched_ids)).logits)
    mask = [[1, 1, 1], [1, 1, 0]]  # 填充位置置 0
    out = model(torch.tensor(batched_ids), attention_mask=torch.tensor(mask))
    print("✅ 给正确 mask（忽略填充，第二行 logits 才对）：")
    print(out.logits)
    print("👉 结论：批处理时 padding 必须配 attention_mask。")
    print("   平时 tok(..., padding=True) 会自动帮你算好两者。")

def case_07():
    title(7, "AutoModel（无头，出特征） vs ForSequenceClassification（有头，出分数）")
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(ckpt)
    inputs =tok([
        "I love this!", "So have I!"
    ],
        padding=True,
        truncation=True,
        return_tensors = "pt"
    )
    base = AutoModel.from_pretrained(ckpt)
    print("AutoModel 输出 last_hidden_state 形状：",
          tuple(base(**inputs).last_hidden_state.shape),
          " = [批, 序列长, 隐藏维768]（是特征，不是答案）")
    clf = AutoModelForSequenceClassification.from_pretrained(ckpt)
    print("ForSequenceClassification 输出 logits 形状：",
          tuple(clf(**inputs).logits.shape),
          " = [批, 标签数2]（每个标签一个分数）")

def case_08():
    title(8, "logits → softmax → 标签（手动复现 pipeline）")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    texts = ["I've been waiting for this my whole life.", "I hate this!"]
    tokens = tok(texts, padding=True, truncation=True, return_tensors="pt")
    logits=model(**tokens).logits
    print("原始 logits（未归一化，可能有负数）：\n", logits)
    probs=softmax(logits,dim=1)
    print("softmax 后概率（每行加起来=1）：\n", probs)
    # 用 texts 一起 zip，才能打印出“原文 -> 标签”，否则打印的是张量看不懂
    for t, p in zip(texts, probs):
        idx=p.argmax().item()
        print(f"  {t!r:45s} -> {model.config.id2label[idx]} ({p.max():.4f})")

def case_09():
    title(9, "pipeline 一行，验证与第 8 关结果一致")
    from transformers import pipeline
    clf = pipeline("sentiment-analysis")
    texts = ["I've been waiting for this my whole life.", "I hate this!"]
    for t,r in zip(texts,clf(texts)):
        print(f"  {t!r:45s} -> {r['label']} ({r['score']:.4f})")
    print("👉 pipeline 内部就是第 8 关那三步：分词→模型→softmax。")

def case_10():
    title(10, "中文情感分析实战（mps 加速，真跑出结论）")
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.nn.functional import softmax
    device = pick_device()
    print("设备：", device)
    ckpt = "sanshizhang/Chinese-Sentiment-Analysis-Fund-Direction"
    tok = BertTokenizer.from_pretrained(ckpt)
    model = BertForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()
    labels = {0: "负面", 1: "正面", 2: "中性"}
    for text in ["这家公司业绩大涨，前景光明！", "基金又跌了，亏麻了。"]:
        enc=tok(text,max_length=512,truncation=True,padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad(): # 推理不算梯度
            probs = softmax(model(input_ids=ids,attention_mask=mask).logits,dim=1)
        idx=torch.argmax(probs,dim=1).item()
        print(f"  {text} -> {labels[idx]}（{probs[0][idx].item():.4f}）")

def case_11():
    title(11, "生成参数实验：temperature 高低对比")
    # 新版 transformers(5.x) 推荐把生成参数打包进 GenerationConfig，避免一堆警告
    from transformers import pipeline, AutoTokenizer, GenerationConfig
    # 关掉 BPE 分词器的空格清理，避免 clean_up_tokenization_spaces 警告
    tok = AutoTokenizer.from_pretrained("distilgpt2", clean_up_tokenization_spaces=False)
    gen = pipeline("text-generation", model="distilgpt2", tokenizer=tok, device="cpu")
    prompt = "The future of AI is"

    def make_config(temperature, top_p=0.95, max_new_tokens=30):
        return GenerationConfig(
            max_new_tokens=max_new_tokens,  # 只设它，不设 max_length，避免冲突警告
            do_sample=True,                 # True=随机采样；False=贪心
            temperature=temperature,        # 越低越保守，越高越发散
            top_p=top_p,                    # 核采样：累计概率前 top_p 的词里选
            repetition_penalty=1.3,         # 抑制“复读机”
            pad_token_id=tok.eos_token_id,
        )

    def run(cfg):
        # clean_up_tokenization_spaces=False 显式传入，消除 BPE 警告
        return gen(prompt, generation_config=cfg,
                   clean_up_tokenization_spaces=False)[0]["generated_text"]

    print("\n[temperature=0.2] 保守、确定：")
    print(run(make_config(0.2)))
    print("\n[temperature=1.2] 发散、有创意（也更可能胡说）：")
    print(run(make_config(1.2)))

    print("\n👉 自己改：把 temperature、top_p、max_new_tokens 调一调再跑，体会区别。")
    print("\n[temperature=0.8, top_p=0.7]：")
    print(run(make_config(0.8, top_p=0.7, max_new_tokens=50)))
    print("\n[temperature=1.6, top_p=0.5]：")
    print(run(make_config(1.6, top_p=0.5, max_new_tokens=50)))
    print("   这些参数在‘优化推理部署_学习笔记.py’里有逐条详解。")


def case_12():
    title(12, "综合大案例：batch + padding + truncation + 模型 + softmax 全流程")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = pick_device()
    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()
    texts = [
        "This movie is a masterpiece, I loved every minute!",
        "Terrible. A complete waste of my time.",
        "It was okay, nothing special.",
    ]

    # ① 分词（padding 补齐 + truncation 截断 + 返回张量），并搬到设备
    print("步骤① 分词（padding + truncation）")
    tokens = tok(texts, padding=True, truncation=True,
                 max_length=64, return_tensors="pt").to(device)
    print("   input_ids 形状：", tuple(tokens["input_ids"].shape))
    print("   每行真实 token 数：", tokens["attention_mask"].sum(dim=1).tolist())

    # ② 模型前向（eval + no_grad）
    print("步骤② 模型前向（eval + no_grad）")
    with torch.no_grad():
        logits = model(**tokens).logits.to(device)
    print("   logits：\n", logits)

    # ③ 后处理 softmax → 取标签
    print("步骤③ softmax → 取标签")
    probs = softmax(logits, dim=-1).to(device)
    print("   " + "-" * 60)
    for t, p in zip(texts, probs):
        idx = p.argmax().item()
        bar = "█" * int(p.max().item() * 20)
        print(f"   {model.config.id2label[idx]:8s} {p.max().item():.3f} {bar}  {t}")
    print("   " + "-" * 60)
    print("👉 这就是 pipeline('sentiment-analysis') 的完整内部流程。")


def case_13():
    title(13, "单变量对照实验：一次只变一个参数，固定随机种子看纯粹差异")
    from transformers import pipeline, AutoTokenizer, GenerationConfig, set_seed

    tok = AutoTokenizer.from_pretrained("distilgpt2", clean_up_tokenization_spaces=False)
    gen = pipeline("text-generation", model="distilgpt2", tokenizer=tok, device="cpu")
    prompt = "The future of AI is"

    def run(temperature, top_p, max_new_tokens):
        # ★ 关键：每次生成前固定同一个随机种子，这样输出差异只来自“被改的那个参数”，
        #    而不是随机性。否则你分不清是参数起作用还是运气不同。
        set_seed(42)
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.3,
            pad_token_id=tok.eos_token_id,
        )
        text = gen(prompt, generation_config=cfg,
                   clean_up_tokenization_spaces=False)[0]["generated_text"]
        return text[len(prompt):].strip()   # 只看新生成的部分，方便对比

    # ---- A组：只变 temperature（top_p=0.9, 长度=40 固定）----
    print("\n【A】只变 temperature（越低越保守/确定，越高越发散/大胆）")
    print("    top_p=0.9, max_new_tokens=40 固定")
    for t in [0.2, 0.7, 1.5]:
        print(f"  temperature={t}: {run(t, top_p=0.9, max_new_tokens=40)}")

    # ---- B组：只变 top_p（temperature=1.0, 长度=40 固定）----
    print("\n【B】只变 top_p（候选池大小：越小越稳越窄，越大越丰富越随机）")
    print("    temperature=1.0, max_new_tokens=40 固定")
    for p in [0.3, 0.7, 1.0]:
        print(f"  top_p={p}: {run(1.0, top_p=p, max_new_tokens=40)}")

    # ---- C组：只变 max_new_tokens（temperature=0.8, top_p=0.9 固定）----
    print("\n【C】只变 max_new_tokens（只控制“写多长”，不影响用词风格）")
    print("    temperature=0.8, top_p=0.9 固定")
    for m in [15, 40, 80]:
        print(f"  max_new_tokens={m}: {run(0.8, top_p=0.9, max_new_tokens=m)}")

    print("\n👉 看差异的诀窍：对照实验（控制变量）——一次只动一个，其余不变 + 固定种子。")
    print("   temperature 管‘敢不敢乱来’，top_p 管‘从多大池子里选’，max_new_tokens 管‘写多长’。")


CASES = {
    1: case_01, 2: case_02, 3: case_03, 4: case_04, 5: case_05, 6: case_06,
    7: case_07, 8: case_08, 9: case_09, 10: case_10, 11: case_11, 12: case_12,
    13: case_13,
}
MENU = """\
用法：
  python3 案例闯关_从零理解Pipeline.py <关号>   跑单关，如 1 / 10 / 12
  python3 案例闯关_从零理解Pipeline.py all      跑全部

关卡列表：
  1  分词器三字段         2  句子对token_type_ids   3  padding 填充
  4  truncation 截断      5  底层三步               6  attention_mask 对照
  7  AutoModel对比头      8  logits→softmax→标签    9  pipeline 一行验证
  10 中文情感分析   11 生成参数实验            12 综合大案例(完整流水线)
  13 单变量对照实验(看每个参数各自的差异)
"""
def run_choice(choice: str):
    """根据输入的字符串跑对应关卡：数字=单关，all=全部。"""
    choice = choice.strip().lower()
    if choice == "all":
        for n in sorted(CASES):
            CASES[n]()
    else:
        try:
            CASES[int(choice)]()
        except (ValueError, KeyError):
            print(f"没有第 {choice} 关，请输入 1~13 或 all。")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        # 方式A：命令行带参数，如  python3 本文件.py 6
        run_choice(args[0])
    else:
        # 方式B：在 PyCharm 里直接点运行 → 弹出提示让你输入关号
        print(MENU)
        while True:
            choice = input("请输入关号（1~13，或 all 全部，回车/q 退出）：")
            if choice.strip().lower() in ("", "q", "quit", "exit"):
                print("已退出。")
                break
            run_choice(choice)
            print()  # 跑完一关空一行，方便继续输入下一关