"""
================================================================================
 案例闯关：从零理解 Pipeline（12 关，一关一个概念，最后一关综合）
================================================================================
 用法（在本文件所在目录的终端里）：

   python3 案例闯关_从零理解Pipeline.py            # 不带参数：列出所有关卡菜单
   python3 案例闯关_从零理解Pipeline.py 1          # 只跑第 1 关
   python3 案例闯关_从零理解Pipeline.py 10         # 只跑第 10 关（中文情感）
   python3 案例闯关_从零理解Pipeline.py all        # 从头到尾全部跑一遍

 建议玩法：按 1→12 顺序，一关一关跑，读注释 + 看输出 + 自己改参数再跑。
 关卡地图：
   —— 单点概念（各自独立，跑一个就懂一个）——
   1  分词器三字段 input_ids / token_type_ids / attention_mask
   2  句子对：token_type_ids 如何区分 A/B 句
   3  padding 填充：把短句补齐
   4  truncation 截断：把超长句砍短
   5  底层三步 tokenize → convert_tokens_to_ids → decode
   6  attention_mask 对照实验（有/无，结果差多少）
   7  AutoModel vs AutoModelForSequenceClassification（有无任务头）
   8  logits → softmax → 标签（手动复现英文情感分类）
   9  pipeline 一行，验证和第 8 关结果一致
   10 中文情感分析实战（mps 加速，真跑出结论）
   11 生成参数实验：temperature / top_p / max_new_tokens 手感
   —— 综合 ——
   12 综合大案例：把 batch + padding + truncation + 模型 + softmax 串成完整流水线
================================================================================
"""

import sys
import torch


# ------------------------------------------------------------------------------
# 公共小工具：自动选设备（Apple Silicon 用 mps，其次 cpu）
# ------------------------------------------------------------------------------
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


# ==============================================================================
# 第 1 关：分词器三字段
# ==============================================================================
def case_01():
    title(1, "分词器三字段 input_ids / token_type_ids / attention_mask")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    enc = tok("Hello, I'm a single sentence!")
    print("编码结果：", enc)
    print("input_ids     = token 的编号（101=[CLS]起始, 102=[SEP]结束）")
    print("token_type_ids = 句子A/B 区分，单句全 0")
    print("attention_mask = 1 关注 / 0 忽略")
    print("解码回文本：", tok.decode(enc["input_ids"]))
    # 👉 自己试：把句子换成中文，看 input_ids 怎么变


# ==============================================================================
# 第 2 关：句子对
# ==============================================================================
def case_02():
    title(2, "句子对：token_type_ids 区分前后句")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    enc = tok("How are you?", "I'm fine, thank you!")
    print(enc)
    print("👉 注意 token_type_ids：前一句是 0，后一句变成 1")
    print("   这类‘句子对’输入用于问答、句子关系判断等任务")


# ==============================================================================
# 第 3 关：padding 填充
# ==============================================================================
def case_03():
    title(3, "padding 填充：把短句补齐到统一长度")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    enc = tok(["How are you?", "I'm fine, thank you!"],
              padding=True, return_tensors="pt")
    print(enc)
    print("👉 短句尾部补了 0（padding），对应 attention_mask 也是 0")
    print("   padding=True/'longest' 补到本批最长；'max_length' 补到指定/模型上限")


# ==============================================================================
# 第 4 关：truncation 截断
# ==============================================================================
def case_04():
    title(4, "truncation 截断：超长句砍短（BERT 上限 512）")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    long_text = "This is a very " + "very " * 60 + "long sentence."
    no_trunc = tok(long_text)
    trunc = tok(long_text, truncation=True, max_length=10)
    print("不截断长度：", len(no_trunc["input_ids"]))
    print("截断到 10  ：", len(trunc["input_ids"]), "->", trunc["input_ids"])
    print("👉 padding 管‘补短’，truncation 管‘砍长’，常一起用保证长度统一")


# ==============================================================================
# 第 5 关：底层三步
# ==============================================================================
def case_05():
    title(5, "底层三步 tokenize → convert_tokens_to_ids → decode")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    s = "Using a Transformer network is simple"
    tokens = tok.tokenize(s)
    print("①切词  tokenize:", tokens, "  （##former 表示接在前词后）")
    ids = tok.convert_tokens_to_ids(tokens)
    print("②转ID  convert_tokens_to_ids:", ids, "  （不含 [CLS]/[SEP]）")
    print("③解码  decode:", tok.decode(ids))
    print("👉 对比：tok('文本') 会自动加 [CLS]/[SEP]，这三步手动版不会")


# ==============================================================================
# 第 6 关：attention_mask 对照实验
# ==============================================================================
def case_06():
    title(6, "attention_mask 对照实验：有 vs 无，结果差多少")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)

    batched_ids = [
        [200, 200, 200],
        [200, 200, tok.pad_token_id],   # 第二行尾部是填充
    ]
    print("❌ 不给 mask（把填充也算进去，第二行 logits 是错的）：")
    print(model(torch.tensor(batched_ids)).logits)

    mask = [[1, 1, 1], [1, 1, 0]]       # 填充位置置 0
    out = model(torch.tensor(batched_ids), attention_mask=torch.tensor(mask))
    print("✅ 给正确 mask（忽略填充，第二行 logits 才对）：")
    print(out.logits)
    print("👉 结论：批处理时 padding 必须配 attention_mask。")
    print("   平时 tok(..., padding=True) 会自动帮你算好两者。")


# ==============================================================================
# 第 7 关：AutoModel vs AutoModelForSequenceClassification
# ==============================================================================
def case_07():
    title(7, "AutoModel（无头，出特征） vs ForSequenceClassification（有头，出分数）")
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(ckpt)
    inputs = tok(["I love this!", "So have I!"],
                 padding=True, truncation=True, return_tensors="pt")

    base = AutoModel.from_pretrained(ckpt)
    print("AutoModel 输出 last_hidden_state 形状：",
          tuple(base(**inputs).last_hidden_state.shape),
          " = [批, 序列长, 隐藏维768]（是特征，不是答案）")

    clf = AutoModelForSequenceClassification.from_pretrained(ckpt)
    print("ForSequenceClassification 输出 logits 形状：",
          tuple(clf(**inputs).logits.shape),
          " = [批, 标签数2]（每个标签一个分数）")


# ==============================================================================
# 第 8 关：logits → softmax → 标签（手动复现英文情感分类）
# ==============================================================================
def case_08():
    title(8, "logits → softmax → 标签（手动复现 pipeline）")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)

    texts = ["I've been waiting for this my whole life.", "I hate this!"]
    tokens = tok(texts, padding=True, truncation=True, return_tensors="pt")
    logits = model(**tokens).logits
    print("原始 logits（未归一化，可能有负数）：\n", logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    print("softmax 后概率（每行加起来=1）：\n", probs)
    for t, p in zip(texts, probs):
        idx = p.argmax().item()
        print(f"  {t!r:45s} -> {model.config.id2label[idx]} ({p.max():.4f})")


# ==============================================================================
# 第 9 关：pipeline 一行验证
# ==============================================================================
def case_09():
    title(9, "pipeline 一行，验证与第 8 关结果一致")
    from transformers import pipeline

    clf = pipeline("sentiment-analysis")
    texts = ["I've been waiting for this my whole life.", "I hate this!"]
    for t, r in zip(texts, clf(texts)):
        print(f"  {t!r:45s} -> {r['label']} ({r['score']:.4f})")
    print("👉 pipeline 内部就是第 8 关那三步：分词→模型→softmax。")


# ==============================================================================
# 第 10 关：中文情感分析实战
# ==============================================================================
def case_10():
    title(10, "中文情感分析实战（mps 加速，真跑出结论）")
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.nn.functional import softmax

    device = pick_device()
    print("设备：", device)
    ckpt = "sanshizhang/Chinese-Sentiment-Analysis-Fund-Direction"
    tok = BertTokenizer.from_pretrained(ckpt)
    model = BertForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()  # 评估模式

    labels = {0: "负面", 1: "正面", 2: "中性"}
    for text in ["这家公司业绩大涨，前景光明！", "基金又跌了，亏麻了。"]:
        enc = tok(text, max_length=512, truncation=True,
                  padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():                     # 推理不算梯度
            probs = softmax(model(input_ids=ids, attention_mask=mask).logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        print(f"  {text} -> {labels[idx]}（{probs[0][idx].item():.4f}）")


# ==============================================================================
# 第 11 关：生成参数实验（temperature / top_p / max_new_tokens）
# ==============================================================================
def case_11():
    title(11, "生成参数实验：temperature 高低对比")
    # 新版 transformers(5.x) 推荐把生成参数打包进 GenerationConfig 对象统一传，
    # 这样不会触发 “deprecated / max_length 冲突” 等警告（也是更规范的写法）。
    from transformers import pipeline, AutoTokenizer, GenerationConfig

    # 关掉 BPE 分词器的空格清理，避免 clean_up_tokenization_spaces 警告
    tok = AutoTokenizer.from_pretrained("distilgpt2", clean_up_tokenization_spaces=False)
    # 很小的英文生成模型，纯 CPU 就能跑，目的是“感受参数”而非质量
    gen = pipeline("text-generation", model="distilgpt2", tokenizer=tok, device="cpu")
    prompt = "The future of AI is"

    def make_config(temperature):
        # 只放 max_new_tokens（不放 max_length），避免二者冲突警告
        return GenerationConfig(
            max_new_tokens=30,
            do_sample=True,          # True=按参数随机采样；False=贪心永远选最高分
            temperature=temperature, # 越低越保守、越高越发散
            top_p=0.95,              # 核采样：只从累计概率前 95% 的词里选
            repetition_penalty=1.3,  # 抑制“复读机”，让输出更可读
            pad_token_id=tok.eos_token_id,
        )

    # clean_up_tokenization_spaces=False 在调用时显式传入，消除 BPE 分词器那条警告
    print("\n[temperature=0.2] 保守、确定：")
    print(gen(prompt, generation_config=make_config(0.2),
              clean_up_tokenization_spaces=False)[0]["generated_text"])

    print("\n[temperature=1.2] 发散、有创意（也更可能胡说）：")
    print(gen(prompt, generation_config=make_config(1.2),
              clean_up_tokenization_spaces=False)[0]["generated_text"])

    print("\n👉 自己改：把 temperature、top_p、max_new_tokens 调一调再跑，体会区别。")
    print("   这些参数在‘优化推理部署_学习笔记.py’里有逐条详解。")


# ==============================================================================
# 第 12 关：综合大案例 —— 完整流水线一步步打印
# ==============================================================================
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

    print("步骤① 分词（padding 补齐 + truncation 截断 + 返回张量）")
    tokens = tok(texts, padding=True, truncation=True,
                 max_length=64, return_tensors="pt").to(device)
    print("   input_ids 形状：", tuple(tokens["input_ids"].shape),
          "（3条 × 统一长度）")
    print("   attention_mask 每行 1 的个数（真实 token 数）：",
          tokens["attention_mask"].sum(dim=1).tolist())

    print("步骤② 模型前向（eval + no_grad）")
    with torch.no_grad():
        logits = model(**tokens).logits
    print("   logits：\n", logits)

    print("步骤③ 后处理 softmax → 取标签")
    probs = torch.nn.functional.softmax(logits, dim=-1)
    print("   " + "-" * 60)
    for t, p in zip(texts, probs):
        idx = p.argmax().item()
        bar = "█" * int(p.max().item() * 20)
        print(f"   {model.config.id2label[idx]:8s} {p.max().item():.3f} {bar}  {t}")
    print("   " + "-" * 60)
    print("👉 这就是 pipeline('sentiment-analysis') 的完整内部流程。")


# ------------------------------------------------------------------------------
# 调度器：根据命令行参数决定跑哪一关
# ------------------------------------------------------------------------------
CASES = {
    1: case_01, 2: case_02, 3: case_03, 4: case_04, 5: case_05, 6: case_06,
    7: case_07, 8: case_08, 9: case_09, 10: case_10, 11: case_11, 12: case_12,
}

MENU = """\
用法：
  python3 案例闯关_从零理解Pipeline.py <关号>   跑单关，如 1 / 10 / 12
  python3 案例闯关_从零理解Pipeline.py all      跑全部

关卡列表：
  1  分词器三字段         2  句子对token_type_ids   3  padding 填充
  4  truncation 截断      5  底层三步               6  attention_mask 对照
  7  AutoModel对比头      8  logits→softmax→标签    9  pipeline 一行验证
  10 中文情感分析(可跑)   11 生成参数实验            12 综合大案例(完整流水线)
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
            print(f"没有第 {choice} 关，请输入 1~12 或 all。")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        # 方式A：命令行带参数，如  python3 本文件.py 6
        run_choice(args[0])
    else:
        # 方式B：在 PyCharm 里直接点运行 → 弹出提示让你输入关号
        print(MENU)
        while True:
            choice = input("请输入关号（1~12，或 all 全部，回车/q 退出）：")
            if choice.strip().lower() in ("", "q", "quit", "exit"):
                print("已退出。")
                break
            run_choice(choice)
            print()  # 跑完一关空一行，方便继续输入下一关
