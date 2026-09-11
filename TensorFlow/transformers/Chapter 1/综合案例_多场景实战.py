"""
================================================================================
 综合案例 · 多场景实战（4 个场景，每个都是端到端完整流程）
================================================================================
 这是把「Chapter 1」学到的东西用到真实业务里的综合练习。
 每个场景独立、可单独跑，全部能在你的 M4 Mac 上真跑（无需再装任何库）。

 用法：
   python3 综合案例_多场景实战.py           # 看菜单
   python3 综合案例_多场景实战.py 1          # 只跑场景 1
   python3 综合案例_多场景实战.py all         # 全部跑

 四个场景（各自不同的业务与技术点）：
   场景1  中文舆情批量分析系统   —— 批处理 + 情感统计报表（清洗→分词→推理→聚合）
   场景2  智能 FAQ 语义匹配      —— 句向量 + 余弦相似度（把用户问题匹配到最像的FAQ）
   场景3  微调前数据预处理(Ch2)  —— 句子对 + 动态填充 DataCollator（省显存的关键）
   场景4  生产级分类分流服务      —— 置信度阈值 + 低置信度转人工复核
================================================================================
"""

import sys
import torch


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def title(n, text):
    print("\n" + "=" * 72)
    print(f"  场景 {n}：{text}")
    print("=" * 72)


# ==============================================================================
# 场景 1：中文舆情批量分析系统
#   业务：给定一批用户评论/新闻，批量判断情感，产出统计报表（正/负/中占比）。
#   技术串联：文本清洗 → 批量分词(padding) → 一次前向 → softmax → 聚合统计
# ==============================================================================
def scenario_01():
    title(1, "中文舆情批量分析系统（批处理 + 统计报表）")
    import re
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.nn.functional import softmax

    device = pick_device()
    ckpt = "sanshizhang/Chinese-Sentiment-Analysis-Fund-Direction"
    tok = BertTokenizer.from_pretrained(ckpt)
    model = BertForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()
    labels = {0: "负面", 1: "正面", 2: "中性"}

    # 模拟一批舆情数据
    comments = [
        "这家公司业绩暴涨，强烈看好！",
        "基金连续跌了三个月，血亏，退了。",
        "今天大盘平稳，没什么波动。",
        "重大利好消息，明天要涨停！",
        "管理层跑路了，太坑人了。",
        "行情一般，继续观望吧。",
    ]

    def clean(t):  # 只保留中文/数字/字母/常见标点
        return re.sub(r"[^一-鿿\d.a-zA-Z%+\-。！？，、；：]", "", t)

    cleaned = [clean(c) for c in comments]

    # ★ 批量分词：padding=True 把整批补齐成矩形，一次推理更高效
    enc = tok(cleaned, padding=True, truncation=True,
              max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = softmax(model(**enc).logits, dim=1)
    preds = probs.argmax(dim=1).tolist()

    print("逐条结果：")
    for c, p, pr in zip(comments, preds, probs):
        print(f"  [{labels[p]}] {pr[p].item():.3f}  {c}")

    # 聚合成报表
    from collections import Counter
    counter = Counter(labels[p] for p in preds)
    total = len(preds)
    print("\n📊 舆情统计报表：")
    for k in ["正面", "负面", "中性"]:
        n = counter.get(k, 0)
        bar = "█" * int(n / total * 30)
        print(f"  {k}: {n:>2} 条 ({n/total*100:4.1f}%) {bar}")
    print(f"  情绪倾向：{'偏正面' if counter['正面'] > counter['负面'] else '偏负面'}")


# ==============================================================================
# 场景 2：智能 FAQ 语义匹配
#   业务：用户随口问一句，系统从 FAQ 知识库里找出语义最接近的问题并给答案。
#   技术串联：AutoModel 取句向量(mean pooling) → 余弦相似度 → 取 Top-1
#   要点：这里用的是“语义”相似，不是关键词匹配——换个说法也能匹配上。
# ==============================================================================
def scenario_02():
    title(2, "智能 FAQ 语义匹配（句向量 + 余弦相似度）")
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    device = pick_device()
    ckpt = "bert-base-chinese"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt).to(device)
    model.eval()

    # FAQ 知识库（问题 → 答案）
    faq = {
        "如何修改密码？": "在【设置-账号安全】里点击“修改密码”即可。",
        "怎么申请退款？": "进入订单详情页，点击“申请退款”，填写原因提交。",
        "支持哪些付款方式？": "支持微信、支付宝、银行卡三种付款方式。",
        "配送需要多久？": "普通快递 3-5 天，加急次日达。",
    }
    questions = list(faq.keys())

    def embed(texts):
        enc = tok(texts, padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state         # [批, 序列, 768]
        # ★ mean pooling：按 attention_mask 加权平均，得到每句一个向量
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(vec, dim=1)                   # 归一化，方便算余弦

    faq_vecs = embed(questions)

    # 用户用“不一样的说法”来问
    user_queries = ["密码忘了怎么办", "钱多久能退回来", "能用花呗吗"]
    for q in user_queries:
        qv = embed([q])
        sims = (qv @ faq_vecs.T).squeeze(0)              # 余弦相似度
        best = sims.argmax().item()
        print(f"\n用户问：{q}")
        print(f"  最匹配FAQ：{questions[best]}（相似度 {sims[best].item():.3f}）")
        print(f"  回答：{faq[questions[best]]}")
    print("\n👉 注意：用户说法和 FAQ 原文不同，但靠语义向量仍能匹配上。")
    print("   （生产里建议换成专门的句向量模型，如 text2vec / bge，效果更好）")


# ==============================================================================
# 场景 3：微调前数据预处理（Chapter 2 核心：动态填充 DataCollator）
#   业务：微调分类模型前，要把“句子对”数据处理成模型输入 batch。
#   技术要点：为什么用 DataCollatorWithPadding「动态填充」，而不是一开始就
#             padding="max_length" 补到 512 —— 后者浪费大量算力在无意义的 0 上。
# ==============================================================================
def scenario_03():
    title(3, "微调前数据预处理：句子对 + 动态填充 DataCollator")
    from transformers import AutoTokenizer, DataCollatorWithPadding

    ckpt = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(ckpt)

    # 模拟 MRPC 那样的“句子对 + 标签”数据（判断两句是否同义）
    raw = [
        ("The cat sat on the mat.", "A cat is sitting on the mat.", 1),
        ("He bought a new car.", "The weather is nice today.", 0),
        ("I love this movie so much!", "This film is amazing, I adore it.", 1),
    ]

    # ① 只分词、先不填充（truncation 防超长，但不 padding）
    features = []
    for s1, s2, label in raw:
        item = tok(s1, s2, truncation=True)     # 句子对：一次传两句
        item["labels"] = label                  # 训练需要标签
        features.append(item)

    print("① 分词后（未填充）各样本长度：",
          [len(f["input_ids"]) for f in features])

    # ② 动态填充：DataCollator 只把“本 batch”补齐到批内最长，而非固定 512
    collator = DataCollatorWithPadding(tokenizer=tok, return_tensors="pt")
    batch = collator(features)
    print("② 动态填充后 input_ids 形状：", tuple(batch["input_ids"].shape),
          "（补到本批最长，不浪费）")
    print("   token_type_ids 存在？", "token_type_ids" in batch,
          "（句子对任务用它区分A/B句）")
    print("   labels：", batch["labels"].tolist())

    # ③ 对比：如果一上来就 padding='max_length' 补到 512 有多浪费
    fixed = tok([r[0] for r in raw], [r[1] for r in raw],
                padding="max_length", max_length=512, truncation=True,
                return_tensors="pt")
    dyn_tokens = batch["input_ids"].numel()
    fixed_tokens = fixed["input_ids"].numel()
    print(f"\n📊 算力对比（越少越好）：")
    print(f"   动态填充总token数：{dyn_tokens}")
    print(f"   固定填充到512总token数：{fixed_tokens}")
    print(f"   固定填充浪费了 {fixed_tokens/dyn_tokens:.1f} 倍算力在无意义的 0 上！")
    print("👉 这就是 Chapter 2 为什么强调用 DataCollatorWithPadding 做动态填充。")


# ==============================================================================
# 场景 4：生产级分类分流服务（置信度阈值 + 转人工）
#   业务：自动分类，但对“模型没把握”的样本不硬判，转人工复核，保证质量。
#   技术串联：批量推理 → softmax 得置信度 → 按阈值分流（自动处理 / 人工队列）
# ==============================================================================
def scenario_04():
    title(4, "生产级分类分流服务（置信度阈值 + 转人工复核）")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = pick_device()
    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    model.eval()

    THRESHOLD = 0.90   # 置信度阈值：低于此值不自动判，转人工

    incoming = [
        "Absolutely fantastic, best purchase ever!",
        "Worst experience of my life, do not buy.",
        "It's fine I guess, does the job.",          # 模糊，置信度可能不高
        "Not bad, but could be better honestly.",    # 模糊
        "I am extremely happy with this product!",
    ]

    enc = tok(incoming, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**enc).logits, dim=-1)
    conf, pred = probs.max(dim=-1)

    auto, manual = [], []
    for text, c, p in zip(incoming, conf.tolist(), pred.tolist()):
        label = model.config.id2label[p]
        if c >= THRESHOLD:
            auto.append((text, label, c))
        else:
            manual.append((text, label, c))

    print(f"✅ 自动处理（置信度 ≥ {THRESHOLD}）：")
    for t, l, c in auto:
        print(f"   [{l}] {c:.3f}  {t}")
    print(f"\n⚠️  转人工复核（置信度 < {THRESHOLD}）：")
    for t, l, c in manual:
        print(f"   [模型猜{l}] {c:.3f}  {t}")
    print(f"\n📊 自动化率：{len(auto)}/{len(incoming)} = "
          f"{len(auto)/len(incoming)*100:.0f}%（其余转人工，保证质量）")


# ------------------------------------------------------------------------------
# 调度器
# ------------------------------------------------------------------------------
SCENARIOS = {1: scenario_01, 2: scenario_02, 3: scenario_03, 4: scenario_04}

MENU = """\
用法：
  python3 综合案例_多场景实战.py <场景号>   跑单个，如 1 / 2 / 3 / 4
  python3 综合案例_多场景实战.py all        跑全部

场景列表：
  1  中文舆情批量分析系统   （批处理 + 情感统计报表）
  2  智能 FAQ 语义匹配      （句向量 + 余弦相似度）
  3  微调前数据预处理(Ch2)  （句子对 + 动态填充 DataCollator）
  4  生产级分类分流服务      （置信度阈值 + 转人工复核）
"""

def run_choice(choice: str):
    """根据输入跑对应场景：数字=单个，all=全部。"""
    choice = choice.strip().lower()
    if choice == "all":
        for n in sorted(SCENARIOS):
            SCENARIOS[n]()
    else:
        try:
            SCENARIOS[int(choice)]()
        except (ValueError, KeyError):
            print(f"没有场景 {choice}，请输入 1~4 或 all。")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        # 方式A：命令行带参数，如  python3 本文件.py 3
        run_choice(args[0])
    else:
        # 方式B：在 PyCharm 里直接点运行 → 弹出提示让你输入场景号
        print(MENU)
        while True:
            choice = input("请输入场景号（1~4，或 all 全部，回车/q 退出）：")
            if choice.strip().lower() in ("", "q", "quit", "exit"):
                print("已退出。")
                break
            run_choice(choice)
            print()
