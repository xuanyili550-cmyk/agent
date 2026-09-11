"""
================================================================================
 综合实战 · Chapter 1~6 全链路（生产级 + 加强记忆）· 智能客服工单处理系统
================================================================================
 这是把 Ch1~6 揉进「一个能上线的小系统」的综合练习。一条工单进来，系统要：
   ① 情感分类（急/不急）        ② 低置信度自动转人工（生产级质量兜底）
   ③ 语义检索最相关 FAQ 自动回复  ④ 用 offset 把关键词高亮回原文
 每一步都标注【用到第几章】+【为什么这样算】，读完能把 1~6 串成一条线。

 —— 两种用法 ——
   A) 当“生产参考”读：本文件已填好、可直接运行
        python3 综合实战_1到6_挖空练习.py           # 处理一批工单，打印结构化报告
        python3 综合实战_1到6_挖空练习.py demo        # 同上
   B) 当“加强记忆”练：把每个「练习N」那行遮住，凭记忆默写，再运行对答案
        （关键考点都用「练习N：」标出；底部有「考点速记 / 答案区」）

 —— 系统数据流（生产级分层）——
   工单文本
     │  [Ch2/Ch6] 批量分词(padding+truncation) → [Ch1/Ch2] 模型 → [Ch2] softmax
     ▼
   情感标签 + 置信度 ──低于阈值──▶ [生产] 转人工队列
     │ 高置信度
     ▼
   [Ch5] 句向量(mean-pooling) → 归一化 → 余弦相似度 → 命中 FAQ 自动回复
     │
     ▼
   [Ch6] offset 关键词高亮 → 汇总成结构化结果(dict) → 报告

 —— 会用到的小模型（首次自动下载并缓存）——
   distilbert-base-uncased-finetuned-sst-2-english   情感分类  ~268MB
   sentence-transformers/all-MiniLM-L6-v2            句向量    ~90MB
================================================================================
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field


# ==============================================================================
# 0) 生产级基建：配置 + 设备选择 + 模型惰性单例
# ==============================================================================
@dataclass
class Config:
    """把“会变的参数”集中到一处，是生产代码的基本素养（改一处即可，不散落魔法数）。"""
    clf_ckpt: str = "distilbert-base-uncased-finetuned-sst-2-english"
    embed_ckpt: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 128          # 截断上限：过长文本切掉尾部，防止超过模型位置上限报错
    # 置信度阈值：低于它就转人工。
    # 【为什么要阈值】生产系统里“错误的自动处理”比“转人工”代价大得多，
    #  用置信度当“敢不敢自动决策”的闸门，是最常见的质量兜底手段。
    confidence_threshold: float = 0.90
    # 语义命中阈值：相似度太低说明 FAQ 里没有对口答案，宁可不乱答。
    faq_match_threshold: float = 0.40
    highlight_words: set = field(default_factory=lambda: {
        "crash", "crashing", "crashes", "charged", "twice", "frustrating", "refund",
    })


def pick_device() -> str:
    """Ch1 学过：优先 NVIDIA(cuda) / 苹果(mps)，都没有退回 cpu。"""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Models:
    """惰性加载 + 复用模型。
    【为什么这样做】加载模型(几百 MB)很慢，生产里绝不能每来一条工单就重载一次；
    这里第一次用到才加载、之后一直复用（单例思想）。
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = pick_device()
        self._clf_tok = self._clf = None
        self._emb_tok = self._emb = None

    def classifier(self):
        if self._clf is None:
            from transformers import (AutoTokenizer,
                                      AutoModelForSequenceClassification)
            # 练习1：加载“分词器”和“带分类头的模型”（AutoXxx 的哪个方法？）
            self._clf_tok = AutoTokenizer.from_pretrained(self.cfg.clf_ckpt)
            self._clf = AutoModelForSequenceClassification.from_pretrained(
                self.cfg.clf_ckpt).to(self.device).eval()
        return self._clf_tok, self._clf

    def embedder(self):
        if self._emb is None:
            from transformers import AutoTokenizer, AutoModel
            self._emb_tok = AutoTokenizer.from_pretrained(self.cfg.embed_ckpt)
            # AutoModel(无任务头)：我们只要它输出的 token 向量，自己做池化
            self._emb = AutoModel.from_pretrained(
                self.cfg.embed_ckpt).to(self.device).eval()
        return self._emb_tok, self._emb


# ==============================================================================
# 1) 情感分类（Ch1 高层概念 + Ch2 内部机制：分词 → 模型 → softmax）
# ==============================================================================
def classify_sentiment(texts: list[str], M: Models) -> list[dict]:
    import torch
    tok, model = M.classifier()

    # 练习2：批量分词——补齐(padding)+截断(truncation)+返回 PyTorch 张量
    # 【为什么 padding】一个 batch 要堆成规整张量(B×L)，必须等长 → 短的补 [PAD]。
    # 【为什么 truncation】超过模型最大位置数会直接报错 → 统一截到 max_length。
    enc = tok(texts, padding=True, truncation=True,
              max_length=M.cfg.max_length, return_tensors="pt").to(M.device)

    # 练习3：前向推理，取 logits；推理阶段关梯度省显存/加速
    # 【为什么 no_grad】推理不反向传播，不需要建计算图，关掉它显存更省、速度更快。
    with torch.no_grad():
        logits = model(**enc).logits

    # 练习4：logits → 概率（最后一维做 softmax）
    # 【为什么用 softmax 而不是直接比 logits】
    #   logits 是任意实数，没法当“置信度”；softmax 把它压成 [0,1] 且每行和为 1 的概率，
    #   数值可解释、可跨样本比较、可设阈值。注意 argmax 结果不变(softmax 单调)，
    #   我们要的是那个“概率值”本身，用来做转人工的判断。
    probs = torch.softmax(logits, dim=-1)

    results = []
    for text, p in zip(texts, probs):
        label_id = int(p.argmax())            # 概率最大的类别下标
        results.append({
            "text": text,
            "label": model.config.id2label[label_id],   # id → 人类可读标签
            "confidence": float(p[label_id]),           # 该类别的概率=置信度
        })
    return results


# ==============================================================================
# 2) 语义检索（Ch5：嵌入 → mean-pooling → 归一化 → 余弦相似度）
# ==============================================================================
def embed_texts(texts: list[str], M: Models):
    import torch
    import torch.nn.functional as F
    tok, model = M.embedder()

    enc = tok(texts, padding=True, truncation=True,
              max_length=M.cfg.max_length, return_tensors="pt").to(M.device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state       # (B, L, H)：每个 token 一个向量

    # 练习5：mean-pooling，但要用 attention_mask 加权平均
    # 【为什么必须用 mask 加权，而不是直接 .mean(1)】
    #   padding 位置([PAD])也有向量，但它是“凑长度的废料”，没有语义。
    #   直接对整行平均会把这些废料算进去，句子越短被稀释越狠 → 句向量失真。
    #   正确做法：只对“真实 token”求和再除以“真实 token 个数”。
    mask = enc["attention_mask"].unsqueeze(-1).float()   # (B, L, 1)
    summed = (out * mask).sum(dim=1)                      # 真实 token 向量求和
    counts = mask.sum(dim=1).clamp(min=1e-9)             # 真实 token 个数(防除0)
    sentence = summed / counts                           # (B, H) 一句一个向量

    # 练习6：L2 归一化
    # 【为什么归一化后点积 = 余弦相似度】
    #   余弦相似度 = (a·b)/(|a||b|)，只看“方向”(语义)不看“长度”(篇幅)。
    #   把每个向量先除以自己的模长(变成单位向量)，之后两两点积就正好等于余弦值，
    #   于是“矩阵乘法一次算完所有相似度”，又快又对。
    return F.normalize(sentence, p=2, dim=1)


def route_to_faq(query: str, faq_vecs, faq: list[tuple[str, str]], M: Models) -> dict:
    q = embed_texts([query], M)                  # (1, H)
    # 练习7：相似度矩阵 = 查询向量 · FAQ向量转置（都已归一化 → 点积即余弦）
    sims = (q @ faq_vecs.T)[0]                    # (FAQ数,)
    best = int(sims.argmax())
    best_score = float(sims[best])
    # 【为什么还要个 faq_match_threshold】相似度最高≠一定对口；
    #  太低说明知识库里根本没有对应答案，生产上宁可“转人工”也别硬答错。
    hit = best_score >= M.cfg.faq_match_threshold
    return {
        "faq_question": faq[best][0] if hit else None,
        "faq_answer": faq[best][1] if hit else None,
        "faq_score": best_score,
        "faq_hit": hit,
    }


# ==============================================================================
# 3) 关键词高亮（Ch6：快速分词器 offset + word_ids 把子词对回原文整词）
# ==============================================================================
def highlight(text: str, M: Models) -> str:
    from transformers import AutoTokenizer
    # 用快速分词器（bert-base-cased）拿 offset；它很小、只下分词器文件
    tok = AutoTokenizer.from_pretrained("bert-base-cased")

    # 练习8：编码并要求返回 offset 映射
    # 【为什么要 offset】模型/分词是按 token 工作的，一个词可能被切成多个子词
    #  (frustrating → fr ##ust ##rating)；offset 记录每个 token 的原文字符区间 [start,end)，
    #  是把“token 级结果”对齐回“原文字符位置”的唯一桥梁(NER/QA/高亮全靠它)。
    enc = tok(text, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    word_ids = enc.word_ids()

    # 练习9：用 word_ids 把“属于同一个单词的子词”聚合成整词的字符区间
    # 【为什么先聚合成整词】否则 frustrating 被切碎后，任何单个子词都 != "frustrating"，
    #  按子词匹配会漏词；先按 word_ids 合并出整词区间，再拿整词去比关键词才准。
    spans: dict[int, list[int]] = {}
    for wid, (s, e) in zip(word_ids, offsets):
        if wid is None:                     # 特殊标记([CLS]/[SEP])没有单词号
            continue
        if wid not in spans:
            spans[wid] = [s, e]
        else:
            spans[wid][1] = e               # 同词的后续子词，把右边界推到末尾

    marked = list(text)
    for s, e in spans.values():
        if text[s:e].lower() in M.cfg.highlight_words:
            marked[s] = "【" + marked[s]
            marked[e - 1] = marked[e - 1] + "】"
    return "".join(marked)


# ==============================================================================
# 4) 编排：把 1~6 串成一条生产流水线
# ==============================================================================
# 迷你 FAQ 知识库（问题 → 标准答复）
FAQ: list[tuple[str, str]] = [
    ("How can I reset my password?",
     "Go to Settings > Security > Reset Password, then follow the email link."),
    ("The app crashes when uploading photos.",
     "Please update to v3.2 or later; a photo-upload crash was fixed there."),
    ("I was charged twice for one order.",
     "We'll refund the duplicate charge within 3-5 business days after verification."),
    ("How do I contact a human agent?",
     "Type 'agent' in the chat, or call our hotline 9am-6pm."),
]

TICKETS = [
    "The app keeps crashing every time I try to upload a photo, this is so frustrating!",
    "I was charged twice for the same order, I want a refund now.",
    "Thanks a lot, the new update fixed my login problem. Great support!",
    "hmmm ok",   # 故意给一条“模型也拿不准”的短工单，触发“转人工”
]


def process(tickets: list[str], M: Models, faq_vecs) -> list[dict]:
    """一条工单走完全链路，产出结构化结果（生产系统都以结构化 dict/JSON 交付）。"""
    sentiments = classify_sentiment(tickets, M)   # 批量分类(一次前向，省算力)
    reports = []
    for s in sentiments:
        text = s["text"]
        # —— 生产级质量闸门：置信度不够就转人工，不硬做后续自动决策 ——
        if s["confidence"] < M.cfg.confidence_threshold:
            reports.append({
                **s, "action": "转人工", "reason": "置信度低于阈值",
                "highlight": highlight(text, M),
            })
            continue
        faq = route_to_faq(text, faq_vecs, FAQ, M)
        reports.append({
            **s,
            "action": "自动回复" if faq["faq_hit"] else "转人工",
            "reason": "命中 FAQ" if faq["faq_hit"] else "无对口 FAQ",
            "highlight": highlight(text, M),
            **faq,
        })
    return reports


def demo():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("需要 transformers + torch：pip install transformers torch"); return

    cfg = Config()
    M = Models(cfg)
    print(f"设备 = {M.device}    分类阈值={cfg.confidence_threshold}  FAQ阈值={cfg.faq_match_threshold}\n")

    # FAQ 向量可以“离线预先算好并缓存”，线上只算查询向量 → 这才是可扩展的检索做法
    faq_vecs = embed_texts([q for q, _ in FAQ], M)

    reports = process(TICKETS, M, faq_vecs)
    for i, r in enumerate(reports, 1):
        print("─" * 72)
        print(f"工单{i}: {r['text']}")
        print(f"  情感 = {r['label']}  置信度 = {r['confidence']:.3f}")
        print(f"  高亮 = {r['highlight']}")
        print(f"  决策 = {r['action']}（{r['reason']}）")
        if r.get("faq_hit"):
            print(f"  命中FAQ(相似度{r['faq_score']:.3f}) = {r['faq_question']}")
            print(f"  自动回复 = {r['faq_answer']}")
    print("─" * 72)
    print("自检：工单1/2 应高置信度NEGATIVE并命中FAQ；工单3 POSITIVE；工单4 'hmmm ok' 多半转人工。")


# ==============================================================================
# 菜单
# ==============================================================================
if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if arg in ("demo", "all", "run"):
        demo()
    else:
        print(__doc__)


# ==============================================================================
#  考点速记 / 答案区（把上面「练习N」遮住默写，再来这里对答案）
# ------------------------------------------------------------------------------
#  1: AutoTokenizer.from_pretrained(ckpt) / AutoModelForSequenceClassification.from_pretrained(ckpt)
#  2: tok(texts, padding=True, truncation=True, max_length=..., return_tensors="pt")
#  3: with torch.no_grad(): logits = model(**enc).logits
#  4: probs = torch.softmax(logits, dim=-1)          # 为什么：logits→可解释概率，用于阈值
#  5: (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # 为什么：mask 去掉 PAD 废料
#  6: F.normalize(sentence, p=2, dim=1)              # 为什么：归一化后点积=余弦相似度
#  7: sims = (q @ faq_vecs.T)[0]                      # 为什么：单位向量点积=余弦
#  8: tok(text, return_offsets_mapping=True)         # 为什么：offset 把 token 对回原文字符
#  9: 用 word_ids() 聚合子词→整词区间             # 为什么：整词才能匹配关键词，不漏切碎的词
#
#  一句话把 1~6 串起来：
#   Ch1 pipeline 是门面 → Ch2 拆开是“分词→模型→softmax” → Ch6 分词器提供 padding/截断/offset
#   → Ch3 的(微调)模型负责分类并给置信度 → Ch5 语义检索用 mean-pooling+余弦找对口 FAQ。
# ==============================================================================
