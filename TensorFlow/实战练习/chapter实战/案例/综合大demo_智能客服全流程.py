"""
================================================================================
 综合大 demo · 智能客服全流程（串起 Ch1 / Ch5 / Ch6 / Ch9，指向 Ch11 / Ch12）
================================================================================
 一个端到端的小产品：用户发来一句话 → 系统判情绪 → 语义检索知识库 → 组织回复 → 决定是否转人工。
 这是把前面各章“串成一个完整流程”的大 demo(照着手写一遍，就懂各章怎么协作)。

 数据流(每一步标了用哪章、为什么)：
   用户消息
     │ [Ch6 分词器 + Ch1 模型] 情感分类(distilbert-sst2)：判断急不急/满不满意
     │   为什么：先分流——负面/紧急的优先处理，正面的可自动答。
     ▼
   情感 + 置信度 ──低置信度──▶ [生产] 转人工(不硬答)
     │ 高置信度
     ▼ [Ch5 语义检索 + Ch6 嵌入] 从 FAQ 知识库按“意思”找最相关条目
     │   为什么：LLM 不知道你的私有知识；先检索(RAG 前半段)才能准确、可引用。
     ▼
   命中 FAQ ──相似度太低──▶ 转人工(知识库没有对口答案)
     │ 命中
     ▼ 组织回复(真实项目里把 FAQ 拼进提示喂 LLM 生成自然语言，见文末 Ch11/Ch12 接入点)
   结构化结果(供上线用；[Ch9] 可用 Gradio 包成网页/API)

 直接运行：python3 综合大demo_智能客服全流程.py
   纯推理(不训练)，用现成小模型，几秒；下 distilbert-sst2 + MiniLM 各一次。
================================================================================
"""
import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForSequenceClassification)


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


# FAQ 知识库(问题不重要，检索的是答案文本)。
# 注意：情感模型 distilbert-sst2 是英文的，所以这里工单/FAQ 都用英文(三个模型语言对齐才准)。
#      要做中文：把情感模型换成多语言的(如 lxyuan/distilbert-...-sentiments-student)+
#      嵌入换多语言(paraphrase-multilingual-MiniLM)，其余流程不变。
FAQ = [
    ("reset password", "Go to Settings > Security > Reset Password and follow the email link."),
    ("app crash on upload", "Please upgrade to app v3.2 or later; the crash was fixed there."),
    ("double charge", "Duplicate charges are refunded within 3-5 business days after verification."),
    ("contact agent", "Type 'agent' in the chat, or call the hotline from 9am to 6pm."),
]

TICKETS = [
    "Your app keeps crashing every time I upload a photo, so frustrating!",
    "Thanks, the new version fixed my login problem. Great job!",
    "I was charged twice and I want a refund.",
    "hmm ok i guess",
]

# [知识点·置信度分级/兜底] 为什么要阈值分级，而不是“模型说啥就是啥”？
#   模型永远会给一个答案，但不代表它有把握。生产要按“置信度”分级处理：
#     高置信 → 自动答(省人力)；中低置信/无对口知识 → 转人工(别硬答，硬答=事故)。
#   这就是“宁可转人工，也别自信地答错”——错误答案的代价远大于转人工。阈值靠验证集/线上数据校准，
#   还常做三级(自动答 / 建议答案给人工确认 / 直接转人工)，在“自动化率”和“错误率”间权衡。
CONF_THRESHOLD = 0.85     # 情感置信度阈值：低于就转人工
FAQ_THRESHOLD = 0.35      # 语义相似度阈值：低于就转人工


class Pipeline:
    """生产思路：模型只加载一次、全程复用(惰性单例)。
    [知识点·模型单例] 为什么：加载模型(读权重、上 GPU)是秒级重操作，每来一个请求都加载会拖垮服务、
    还爆显存。所以进程启动时加载一次、常驻内存，所有请求复用同一实例(Web 服务里放全局/依赖注入)。"""
    def __init__(self):
        self.device = pick_device()
        # [Ch1/Ch6] 情感分类器
        self.clf_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.clf = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english").to(self.device).eval()
        # [Ch5/Ch6] 句向量(英文用 all-MiniLM 即可；中文换 paraphrase-multilingual-MiniLM)
        emb = "sentence-transformers/all-MiniLM-L6-v2"
        self.emb_tok = AutoTokenizer.from_pretrained(emb)
        self.emb = AutoModel.from_pretrained(emb).to(self.device).eval()
        self.faq_vecs = self.embed([a for _, a in FAQ])   # 知识库向量预先算好

    def sentiment(self, text):
        # [Ch6 分词] → [Ch1 模型] → [Ch1 后处理 softmax]
        enc = self.clf_tok(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            p = torch.softmax(self.clf(**enc).logits, -1)[0]
        i = int(p.argmax())
        return self.clf.config.id2label[i], float(p[i])

    def embed(self, texts):
        enc = self.emb_tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.emb(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # [Ch5] mask 加权 mean 池化
        return F.normalize(v, p=2, dim=1)                       # 归一化 → 点积=余弦

    def retrieve(self, text):
        sims = (self.embed([text]) @ self.faq_vecs.T)[0]        # [Ch5] 余弦相似度
        best = int(sims.argmax())
        return best, float(sims[best])

    def handle(self, text):
        label, conf = self.sentiment(text)
        if conf < CONF_THRESHOLD:                    # [生产] 拿不准就转人工
            return {"text": text, "sentiment": label, "conf": conf, "action": "转人工",
                    "reason": "情感置信度低"}
        idx, score = self.retrieve(text)
        if score < FAQ_THRESHOLD:                    # 知识库没对口答案
            return {"text": text, "sentiment": label, "conf": conf, "action": "转人工",
                    "reason": "无对口 FAQ", "faq_score": score}
        return {"text": text, "sentiment": label, "conf": conf, "action": "自动回复",
                "faq_score": score, "reply": FAQ[idx][1]}


def main():
    pipe = Pipeline()
    print(f">>> 设备={pipe.device}  情感阈值={CONF_THRESHOLD}  FAQ阈值={FAQ_THRESHOLD}\n")
    for t in TICKETS:
        r = pipe.handle(t)
        # print("─" * 60)
        print(f"工单: {r['text']}")
        print(f"  情感={r['sentiment']}({r['conf']:.2f})  决策={r['action']}"
              + (f"（{r['reason']}）" if 'reason' in r else ""))
        if r["action"] == "自动回复":
            print(f"  命中FAQ(相似度{r['faq_score']:.2f}) → 回复: {r['reply']}")
    # print("─" * 60)
    print("""
✅ 全流程跑通：分词(Ch6)→情感(Ch1)→阈值分流(生产)→语义检索(Ch5)→组织回复。
   ── 还能怎么升级(各章接入点)──
   · [Ch11 LoRA] 把“组织回复”换成一个 LoRA 微调过的小 LLM，用检索到的 FAQ 生成自然语言回答。
   · [Ch12 GRPO] 用“用户是否满意/是否解决”当奖励，GRPO 进一步优化回复质量。
   · [Ch9 Gradio] 把 pipe.handle 包成 gr.Interface，一行 launch 成网页/API 上线。
   · [Ch7 训练工程] 情感/意图模型可用你自己的工单数据微调(案例1 那套流程)。
""")


# ==============================================================================
# 面试题(端到端 LLM 应用 / 系统设计)
# ==============================================================================
# Q1: 给你一个客服业务，怎么从 0 搭这套系统？
# A : ①定任务和指标(自动化率/准确率/满意度) ②先分流(情感/意图分类,分优先级) ③RAG 接私有知识
#     ④组织回复(检索到的 FAQ 拼提示喂 LLM) ⑤置信度分级兜底(拿不准转人工) ⑥部署(单例+网关+监控)
#     ⑦灰度上线持续迭代。能用现成模型+提示+RAG 就别急着微调。
# Q2: 为什么要置信度分级 + 转人工兜底，不让模型每条都自动答？
# A : 模型总会给答案但不代表有把握;错误答案的代价远高于转人工。按置信度分级(高→自动、中低/无对口
#     知识→转人工),在“自动化率”和“错误率”间权衡。阈值用验证集/线上数据校准。
# Q3: 生产上模型怎么加载和复用？为什么不能每请求加载？
# A : 加载=读权重+上 GPU,秒级重操作;每请求加载会拖垮延迟、爆显存。启动时加载一次常驻(单例/依赖注入),
#     所有请求复用;服务无状态好水平扩容。
# Q4: 多个模型语言不一致会怎样？本 demo 为什么全用英文？
# A : 情感(英文 sst2)+嵌入模型语言要对齐,否则中文进英文模型→乱码般的表示、判断/检索全不准。
#     做中文要整条链都换多语言模型(多语言情感 + paraphrase-multilingual-MiniLM),流程不变。
# Q5: 这套 demo 上生产还差什么？(扩展点)
# A : 把“组织回复”换成 LoRA 微调的小 LLM 生成自然语言(Ch11);用满意度当奖励 GRPO 优化(Ch12);
#     Gradio/FastAPI 包成服务(Ch9);加日志/监控/限流/缓存/灰度;向量库换 Qdrant/Milvus 支持规模。


if __name__ == "__main__":
    main()
