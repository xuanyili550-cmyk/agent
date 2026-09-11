"""
================================================================================
 综合项目1 · 智能客服助手（中文版，整合 Ch1 + Ch2 + Ch5 + Ch6 + Ch7 + Ch9）
================================================================================
 一条工单进来，系统自动：判情绪 → 抽实体 → 语义检索 FAQ → 组织回复/转人工。
 全部用【中文模型】，中文工单才处理得准(英文模型喂中文会失准)：
   [Ch1 情感]     uer/roberta-base-finetuned-dianping-chinese：判负面/正面，先分流(急/不满优先)。
   [Ch6+Ch7 NER]  uer/roberta-base-finetuned-cluener2020-chinese：抽人名/公司/地址(打标签、路由、脱敏)。
   [Ch5 语义检索] BAAI/bge-small-zh-v1.5：把工单和 FAQ 嵌成向量，按余弦找最相关(换说法也命中)。
   [Ch2 微调]     意图分类器可用你自己的工单微调(见 ../案例1)；这里先用规则+检索。
   [Ch9 上线]     把 handle() 包成 Gradio(ui) 或 FastAPI，一键成网页/API。

 中文模型的两个坑(和英文版不同)：
   · bge 中文嵌入用 CLS 池化(取 [:,0])，不是 mean 池化；检索时给“查询”加前缀提升召回。
   · cluener 是逐字模型，实体 word 里字间有空格(如 '腾 讯')，要 .replace(' ','')。

 本地跑：python3 项目1_智能客服助手.py            # 处理一批中文工单，出结构化结果
        python3 项目1_智能客服助手.py ui         # 起 Gradio 网页(需 gradio)
================================================================================
"""
import sys
import torch
import torch.nn.functional as F

# 检索“查询”前缀(bge 中文检索技巧：查询加、文档不加)
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


FAQ = [
    ("重置密码", "去 设置 > 安全 > 重置密码，按邮件里的链接操作即可。"),
    ("上传照片闪退", "请把 App 升级到 v3.2 或更高版本，照片上传闪退的问题在该版本已修复。"),
    ("重复扣款退款", "重复扣款会在核实后 3-5 个工作日内原路退回。"),
    ("联系人工客服", "在聊天窗口输入 'agent'，或工作日 9:00-18:00 拨打客服热线。"),
    ("存储空间", "免费账户有 5GB 空间；升级 Pro 可获得 1TB 存储。"),
]

TICKETS = [
    "你们的App一上传照片就崩溃，我iPhone上试了好几次，太气人了！",
    "谢谢，更新之后我的登录问题解决了，客服小李态度也很好！",
    "我通过支付宝被重复扣了两次费，要求马上退款。",
    "嗯……不太确定，可能待会再说吧",
]

CONF_TH, FAQ_TH = 0.80, 0.55   # FAQ 相似度阈值收紧：好评/闲聊类工单没有对口 FAQ 就转人工，别硬答


class Assistant:
    """生产思路：模型惰性加载一次、全程复用(单例)。"""
    def __init__(self):
        from transformers import (AutoTokenizer, AutoModel,
                                  AutoModelForSequenceClassification, pipeline)
        self.dev = pick_device()
        # [Ch1] 中文情感分类器(点评微调；标签 0=负面 1=正面)
        s = "uer/roberta-base-finetuned-dianping-chinese"
        self.sent_tok = AutoTokenizer.from_pretrained(s)
        self.sent = AutoModelForSequenceClassification.from_pretrained(s).to(self.dev).eval()
        # [Ch6+Ch7] 中文 NER(CLUENER；aggregation_strategy=simple 合并子词)
        self.ner = pipeline("token-classification",
                            model="uer/roberta-base-finetuned-cluener2020-chinese",
                            aggregation_strategy="simple",
                            device=0 if self.dev == "cuda" else -1)
        # [Ch5] 中文句向量(bge-small-zh：CLS 池化)
        e = "BAAI/bge-small-zh-v1.5"
        self.emb_tok = AutoTokenizer.from_pretrained(e)
        self.emb = AutoModel.from_pretrained(e).to(self.dev).eval()
        self.faq_vecs = self.embed([a for _, a in FAQ])          # 文档不加前缀

    def sentiment(self, text):
        enc = self.sent_tok(text, return_tensors="pt", truncation=True).to(self.dev)
        with torch.no_grad():
            p = torch.softmax(self.sent(**enc).logits, -1)[0]     # [Ch1] logits→概率
        i = int(p.argmax())
        return ("正面" if i == 1 else "负面"), float(p[i])

    def entities(self, text):
        # [Ch6/7] cluener 逐字模型：word 去掉字间空格
        return [(e["entity_group"], e["word"].replace(" ", "")) for e in self.ner(text)]

    def embed(self, texts, is_query=False):
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]             # 查询加前缀
        enc = self.emb_tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            v = self.emb(**enc).last_hidden_state[:, 0]           # [Ch5] bge 用 CLS 池化(取第0个token)
        return F.normalize(v, p=2, dim=1)                         # 归一化 → 点积=余弦

    def retrieve(self, text):
        sims = (self.embed([text], is_query=True) @ self.faq_vecs.T)[0]   # [Ch5] 余弦相似度
        i = int(sims.argmax())
        return i, float(sims[i])

    def handle(self, text):
        label, conf = self.sentiment(text)
        ents = self.entities(text)
        if conf < CONF_TH:
            return {"text": text, "sentiment": label, "conf": conf, "entities": ents,
                    "action": "转人工", "reason": "情感置信度低"}
        idx, score = self.retrieve(text)
        if score < FAQ_TH:
            return {"text": text, "sentiment": label, "conf": conf, "entities": ents,
                    "action": "转人工", "reason": "无对口 FAQ", "faq_score": score}
        return {"text": text, "sentiment": label, "conf": conf, "entities": ents,
                "action": "自动回复", "faq_score": score, "reply": FAQ[idx][1]}


def run_cli():
    a = Assistant()
    print(f">>> 设备={a.dev}  情感阈值={CONF_TH}  FAQ阈值={FAQ_TH}（全中文模型）\n")
    for t in TICKETS:
        r = a.handle(t)
        # print("─" * 62)
        print(f"工单: {r['text']}")
        print(f"  情感={r['sentiment']}({r['conf']:.2f})  实体={r['entities']}")
        print(f"  决策={r['action']}" + (f"（{r['reason']}）" if 'reason' in r else ""))
        if r["action"] == "自动回复":
            print(f"  命中FAQ({r['faq_score']:.2f}) → {r['reply']}")
    # print("─" * 62)
    print("✅ 中文智能客服全流程跑通：情感(Ch1)+实体(Ch6/7)+语义检索(Ch5)+阈值分流(生产)。")


def run_ui():
    import gradio as gr
    a = Assistant()
    def fn(text):
        r = a.handle(text)
        ents = ", ".join(f"{g}:{w}" for g, w in r["entities"]) or "无"
        out = f"情感: {r['sentiment']} ({r['conf']:.2f})\n实体: {ents}\n决策: {r['action']}"
        if r["action"] == "自动回复":
            out += f"\n回复: {r['reply']}"
        elif "reason" in r:
            out += f"（{r['reason']}）"
        return out
    demo = gr.Interface(fn=fn, inputs=gr.Textbox(label="工单", lines=3),
                        outputs=gr.Textbox(label="处理结果"),
                        title="智能客服助手(中文)",
                        examples=[[t] for t in TICKETS])
    demo.launch()


# ==============================================================================
# 生产要点(上云怎么搭)
# ==============================================================================
# · 意图分类器用你自己的历史工单微调(Ch2/案例1)，比规则准；情感/NER 可换领域微调版。
# · 中文模型选型：情感→点评/京东微调;检索→bge-zh/text2vec;NER→CLUENER/UIE;生成→Qwen/GLM/百川。
#   核心：模型语种要和数据语种一致，别用英文模型跑中文。
# · FAQ 检索上规模用向量数据库(Qdrant/Milvus，见 ../生产架构/生产02)，别用内存。
# · 回复生成可接中文 LLM：把命中的 FAQ 拼进提示喂 vLLM 生成自然语言(见 ../生产架构/生产01)。
# · 上线：handle() 包 FastAPI(/ticket) + Gradio(客服后台)，K8s 部署，监控转人工率/命中率/延迟。

# ==============================================================================
# 面试题(这个项目会被问什么)
# ==============================================================================
# Q: 做中文 NLP 系统模型怎么选？
# A: 情感→中文点评微调；检索→bge-zh/text2vec；NER→CLUENER/UIE；生成→Qwen/GLM。语种必须对齐。
# Q: 客服系统为什么先分流(情感/置信度)再检索？
# A: 生产要保证质量——负面/紧急优先，低置信度转人工避免错误自动决策；能自动答的才走检索，省成本。
# Q: bge 中文嵌入和 all-MiniLM 有什么不同？
# A: bge 常用 CLS 池化(取 [:,0])且查询建议加前缀；MiniLM 用 mean 池化。池化方式搞错会掉召回。
# Q: 这套怎么支撑高并发？
# A: 模型单例+批处理；服务无状态、K8s 多副本+HPA；FAQ 向量放向量库；相同问题结果缓存。

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        run_ui()
    else:
        run_cli()
