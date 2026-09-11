"""
================================================================================
 中文版 · 智能客服全流程（全中文模型：情感 + NER + 语义检索）
================================================================================
 这是 ../案例/综合大demo 的“真中文版”——之前那份因为英文情感模型跑中文会失准，才用了英文；
 现在三个模型全换成中文的，中文工单就能正确处理了：
   [情感] uer/roberta-base-finetuned-dianping-chinese   判负面/正面(急不急、满不满)
   [NER]  uer/roberta-base-finetuned-cluener2020-chinese 抽人名/公司/地址(脱敏、路由)
   [检索] BAAI/bge-small-zh-v1.5                          按意思检索中文 FAQ
 数据流：工单 → 情感分流(低置信转人工) → 抽实体 → 语义检索 FAQ → 自动回复/转人工。
 跑：python3 中文_智能客服全流程.py
================================================================================
"""
import torch
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModel, pipeline)

FAQ = [
    "重置密码：去 设置 > 安全 > 重置密码，按邮件链接操作。",
    "上传照片闪退：请升级到 App v3.2 或更高版本，该问题已修复。",
    "重复扣款：核实后 3-5 个工作日内退还。",
    "联系人工：在聊天里输入 'agent'，或工作日 9-18 点拨打热线。",
]
TICKETS = [
    "你们的App一传照片就崩溃，太气人了！",
    "谢谢，新版本修好了我的登录问题，很满意！",
    "我在腾讯用你们服务被重复扣款了，要退款。",
    "嗯，还行吧",
]
CONF_TH, FAQ_TH = 0.80, 0.45
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class 中文客服:
    def __init__(self):
        s = "uer/roberta-base-finetuned-dianping-chinese"
        self.sent_tok = AutoTokenizer.from_pretrained(s)
        self.sent = AutoModelForSequenceClassification.from_pretrained(s).eval()
        self.ner = pipeline("token-classification",
                            model="uer/roberta-base-finetuned-cluener2020-chinese",
                            aggregation_strategy="simple")
        e = "BAAI/bge-small-zh-v1.5"
        self.emb_tok = AutoTokenizer.from_pretrained(e)
        self.emb = AutoModel.from_pretrained(e).eval()
        self.faq_vecs = self.embed(FAQ)

    def sentiment(self, text):
        enc = self.sent_tok(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            p = torch.softmax(self.sent(**enc).logits, -1)[0]
        i = int(p.argmax())
        return ("正面" if i == 1 else "负面"), float(p[i])

    def entities(self, text):
        return [(e["entity_group"], e["word"].replace(" ", "")) for e in self.ner(text)]

    def embed(self, texts, is_query=False):
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        enc = self.emb_tok(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            v = self.emb(**enc).last_hidden_state[:, 0]      # bge CLS 池化
        return F.normalize(v, p=2, dim=1)

    def handle(self, text):
        label, conf = self.sentiment(text)
        ents = self.entities(text)
        if conf < CONF_TH:
            return {"text": text, "情感": label, "conf": conf, "实体": ents,
                    "决策": "转人工", "原因": "情感置信度低"}
        sims = (self.embed([text], is_query=True) @ self.faq_vecs.T)[0]
        i, score = int(sims.argmax()), float(sims.max())
        if score < FAQ_TH:
            return {"text": text, "情感": label, "conf": conf, "实体": ents,
                    "决策": "转人工", "原因": "无对口FAQ"}
        return {"text": text, "情感": label, "conf": conf, "实体": ents,
                "决策": "自动回复", "相似度": score, "回复": FAQ[i]}


def main():
    kf = 中文客服()
    print(f">>> 全中文模型  情感阈值={CONF_TH}  FAQ阈值={FAQ_TH}\n")
    for t in TICKETS:
        r = kf.handle(t)
        # print("─" * 60)
        print(f"工单: {r['text']}")
        print(f"  情感={r['情感']}({r['conf']:.2f})  实体={r['实体']}")
        print(f"  决策={r['决策']}" + (f"（{r['原因']}）" if '原因' in r else ""))
        if r["决策"] == "自动回复":
            print(f"  命中FAQ({r['相似度']:.2f}) → {r['回复']}")
    # print("─" * 60)
    print("✅ 中文客服全流程跑通：中文情感+中文NER+中文语义检索，三个中文模型协作。")


# 生产要点：· 中文模型选型见 README_中文版说明.py。· 生成回复可接中文 LLM(Qwen/GLM/百川)。
# · 中文分词/实体在垂直领域(医疗/金融)要用领域模型或 UIE 微调。
# 面试：Q 做中文 NLP 系统模型怎么选? A 情感→中文点评微调;检索→bge-zh/text2vec;NER→CLUENER/UIE;
#      生成→Qwen/GLM 等中文 LLM。核心:模型语种要和数据语种一致。
if __name__ == "__main__":
    main()
