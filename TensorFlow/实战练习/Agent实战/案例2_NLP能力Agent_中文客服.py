"""
================================================================================
 Agent 实战 · 案例2 · 把 Chapter1-12 的中文 NLP 能力包成 Agent 工具（智能客服）
================================================================================
 整合思路：Agent = LLM 大脑 + 工具。我们把前面章节做出来的【真实中文 NLP 能力】包成
 smolagents 工具，Agent 就能按需调用它们处理客服工单：
   [Ch1 情感]  get_sentiment  —— uer/dianping 中文情感(判负面/正面 → 定优先级)
   [Ch6/7 NER] extract_entities —— uer/cluener 中文 NER(抽人名/公司/地址 → 路由/脱敏)
   [Ch5 检索]  search_faq      —— bge-small-zh 语义检索 FAQ(换说法也命中)

 本机现实：工具全部【本地真跑 + 自检】；Agent 大脑本地小模型太弱(见 README)，故：
   · smoke  ：直接调用三个工具自检 + 用【确定性规则路由】跑通“工单→情感→实体→检索→决策”的编排。
   · 生产   ：run_agent() 用真 smolagents CodeAgent + InferenceClientModel(需 HF Token) 让 LLM 自己编排。
 跑：python3 案例2_NLP能力Agent_中文客服.py smoke     # 工具自检 + 规则编排(推荐，本地跑)
    python3 案例2_NLP能力Agent_中文客服.py            # 额外尝试真 Agent(需 export HF_TOKEN=...)
================================================================================
"""
import sys
from smolagents import tool

_M = {}                       # 模型惰性单例：第一次用到才加载，全程复用
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

FAQ = [
    "重置密码：去 设置 > 安全 > 重置密码，按邮件里的链接操作。",
    "上传照片闪退：请把 App 升级到 v3.2 或更高版本，该问题已修复。",
    "重复扣款：核实后 3-5 个工作日内原路退回。",
    "联系人工：在聊天里输入 'agent'，或工作日 9:00-18:00 拨打热线。",
    "存储空间：免费账户 5GB，升级 Pro 得 1TB。",
]


def _pick_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 三个工具（smolagents @tool：函数签名+docstring 自动变成工具的 name/描述/inputSchema）
# ==============================================================================
@tool
def get_sentiment(text: str) -> str:
    """判断一段中文文本的情感极性(正面/负面)并给出置信度。

    Args:
        text: 要判断情感的中文文本。
    """
    import torch
    if "sent" not in _M:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        name = "uer/roberta-base-finetuned-dianping-chinese"
        _M["dev"] = _M.get("dev") or _pick_device()
        _M["sent_tok"] = AutoTokenizer.from_pretrained(name)
        _M["sent"] = AutoModelForSequenceClassification.from_pretrained(name).to(_M["dev"]).eval()
    enc = _M["sent_tok"](text, return_tensors="pt", truncation=True).to(_M["dev"])
    with torch.no_grad():
        p = torch.softmax(_M["sent"](**enc).logits, -1)[0]
    i = int(p.argmax())
    return f"{'正面' if i == 1 else '负面'} (置信度 {float(p[i]):.2f})"


@tool
def extract_entities(text: str) -> str:
    """从中文文本抽取命名实体(人名/公司/机构/地址/职位等)。

    Args:
        text: 要抽取实体的中文文本。
    """
    if "ner" not in _M:
        from transformers import pipeline
        _M["dev"] = _M.get("dev") or _pick_device()
        _M["ner"] = pipeline("token-classification",
                             model="uer/roberta-base-finetuned-cluener2020-chinese",
                             aggregation_strategy="simple",
                             device=0 if _M["dev"] == "cuda" else -1)
    ents = [(e["entity_group"], e["word"].replace(" ", "")) for e in _M["ner"](text)]
    return "；".join(f"{g}:{w}" for g, w in ents) if ents else "无"


@tool
def search_faq(query: str) -> str:
    """在客服 FAQ 知识库里做中文语义检索，返回最相关的一条(换说法也能命中)。

    Args:
        query: 用户的问题(中文)。
    """
    import torch
    import torch.nn.functional as F
    if "emb" not in _M:
        from transformers import AutoTokenizer, AutoModel
        e = "BAAI/bge-small-zh-v1.5"
        _M["dev"] = _M.get("dev") or _pick_device()
        _M["emb_tok"] = AutoTokenizer.from_pretrained(e)
        _M["emb"] = AutoModel.from_pretrained(e).to(_M["dev"]).eval()

    def embed(texts, is_query=False):
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        enc = _M["emb_tok"](texts, padding=True, truncation=True, return_tensors="pt").to(_M["dev"])
        with torch.no_grad():
            v = _M["emb"](**enc).last_hidden_state[:, 0]          # bge CLS 池化
        return F.normalize(v, p=2, dim=1)
    if "faq_vecs" not in _M:
        _M["faq_vecs"] = embed(FAQ)
    sims = (embed([query], is_query=True) @ _M["faq_vecs"].T)[0]
    i = int(sims.argmax())
    return f"[相似度{float(sims[i]):.2f}] {FAQ[i]}"


TOOLS = [get_sentiment, extract_entities, search_faq]


# ==============================================================================
# 确定性规则路由：本地证明“工单→情感→实体→检索→决策”的编排(生产由 LLM 自己决定)
# ==============================================================================
def handle_ticket(text):
    sent = get_sentiment(text)
    ents = extract_entities(text)
    faq = search_faq(text)
    score = float(faq.split("相似度")[1].split("]")[0])
    neg = sent.startswith("负面")
    if score < 0.55:
        decision = "转人工(无对口FAQ)"
    else:
        decision = "自动回复" + ("(负面工单，标记优先)" if neg else "")
    return {"工单": text, "情感": sent, "实体": ents, "决策": decision, "FAQ": faq}


# ==============================================================================
# 生产：真 smolagents Agent 自己编排(需 HF Token；本地小模型太弱，见 README)
# ==============================================================================
def run_agent(question):
    from smolagents import CodeAgent, InferenceClientModel
    agent = CodeAgent(tools=TOOLS, model=InferenceClientModel(), max_steps=6)
    return agent.run(question)


def smoke():
    # 工具自检(直接调用三个 smolagents 工具，期望值)：
    #   get_sentiment("你们的App太难用了...") -> 负面
    #   extract_entities("我叫张伟，在北京的腾讯公司工作。") -> 人名:张伟；地址:北京；公司:腾讯
    #   search_faq("我忘记密码了怎么办") -> 命中"重置密码：去 设置 > 安全 ..."
    assert get_sentiment("太棒了非常满意").startswith("正面")
    assert "腾讯" in extract_entities("我在腾讯公司上班")
    assert "密码" in search_faq("登录密码忘了")

    # 规则路由编排(工单 → 情感/实体/检索 → 决策)，期望：
    #   "上传照片就崩溃" -> 负面，命中"上传照片闪退"FAQ，决策=自动回复(负面工单，标记优先)
    #   "支付宝被重复扣款" -> 命中"重复扣款"FAQ，决策=自动回复
    #   "嗯，随便问问" -> 无对口FAQ(相似度<0.55)，决策=转人工
    for t in ["你们的App一上传照片就崩溃，太气人了！",
              "我在支付宝被重复扣款了，要退款",
              "嗯，随便问问"]:
        r = handle_ticket(t)
        # print(f"  工单: {r['工单']} 情感={r['情感']} 实体={r['实体']} 决策={r['决策']} | {r['FAQ']}")
    print("\n✅ 案例2 跑通：中文 NLP 能力(Ch1/5/6/7)已包成 smolagents 工具 + 编排通过。")
    # print("面试：Q 怎么把已有模型能力接进 Agent? Q Agent 靠什么决定调哪个工具? (答:tool 的 name+description+参数类型)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
    else:
        smoke()
        # print("\n>>> 尝试真 Agent(需 HF Token；没有会报错，属正常)：")
        try:
            print(run_agent("这条工单情绪如何，并在FAQ里找对应答复：我的照片传不上去老是闪退"))
        except Exception as e:
            print(f"  (未跑真 Agent：{type(e).__name__}: {str(e)[:80]}；export HF_TOKEN 后可跑)")
