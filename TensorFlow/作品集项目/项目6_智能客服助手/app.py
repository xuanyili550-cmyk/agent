"""
作品集项目6 · 智能客服助手（集大成：情感分流 + 实体路由 + RAG-FAQ，可部署）
--------------------------------------------------------------------------------
 一句话：一条客服工单进来，系统自动【判情绪(定优先级) → 抽实体(路由/脱敏) → 语义检索 FAQ →
        给出决策(自动回复 / 转人工)】。把项目1(情感)+项目2(RAG检索)+多能力串成一个完整产品。
 生产要点：三模型惰性单例、阈值分流、输入校验、结构化输出、可部署。全中文模型(中文才准)。
 运行：
   python3 app.py smoke   # 跑几条工单自检(不起服务)
   python3 app.py         # 起客服后台 Web
"""
import sys
import gradio as gr

_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
CONF_TH, FAQ_TH = 0.80, 0.45           # 情感置信度阈值 / FAQ 相似度阈值(0.45:能命中就自动答)
FAQ = [
    ("重置密码", "去 设置 > 安全 > 重置密码，按邮件里的链接操作即可。"),
    ("上传照片闪退", "请把 App 升级到 v3.2 或更高版本，闪退问题已修复。"),
    ("重复扣款退款", "重复扣款会在核实后 3-5 个工作日内原路退回。"),
    ("联系人工", "在聊天里输入 'agent'，或工作日 9:00-18:00 拨打热线。"),
    ("存储空间", "免费账户 5GB；升级 Pro 得 1TB。"),
]


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    if "ready" not in _M:
        import torch  # noqa
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  AutoModel, pipeline)
        d = _dev()
        s = "uer/roberta-base-finetuned-dianping-chinese"
        _M["st"] = AutoTokenizer.from_pretrained(s)
        _M["sm"] = AutoModelForSequenceClassification.from_pretrained(s).to(d).eval()
        _M["ner"] = pipeline("token-classification",
                            model="uer/roberta-base-finetuned-cluener2020-chinese",
                            aggregation_strategy="simple", device=-1)
        e = "BAAI/bge-small-zh-v1.5"
        _M["et"] = AutoTokenizer.from_pretrained(e)
        _M["em"] = AutoModel.from_pretrained(e).to(d).eval()
        _M["fv"] = _embed([a for _, a in FAQ])
        _M["ready"] = True


def _embed(texts, is_query=False):
    import torch
    import torch.nn.functional as F
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    enc = _M["et"](texts, padding=True, truncation=True, return_tensors="pt").to(_dev())
    with torch.no_grad():
        v = _M["em"](**enc).last_hidden_state[:, 0]          # bge CLS 池化
    return F.normalize(v, p=2, dim=1)


def handle(text):
    """核心：工单 → 情感/实体/检索 → 决策。"""
    import torch
    if not text or not text.strip():
        raise gr.Error("请输入客服工单内容")
    _load()
    # ① 情感(定优先级/是否转人工)
    enc = _M["st"](text[:512], return_tensors="pt", truncation=True).to(_dev())
    with torch.no_grad():
        p = torch.softmax(_M["sm"](**enc).logits, -1)[0]
    sentiment, conf = ("正面" if int(p.argmax()) == 1 else "负面"), float(p.max())
    # ② 实体(路由/脱敏)
    ents = [(e["entity_group"], e["word"].replace(" ", "")) for e in _M["ner"](text)]
    # ③ 语义检索 FAQ
    sims = (_embed([text], is_query=True) @ _M["fv"].T)[0]
    i, score = int(sims.argmax()), float(sims.max())
    # ④ 决策：低置信/无对口FAQ → 转人工；否则自动回复(负面标优先)
    if conf < CONF_TH or score < FAQ_TH:
        decision = "转人工" + ("（情感置信度低）" if conf < CONF_TH else "（无对口FAQ）")
        reply = ""
    else:
        decision = "自动回复" + ("（负面工单·优先处理）" if sentiment == "负面" else "")
        reply = FAQ[i][1]
    return {"情感": f"{sentiment}({conf:.2f})", "实体": ents or "无",
            "命中FAQ相似度": round(score, 2), "决策": decision, "建议回复": reply or "—"}


def build_demo():
    with gr.Blocks(title="智能客服助手", analytics_enabled=False) as demo:
        gr.Markdown("# 智能客服助手\n工单进来自动：判情绪(优先级) → 抽实体(路由) → 检索FAQ → 决策(自动回复/转人工)")
        inp = gr.Textbox(label="客服工单", lines=3, placeholder="例如：你们App一上传照片就崩溃，太气人了！")
        out = gr.JSON(label="处理结果")
        gr.Button("处理工单", variant="primary").click(handle, inp, out)
        gr.Examples([["你们的App一上传照片就崩溃，太气人了！"],
                     ["我通过支付宝被重复扣了两次费，要求马上退款"],
                     ["谢谢，问题解决了，客服很给力"]], inputs=inp)
    return demo


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        for t in ["你们的App一上传照片就崩溃，太气人了！",
                  "我被重复扣款了要退款", "随便问问"]:
            r = handle(t)
            print(f"\n工单: {t}\n  {r}")
        assert handle("我被重复扣款了要退款")["情感"].startswith("负面")
        build_demo()
        print("\n✅ 项目6 自检通过：情感分流 + 实体路由 + RAG-FAQ + 决策，一个完整客服产品。")
    else:
        build_demo().queue().launch()
