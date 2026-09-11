"""
================================================================================
 综合案例6 · 全能力客服 Agent（集大成：情感 + NER + RAG + 本地LLM生成 + 决策）
================================================================================
 一条工单进来，客服 Agent 一条龙处理：
   ① 判情绪(Ch1 情感) → 定优先级        ② 抽实体(Ch6/7 NER) → 路由/脱敏
   ③ 语义检索 FAQ(Ch5 bge)               ④ 本地 LLM(mlx-lm) 结合 FAQ 生成个性化回复
   ⑤ 决策：能自动答就自动答(负面标优先)，无对口 FAQ / 低置信 → 转人工
 把前面所有 NLP 能力 + RAG + 本地 LLM + 生产分流决策综合到一个 Agent 里。
 跑：python3 综合6_全能力客服Agent.py
================================================================================
"""
_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
CONF_TH, FAQ_TH = 0.80, 0.45
FAQ = [
    ("重置密码", "去 设置 > 安全 > 重置密码，按邮件链接操作。"),
    ("上传照片闪退", "请升级 App 到 v3.2 或更高版本，闪退已修复。"),
    ("重复扣款退款", "重复扣款会在核实后 3-5 个工作日内原路退回。"),
    ("存储空间", "免费 5GB；升级 Pro 得 1TB。"),
]


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    if "ready" not in _M:
        import torch  # noqa
        from transformers import (AutoTokenizer, AutoModel,
                                  AutoModelForSequenceClassification, pipeline)
        s = "uer/roberta-base-finetuned-dianping-chinese"
        _M["st"] = AutoTokenizer.from_pretrained(s)
        _M["sm"] = AutoModelForSequenceClassification.from_pretrained(s).to(_dev()).eval()
        _M["ner"] = pipeline("token-classification",
                            model="uer/roberta-base-finetuned-cluener2020-chinese",
                            aggregation_strategy="simple", device=-1)
        e = "BAAI/bge-small-zh-v1.5"
        _M["et"] = AutoTokenizer.from_pretrained(e)
        _M["em"] = AutoModel.from_pretrained(e).to(_dev()).eval()
        _M["fv"] = _embed([a for _, a in FAQ])
        _M["ready"] = True


def _embed(texts, is_query=False):
    import torch
    import torch.nn.functional as F
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    enc = _M["et"](texts, padding=True, truncation=True, return_tensors="pt").to(_dev())
    with torch.no_grad():
        v = _M["em"](**enc).last_hidden_state[:, 0]
    return F.normalize(v, p=2, dim=1)


def _llm(prompt, n=80):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    t = _M["t"].apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return generate(_M["m"], _M["t"], prompt=t, max_tokens=n, verbose=False).strip()


def handle(text):
    import torch
    _load()
    # ① 情感
    enc = _M["st"](text[:512], return_tensors="pt", truncation=True).to(_dev())
    with torch.no_grad():
        p = torch.softmax(_M["sm"](**enc).logits, -1)[0]
    sent, conf = ("正面" if int(p.argmax()) == 1 else "负面"), float(p.max())
    # ② 实体
    ents = [(e["entity_group"], e["word"].replace(" ", "")) for e in _M["ner"](text)]
    # ③ 检索
    sims = (_embed([text], is_query=True) @ _M["fv"].T)[0]
    i, score = int(sims.argmax()), float(sims.max())
    # ⑤ 决策
    if conf < CONF_TH or score < FAQ_TH:
        return {"工单": text, "情感": f"{sent}({conf:.2f})", "实体": ents or "无",
                "决策": "转人工" + ("(情感置信低)" if conf < CONF_TH else "(无对口FAQ)"), "回复": "—"}
    # ④ 本地 LLM 结合 FAQ 生成个性化回复
    reply = _llm(f"你是客服。用户说：{text}\n参考答案：{FAQ[i][1]}\n用一句友好的话回复用户：")
    prio = "（负面·优先处理）" if sent == "负面" else ""
    return {"工单": text, "情感": f"{sent}({conf:.2f})", "实体": ents or "无",
            "命中FAQ": round(score, 2), "决策": "自动回复" + prio, "回复": reply}


if __name__ == "__main__":
    tickets = ["你们的App一上传照片就崩溃，太气人了！",
               "我被重复扣款了，要退款，我叫张伟在腾讯上班",
               "随便问问"]
    for t in tickets:
        r = handle(t)
        # print("─" * 64)  # 装饰分隔线（静音）
        print(f"工单: {r['工单']}")
        print(f"  情感={r['情感']}  实体={r['实体']}  决策={r['决策']}")
        print(f"  回复: {r['回复'][:80]}")
    assert handle("我要退款")["情感"].startswith("负面")
    print("\n✅ 综合6 跑通(集大成)：情感分流(Ch1)+实体路由(Ch6/7)+RAG检索(Ch5)+mlx生成回复+决策。")
