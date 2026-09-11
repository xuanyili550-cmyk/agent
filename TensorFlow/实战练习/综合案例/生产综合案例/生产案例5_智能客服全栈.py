"""
================================================================================
 生产综合案例5 · 智能客服全栈（真实生产：情感+实体+RAG+Agent决策+LLM回复 + Web）
================================================================================
 一个可上线的客服系统：工单进来 → 情感分流 → 实体路由/脱敏 → RAG 检索 FAQ → Agent 决策 →
 本地 LLM 生成个性化回复。前面几乎所有 NLP 能力 + RAG + Agent 编排 + 生产分流的集大成客服产品。
 覆盖章节功能：
   [Ch1 情感]  uer/dianping 判情绪 → 定优先级
   [Ch7 NER]   uer/cluener 抽实体 → 路由/脱敏(PII 打码)
   [Ch5 检索]  bge-small-zh 语义检索 FAQ(CLS 池化+查询前缀)
   [Agent]     确定性决策：能自动答就答(负面标优先)，低置信/无对口 FAQ → 转人工
   [LLM]       mlx-lm 结合 FAQ 生成自然回复
   [Ch9 Gradio] 客服后台 Web + 结构化结果 + 队列
 生产要点：三模型惰性单例、PII 脱敏、阈值分流、可部署。
 运行：
   python3 生产案例5_智能客服全栈.py smoke   # 跑几条工单自检
   python3 生产案例5_智能客服全栈.py         # 起客服后台 Web
================================================================================
"""
import sys
import gradio as gr

_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
CONF_TH, FAQ_TH = 0.75, 0.45
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
    if not text or not text.strip():
        raise gr.Error("请输入客服工单")
    _load()
    enc = _M["st"](text[:512], return_tensors="pt", truncation=True).to(_dev())          # [Ch1]
    with torch.no_grad():
        p = torch.softmax(_M["sm"](**enc).logits, -1)[0]
    sent, conf = ("正面" if int(p.argmax()) == 1 else "负面"), float(p.max())
    ents = [(e["entity_group"], e["word"].replace(" ", "")) for e in _M["ner"](text)]     # [Ch7]
    masked = text
    for g, w in ents:                                                                     # PII 脱敏
        if g in ("name", "company", "address", "position") and w:
            masked = masked.replace(w, f"[{g}]")
    sims = (_embed([text], is_query=True) @ _M["fv"].T)[0]                                 # [Ch5]
    i, score = int(sims.argmax()), float(sims.max())
    if conf < CONF_TH or score < FAQ_TH:                                                   # [Agent 决策]
        return {"工单": text, "情感": f"{sent}({conf:.2f})", "实体": ents or "无", "脱敏": masked,
                "决策": "转人工" + ("(情感置信低)" if conf < CONF_TH else "(无对口FAQ)"), "回复": "—"}
    reply = _llm(f"你是客服。用户说：{text}\n参考答案：{FAQ[i][1]}\n用一句友好的话回复：")       # [LLM]
    return {"工单": text, "情感": f"{sent}({conf:.2f})", "实体": ents or "无", "脱敏": masked,
            "命中FAQ": round(score, 2),
            "决策": "自动回复" + ("(负面·优先)" if sent == "负面" else ""), "回复": reply}


def build_demo():
    with gr.Blocks(title="智能客服全栈", analytics_enabled=False) as demo:
        gr.Markdown("# 智能客服全栈\n工单 → 情感分流(Ch1) + 实体路由脱敏(Ch7) + RAG检索(Ch5) + Agent决策 + LLM回复")
        inp = gr.Textbox(label="客服工单", lines=3, placeholder="例如：你们App一上传照片就崩溃，太气人了！")
        out = gr.JSON(label="处理结果")
        gr.Button("处理工单", variant="primary").click(handle, inp, out)
        gr.Examples([["你们的App一上传照片就崩溃，太气人了！"],
                     ["我被重复扣款了要退款，我叫张伟在腾讯上班"],
                     ["随便问问"]], inputs=inp)
    return demo


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        for t in ["你们的App一上传照片就崩溃，太气人了！", "我被重复扣款了要退款，我叫张伟在腾讯上班"]:
            r = handle(t)
            # print("─" * 60)
            print(f"工单: {t}\n  情感={r['情感']} 实体={r['实体']} 决策={r['决策']}\n  脱敏: {r['脱敏']}\n  回复: {r['回复'][:60]}")
        assert "张伟" not in handle("我叫张伟在腾讯上班要退款")["脱敏"]      # 已脱敏
        build_demo()
        print("\n✅ 生产案例5 自检通过：情感(Ch1)+NER脱敏(Ch7)+RAG(Ch5)+Agent决策+mlx回复 + Web。")
    else:
        build_demo().queue().launch()
