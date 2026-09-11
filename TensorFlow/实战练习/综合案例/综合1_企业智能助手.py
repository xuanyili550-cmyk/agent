"""
================================================================================
 综合案例1 · 企业智能助手（整合 RAG + 中文NLP + 计算 + 本地LLM生成）
================================================================================
 一个企业助手：用户问什么，助手判断意图 → 调对应能力 → 用本地 LLM 组织成自然语言回答。
 综合了前面几乎所有技术：
   [Ch5 语义检索]  查知识库(bge-small-zh 中文向量检索) —— 回答“公司政策/产品”类问题
   [Ch1 情感]      分析用户文本情绪(uer/dianping)
   [Ch6/7 NER]     抽取文本里的实体(uer/cluener)
   [工具]          计算器(算预算/数字)
   [本地LLM]       mlx-lm(Qwen2.5-0.5B) 把工具结果组织成回答(换 vLLM/云端只改一处)
 编排用【确定性意图路由】(稳)；生产可换 LLM function calling(见 ../Agent实战)。
 跑：python3 综合1_企业智能助手.py
================================================================================
"""
import re

_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
KB = [
    "年假政策：入职满1年 5 天，满3年 10 天，满5年 15 天。",
    "报销流程：登录 OA → 填报销单 → 主管审批 → 财务打款，3 个工作日到账。",
    "远程办公：每周最多 2 天居家，需提前一天在系统申请。",
    "专业版产品每月 12 美元，含 1TB 存储和优先支持。",
]


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---- 能力① 知识库检索(bge) ----
def _embed(texts, is_query=False):
    import torch
    import torch.nn.functional as F
    if "emb" not in _M:
        from transformers import AutoTokenizer, AutoModel
        n = "BAAI/bge-small-zh-v1.5"
        _M["et"] = AutoTokenizer.from_pretrained(n)
        _M["em"] = AutoModel.from_pretrained(n).to(_dev()).eval()
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    enc = _M["et"](texts, padding=True, truncation=True, return_tensors="pt").to(_dev())
    with torch.no_grad():
        v = _M["em"](**enc).last_hidden_state[:, 0]
    return F.normalize(v, p=2, dim=1)


def search_kb(q):
    if "kv" not in _M:
        _M["kv"] = _embed(KB)
    sims = (_embed([q], is_query=True) @ _M["kv"].T)[0]
    return KB[int(sims.argmax())], float(sims.max())


# ---- 能力② 情感 + NER ----
def analyze(text):
    import torch
    if "sent" not in _M:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        s = "uer/roberta-base-finetuned-dianping-chinese"
        _M["st"] = AutoTokenizer.from_pretrained(s)
        _M["sm"] = AutoModelForSequenceClassification.from_pretrained(s).to(_dev()).eval()
        _M["ner"] = pipeline("token-classification",
                            model="uer/roberta-base-finetuned-cluener2020-chinese",
                            aggregation_strategy="simple", device=-1)
    enc = _M["st"](text[:512], return_tensors="pt", truncation=True).to(_dev())
    with torch.no_grad():
        p = torch.softmax(_M["sm"](**enc).logits, -1)[0]
    sent = f"{'正面' if int(p.argmax()) == 1 else '负面'}({float(p.max()):.2f})"
    ents = [(e["entity_group"], e["word"].replace(" ", "")) for e in _M["ner"](text)]
    return f"情感={sent}；实体={ents or '无'}"


# ---- 能力③ 计算 ----
def calc(q):
    q = (q.replace("乘以", "*").replace("乘", "*").replace("加", "+")
         .replace("减", "-").replace("除以", "/").replace("除", "/"))
    # 只抽“含运算符的算式”，别把句子里的散落数字(如 3个部门/8人)也拼进去
    m = re.search(r"[\d.]+(?:\s*[-+*/]\s*[\d.()]+)+", q)
    if not m:
        return "无算式"
    try:
        return str(eval(re.sub(r"\s", "", m.group()), {"__builtins__": {}}, {}))
    except Exception:
        return "计算失败"


def _llm(prompt, n=100):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    t = _M["t"].apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return generate(_M["m"], _M["t"], prompt=t, max_tokens=n, verbose=False).strip()


# ---- 意图路由(确定性) + 综合回答 ----
def route(q):
    if re.search(r"\d.*[+\-*/].*\d|算|乘|加|减|除以|预算|多少钱.*[*+]", q):
        return "calc"
    if re.search(r"情感|情绪|实体|人名|公司|分析这句", q):
        return "analyze"
    return "search"


def assistant(question):
    intent = route(question)
    if intent == "calc":
        obs = "计算结果 = " + calc(question)
    elif intent == "analyze":
        obs = analyze(question)
    else:
        doc, score = search_kb(question)
        obs = f"知识库命中(相似度{score:.2f})：{doc}"
    answer = _llm(f"用户问：{question}\n参考信息：{obs}\n请用一句话自然地回答：", n=80)
    return {"问题": question, "意图": intent, "参考": obs, "回答": answer}


if __name__ == "__main__":
    for q in ["入职满3年有几天年假？", "帮我算一下 3 个部门各 8 人一共多少人：3*8",
              "分析这句话情感和实体：我对腾讯的服务很不满意"]:
        r = assistant(q)
        # print("─" * 64)  # 装饰分隔线（静音）
        print(f"👤 {r['问题']}\n  🎯 意图={r['意图']}  📎 {r['参考']}\n  🤖 {r['回答'][:90]}")
    assert route("算 3*8") == "calc"
    assert route("入职年假政策") == "search"
    assert "10" in search_kb("满3年年假")[0]
    assert "腾讯" in analyze("我在腾讯上班")
    print("\n✅ 综合1 跑通：意图路由 → RAG检索/情感NER/计算 → mlx-lm 综合回答（整合 Ch1/5/6/7+LLM）。")
