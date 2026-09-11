"""
================================================================================
 综合案例3 · 文档智能处理流水线（整合 Ch6切块 + Ch7实体 + Ch1情感 + Ch5检索 + ChromaDB + LLM）
================================================================================
 一篇中文文档进来，流水线自动：切块 → 每块抽实体+判情感 → 抽取式摘要 → 向量入库(ChromaDB) →
 支持基于该文档的问答(检索 + 本地 LLM 生成)。把“文档理解 + RAG”一条龙。
 整合：Ch6 切块、Ch7 NER、Ch1 情感、Ch5 bge 检索、ChromaDB 向量库、mlx-lm 生成。
 跑：python3 综合3_文档智能处理.py
================================================================================
"""
import re

_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
DOC = """
本公司与腾讯、阿里巴巴在2024年达成战略合作，由张伟担任项目负责人，办公地点位于北京中关村。
第一季度营收增长显著，团队士气高涨，客户反馈非常满意。
但第二季度因供应链问题出现延误，部分客户表达了强烈不满，投诉量上升。
公司已成立专项小组，由李娜牵头，承诺在三个月内解决所有遗留问题。
""".strip()


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


def sentiment(text):
    import torch
    enc = _M["st"](text[:512], return_tensors="pt", truncation=True).to(_dev())
    with torch.no_grad():
        p = torch.softmax(_M["sm"](**enc).logits, -1)[0]
    return "正面" if int(p.argmax()) == 1 else "负面"


def process(doc):
    _load()
    chunks = [s.strip() for s in re.split(r"(?<=[。！？])", doc.replace("\n", "")) if s.strip()]
    analyzed = []
    for c in chunks:
        ents = [(e["entity_group"], e["word"].replace(" ", "")) for e in _M["ner"](c)]
        analyzed.append({"句": c, "情感": sentiment(c), "实体": ents})
    # 抽取式摘要(规则)：取实体最多的两句当摘要
    summary = "；".join(x["句"] for x in sorted(analyzed, key=lambda a: -len(a["实体"]))[:2])
    # 向量入库(ChromaDB)
    import chromadb
    _M["client"] = _M.get("client") or chromadb.Client()
    try:
        _M["client"].delete_collection("doc_kb")
    except Exception:
        pass
    col = _M["client"].create_collection("doc_kb")
    col.add(ids=[str(i) for i in range(len(chunks))], documents=chunks,
            embeddings=_embed(chunks).cpu().tolist())
    _M["col"] = col
    return analyzed, summary


def ask(question):
    res = _M["col"].query(query_embeddings=_embed([question], is_query=True).cpu().tolist(), n_results=1)
    hit = res["documents"][0][0]
    try:
        from mlx_lm import load, generate
        if "m" not in _M:
            _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        t = _M["t"].apply_chat_template(
            [{"role": "user", "content": f"根据资料回答。\n资料：{hit}\n问题：{question}\n回答："}],
            add_generation_prompt=True)
        ans = generate(_M["m"], _M["t"], prompt=t, max_tokens=60, verbose=False).strip()
    except Exception as e:
        ans = f"（降级抽取式）{hit}  [{type(e).__name__}]"
    return ans, hit


if __name__ == "__main__":
    analyzed, summary = process(DOC)
    # print(">>> 文档逐句分析(情感 + 实体)：")
    for a in analyzed:
        print(f"  [{a['情感']}] {a['句'][:30]}...  实体={a['实体']}")
    print(f"\n>>> 抽取式摘要(实体最密的句子)：\n  {summary}")
    # print("\n>>> 基于文档问答(检索 + mlx-lm)：")
    for q in ["项目负责人是谁？", "第二季度出了什么问题？"]:
        ans, hit = ask(q)
        print(f"  ❓{q}\n   📎命中：{hit[:36]}...\n   💬{ans[:70]}")
    # 自检
    assert any("张伟" in str(a["实体"]) for a in analyzed)
    assert any(a["情感"] == "负面" for a in analyzed)
    assert "供应链" in ask("第二季度问题")[1]
    print("\n✅ 综合3 跑通：切块→实体+情感→抽取摘要→ChromaDB入库→检索问答（整合 Ch1/5/6/7+库+LLM）。")
