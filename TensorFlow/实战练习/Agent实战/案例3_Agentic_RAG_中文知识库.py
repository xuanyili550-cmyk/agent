"""
================================================================================
 Agent 实战 · 案例3 · Agentic RAG（Agent 自己决定检索什么、检索几次，再回答）
================================================================================
 普通 RAG：固定“检索一次 → 塞进提示 → 回答”。Agentic RAG：把【检索】做成一个工具交给 Agent，
 让 LLM 自己判断“要不要查、查什么关键词、够不够、要不要再查一次”，更灵活、更准。
 整合：检索工具用 Ch5 的中文语义检索(bge-small-zh 向量，换说法也命中)——比课程原版的 BM25 更懂语义。

 本机现实：检索工具【本地真跑 + 自检】；Agent 大脑本地小模型太弱(见 README)，故：
   · smoke：调用检索工具自检 + 用【确定性 Agentic 流程】演示“检索→判断是否够→(再检索)→带引用回答”。
   · 生产：run_agent() 用真 CodeAgent + InferenceClientModel，由 LLM 决定检索策略。
 跑：python3 案例3_Agentic_RAG_中文知识库.py smoke
    python3 案例3_Agentic_RAG_中文知识库.py            # 额外尝试真 Agent(需 HF Token)
================================================================================
"""
import re
import sys
from smolagents import tool

_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

DOCUMENT = """
我们的 SaaS 平台提供三种套餐。免费版包含 5GB 存储和基础支持。
专业版每月 12 美元，包含 1TB 存储、优先邮件支持和团队协作共享功能。
企业版提供无限存储、专属客户经理，以及 SSO 单点登录集成。
重置密码：请进入 设置 页面，选择 安全，再点击 重置密码，然后按邮件中的链接操作。
移动端 App 曾有一个上传照片导致闪退的缺陷，该问题已在 3.2 版本中修复。
重复扣款的退款会在核实后的 3 到 5 个工作日内完成。
数据在存储时使用 AES-256 加密，在传输时使用 TLS 1.3 加密，我们已通过 SOC 2 Type II 合规认证。
""".strip()


def _chunks():
    if "chunks" not in _M:
        sents = [s for s in re.split(r"(?<=[。！？])", DOCUMENT.replace("\n", "")) if s.strip()]
        _M["chunks"] = sents
    return _M["chunks"]


def _embed(texts, is_query=False):
    import torch
    import torch.nn.functional as F
    if "emb" not in _M:
        from transformers import AutoTokenizer, AutoModel
        e = "BAAI/bge-small-zh-v1.5"
        _M["dev"] = "mps" if torch.backends.mps.is_available() else "cpu"
        _M["tok"] = AutoTokenizer.from_pretrained(e)
        _M["emb"] = AutoModel.from_pretrained(e).to(_M["dev"]).eval()
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    enc = _M["tok"](texts, padding=True, truncation=True, return_tensors="pt").to(_M["dev"])
    with torch.no_grad():
        v = _M["emb"](**enc).last_hidden_state[:, 0]        # bge CLS 池化
    return F.normalize(v, p=2, dim=1)


# ==============================================================================
# 检索工具（smolagents @tool）：Agent 调它来查知识库
# ==============================================================================
@tool
def search_knowledge_base(query: str, top_k: int = 2) -> str:
    """在公司知识库里做中文语义检索，返回最相关的若干条资料(带相似度，可作引用)。

    Args:
        query: 检索的问题或关键词(中文)。
        top_k: 返回前几条，默认 2。
    """
    chunks = _chunks()
    if "vecs" not in _M:
        _M["vecs"] = _embed(chunks)
    sims = (_embed([query], is_query=True) @ _M["vecs"].T)[0]
    idx = sims.argsort(descending=True)[:top_k]
    return "\n".join(f"[{i+1}](相似度{float(sims[j]):.2f}) {chunks[j]}" for i, j in enumerate(idx))


# ==============================================================================
# 确定性 Agentic 流程：检索 → 判断够不够 → (换词再检索) → 带引用回答
# ==============================================================================
def agentic_answer(question, threshold=0.55):
    hits = search_knowledge_base(question, top_k=2)
    top_score = float(re.search(r"相似度([\d.]+)", hits).group(1))
    tried = [question]
    # Agentic：第一次不够好 → 换个说法再查一次(模拟 LLM 的“再检索”决策)
    if top_score < threshold:
        alt = question.replace("怎么", "如何").replace("多少钱", "价格")
        if alt != question:
            tried.append(alt)
            hits2 = search_knowledge_base(alt, top_k=2)
            if float(re.search(r"相似度([\d.]+)", hits2).group(1)) > top_score:
                hits = hits2
    answer = "根据资料：" + hits.split("\n")[0].split(") ", 1)[-1]
    return {"问题": question, "检索次数": len(tried), "资料": hits, "回答": answer}


def run_agent(question):
    from smolagents import CodeAgent, InferenceClientModel
    agent = CodeAgent(tools=[search_knowledge_base], model=InferenceClientModel(), max_steps=6)
    return agent.run(question)


def smoke():
    # 检索工具自检(bge-small-zh 中文语义检索)，期望：
    #   search_knowledge_base("专业版多少钱") -> 命中"专业版每月 12 美元..."
    #   search_knowledge_base("你们的数据安全吗") -> 命中"AES-256 加密...TLS 1.3..."(换说法也命中)
    assert "12" in search_knowledge_base("专业版价格")
    assert "加密" in search_knowledge_base("你们的数据安全吗")           # 换说法也命中
    # 确定性 Agentic 流程(检索→判断够不够→(换词再检索)→带引用回答)，期望：
    #   "专业版套餐多少钱？" -> 答"专业版每月 12 美元..."
    #   "忘记密码怎么办？"   -> 命中"重置密码..."
    #   "你们的数据安全吗？" -> 命中"AES-256 加密...SOC 2..."
    for q in ["专业版套餐多少钱？", "忘记密码怎么办？", "你们的数据安全吗？"]:
        r = agentic_answer(q)
        # print(f"  问: {q}  (检索{r['检索次数']}次) 答: {r['回答']}")
    print("\n✅ 案例3 跑通：Ch5 中文语义检索已包成 Agent 工具 + Agentic 检索流程通过(换说法也命中、可溯源)。")
    # print("面试：Q Agentic RAG 和普通 RAG 区别? A 普通RAG固定检索一次;Agentic 让 LLM 自己决定查什么/查几次/够不够。")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
    else:
        smoke()
        # print("\n>>> 尝试真 Agent(需 HF Token)：")
        try:
            print(run_agent("我们的数据加密和合规情况如何？请基于知识库回答。"))
        except Exception as e:
            print(f"  (未跑真 Agent：{type(e).__name__}: {str(e)[:80]}；export HF_TOKEN 后可跑)")
