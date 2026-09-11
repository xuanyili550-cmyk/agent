"""
================================================================================
 综合项目2 · RAG 企业知识库问答（中文版，整合 Ch5 + Ch6 + Ch7 + Ch9）
================================================================================
 完整 RAG 检索流程：长文档 → 切块 → 嵌入 → 向量检索 top-k → (重排) → 拼提示 → LLM 生成带引用回答。
 全部用【中文模型/中文文档】：
   [Ch6 切块]     中文按标点(。！？)切句 + 按字数聚块 + 重叠(中文没有空格，不能按词数切)。
   [Ch5 嵌入检索] BAAI/bge-small-zh-v1.5：CLS 池化 + 查询加前缀，按余弦找最相关(换说法也命中)。
   [Ch7 QA]       检索到的块拼进提示，让 LLM 基于它回答并标注引用；这里用抽取式返回最相关块。
   [Ch9 上线]     包成 Gradio/FastAPI 问答界面。
   [为什么 RAG]   LLM 不知道你的私有/最新知识、会幻觉；RAG 先检索再回答，准确、可溯源、免重训。

 本地跑：python3 项目2_RAG企业知识库.py
        python3 项目2_RAG企业知识库.py ui       # Gradio 问答界面
 生产版(向量数据库 Qdrant/Milvus)见 ../生产架构/生产02。
================================================================================
"""
import re
import sys
import torch
import torch.nn.functional as F

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："      # bge 中文检索：查询加前缀


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


# 一篇“公司文档”(真实项目里是很多 PDF/wiki，这里用一段中文长文本演示切块)
DOCUMENT = """
我们的 SaaS 平台提供三种套餐。免费版包含 5GB 存储和基础支持。
专业版每月 12 美元，包含 1TB 存储、优先邮件支持和团队协作共享功能。
企业版提供无限存储、专属客户经理，以及 SSO 单点登录集成。
重置密码：请进入 设置 页面，选择 安全，再点击 重置密码，然后按邮件中的链接操作。
移动端 App 曾有一个上传照片导致闪退的缺陷，该问题已在 3.2 版本中修复。
重复扣款的退款会在核实后的 3 到 5 个工作日内完成。
双重身份验证可在 设置 中的 安全 里开启 2FA，配合身份验证器 App 使用。
数据在存储时使用 AES-256 加密，在传输时使用 TLS 1.3 加密，我们已通过 SOC 2 Type II 合规认证。
""".strip()


def chunk_text(text, max_chars=50, overlap=True):
    """[Ch6] 中文按句切块 + 重叠。中文没有空格，按【标点切句 + 字数聚块】，别按词数。
    真实项目按 token 数切(如 300token)、用 langchain/自研切块器。"""
    sents = [s for s in re.split(r"(?<=[。！？])", text.replace("\n", "")) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(c) for c in cur) >= max_chars:       # 按字符数聚块
            chunks.append("".join(cur))
            cur = cur[-1:] if overlap else []           # 保留最后一句做重叠(别让信息被切断)
    if cur:
        chunks.append("".join(cur))
    return chunks


class RAG:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        self.dev = pick_device()
        e = "BAAI/bge-small-zh-v1.5"
        self.tok = AutoTokenizer.from_pretrained(e)
        self.emb = AutoModel.from_pretrained(e).to(self.dev).eval()
        self.chunks = chunk_text(DOCUMENT)                 # 离线：切块
        self.vecs = self.embed(self.chunks)                # 离线：嵌入建“索引”(内存版)

    def embed(self, texts, is_query=False):
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        enc = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            v = self.emb(**enc).last_hidden_state[:, 0]    # [Ch5] bge 用 CLS 池化
        return F.normalize(v, p=2, dim=1)                  # 归一化 → 点积=余弦

    def retrieve(self, query, top_k=2):
        sims = (self.embed([query], is_query=True) @ self.vecs.T)[0]   # [Ch5] 余弦相似度
        idx = sims.argsort(descending=True)[:top_k]
        return [(float(sims[i]), self.chunks[i]) for i in idx]

    def answer(self, query, top_k=2):
        hits = self.retrieve(query, top_k)
        # [Ch7] 真实项目：把 hits 拼进提示喂 LLM 生成回答。这里用“抽取式”返回最相关块 + 引用。
        context = "\n".join(f"[{i+1}] {c}" for i, (_, c) in enumerate(hits))
        return {"query": query, "sources": hits, "context": context,
                "answer": f"根据资料[1]：{hits[0][1]}"}


def run_cli():
    rag = RAG()
    print(f">>> 设备={rag.dev}  文档切成 {len(rag.chunks)} 块（全中文模型）\n")
    for q in ["专业版套餐多少钱？都包含哪些功能？",
              "我忘记密码了，该怎么办？",
              "你们的数据安全吗？"]:
        r = rag.answer(q)
        # print("─" * 62)
        print(f"问: {q}")
        print(f"  答: {r['answer']}")
        print(f"  引用来源(相似度): {[round(s,2) for s,_ in r['sources']]}")
    # print("─" * 62)
    # 自检：安全性问题应命中含 加密/AES 的块
    assert any("加密" in c or "AES" in c for _, c in rag.retrieve("你们的数据安全吗？"))
    print("✅ 中文 RAG 跑通：切块(Ch6)→嵌入检索(Ch5)→带引用回答(Ch7)。")


def run_ui():
    import gradio as gr
    rag = RAG()
    def fn(q):
        r = rag.answer(q)
        src = "\n".join(f"[{i+1}] ({s:.2f}) {c}" for i, (s, c) in enumerate(r["sources"]))
        return r["answer"] + "\n\n引用来源:\n" + src
    gr.Interface(fn=fn, inputs=gr.Textbox(label="问题", lines=2),
                 outputs=gr.Textbox(label="回答 + 引用", lines=8),
                 title="企业知识库问答(RAG·中文)",
                 examples=[["专业版多少钱？"], ["怎么开启双重验证？"]]).launch()


# ==============================================================================
# 生产要点
# ==============================================================================
# · 向量库：上规模用 Qdrant/Milvus(见 ../生产架构/生产02)，支持增量更新/元数据过滤/持久化。
# · 切块：中文按句/标点+字数(300-500字)+重叠；表格/代码要特殊处理。切块质量直接决定检索质量。
# · 嵌入模型：中文选 bge-large-zh/bge-m3/text2vec；嵌入服务可单独部署批量推理。
# · 混合检索：向量(语义)+BM25(关键词，中文要先分词)用 RRF 融合，召回更全。
# · 重排(rerank)：先召回 top-50，再用 cross-encoder(如 bge-reranker-base)重排取 top-5，精度更高。
# · 生成：把 top-k 块拼进提示喂中文 LLM(Qwen/GLM，见 生产01)，要求“只根据资料回答并标引用”防幻觉。
# · 权限：按租户/权限过滤向量(元数据)，防跨权限泄露。

# ==============================================================================
# 面试题
# ==============================================================================
# Q: RAG 为什么比直接微调塞知识好？
# A: 知识会变，微调追不上且贵/易过时；RAG 改知识库即可、能引用来源、免重训。常“微调管怎么答+RAG管答什么”。
# Q: 中文切块和英文有什么不同？
# A: 中文没有空格不能按词数切，要按标点(。！？)切句再按字数聚块；也可按 token 数切。
# Q: 召回不准怎么排查？
# A: 看切块质量→嵌入模型是否匹配领域/语言(中文要用中文嵌入)→是否需要混合检索/重排→query 改写。
# Q: 怎么防止 RAG 幻觉？
# A: 提示里强制“只根据给定资料回答、无据就说不知道、标注引用编号”；答案可做“是否有据”校验。

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        run_ui()
    else:
        run_cli()
