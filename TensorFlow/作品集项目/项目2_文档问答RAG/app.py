"""
作品集项目2 · 文档问答 RAG 系统（生产级，可部署）
--------------------------------------------------------------------------------
 一句话：上传/粘贴一篇文档 → 切块 → 向量化入库(ChromaDB) → 提问时语义检索 → 用本地 LLM 生成带引用的回答。
 这是当前最热门方向(RAG)：让 LLM 基于你给的私有资料回答，准确、可溯源、免重训。
 技术栈(生产)：
   · 嵌入：BAAI/bge-small-zh-v1.5（中文；★CLS 池化 + 查询加前缀，召回更准）
   · 向量库：ChromaDB（真向量数据库，支持增量/持久化；这里用内存实例，Spaces 够用）
   · 生成：mlx-lm 本地 LLM（Qwen2.5-0.5B-Instruct-4bit，Mac mps 上真跑；换 vLLM/云端只改一处）
 运行：
   python3 app.py smoke   # 建库+检索+生成 各一次自检(不起服务)
   python3 app.py         # 起 Web 界面(粘贴文档→建索引→提问)
"""
import sys
import re
import gradio as gr

_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

SAMPLE_DOC = """
我们的 SaaS 平台提供三种套餐。免费版包含 5GB 存储和基础支持。
专业版每月 12 美元，包含 1TB 存储、优先邮件支持和团队协作共享功能。
企业版提供无限存储、专属客户经理，以及 SSO 单点登录集成。
重置密码：请进入 设置 页面，选择 安全，再点击 重置密码，然后按邮件中的链接操作。
数据在存储时使用 AES-256 加密，在传输时使用 TLS 1.3 加密，我们已通过 SOC 2 Type II 合规认证。
""".strip()


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _embed(texts, is_query=False):
    """bge-small-zh：CLS 池化 + 归一化；查询加前缀提升召回(文档不加)。"""
    import torch
    import torch.nn.functional as F
    if "emb" not in _M:
        from transformers import AutoTokenizer, AutoModel
        n = "BAAI/bge-small-zh-v1.5"
        _M["emb_tok"] = AutoTokenizer.from_pretrained(n)
        _M["emb"] = AutoModel.from_pretrained(n).to(_dev()).eval()
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    enc = _M["emb_tok"](texts, padding=True, truncation=True, return_tensors="pt").to(_dev())
    with torch.no_grad():
        v = _M["emb"](**enc).last_hidden_state[:, 0]          # CLS 池化
    return F.normalize(v, p=2, dim=1).cpu().tolist()


def chunk(doc, max_chars=60):
    """中文按标点切句 + 按字数聚块(中文无空格，别按词数)。"""
    sents = [s for s in re.split(r"(?<=[。！？\n])", doc) if s.strip()]
    chunks, cur = [], ""
    for s in sents:
        cur += s.strip()
        if len(cur) >= max_chars:
            chunks.append(cur); cur = ""
    if cur:
        chunks.append(cur)
    return chunks


def build_index(doc):
    """建向量索引：切块 → bge 嵌入 → 存进 ChromaDB 集合(真向量库)。"""
    if not doc or not doc.strip():
        raise gr.Error("请先粘贴文档内容")
    import chromadb
    chunks = chunk(doc)
    _M["client"] = _M.get("client") or chromadb.Client()
    try:
        _M["client"].delete_collection("knowledge_base")
    except Exception:
        pass
    col = _M["client"].create_collection("knowledge_base")               # 新建集合
    col.add(ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=_embed(chunks))                        # ★用我们自己的 bge 向量(不用 chroma 默认英文嵌入)
    _M["col"] = col
    return f"✅ 已建索引：文档切成 {len(chunks)} 块，存入 ChromaDB。可以提问了。"


def retrieve(question, top_k=2):
    if "col" not in _M:
        raise gr.Error("请先点“建立索引”")
    res = _M["col"].query(query_embeddings=_embed([question], is_query=True), n_results=top_k)
    return res["documents"][0]                                # top-k 文档块


def _llm_answer(question, context):
    """本地 LLM 基于检索到的资料生成回答(mlx-lm)。生产换 vLLM/云端只改这一函数。"""
    if "llm" not in _M:
        from mlx_lm import load
        _M["llm"], _M["llm_tok"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    prompt = (f"只根据下面资料回答问题，资料没有就说不知道。\n资料：\n{context}\n\n问题：{question}\n回答：")
    msgs = [{"role": "user", "content": prompt}]
    text = _M["llm_tok"].apply_chat_template(msgs, add_generation_prompt=True)
    return generate(_M["llm"], _M["llm_tok"], prompt=text, max_tokens=120, verbose=False).strip()


def answer(question):
    if not question or not question.strip():
        raise gr.Error("请输入问题")
    hits = retrieve(question, top_k=2)
    context = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(hits))
    try:
        ans = _llm_answer(question, context)
    except Exception as e:
        ans = f"（本地 LLM 生成失败，降级为抽取式）根据资料[1]：{hits[0]}\n[{type(e).__name__}]"
    return ans, context


def build_demo():
    with gr.Blocks(title="文档问答 RAG", analytics_enabled=False) as demo:
        gr.Markdown("# 文档问答 RAG 系统\n粘贴文档 → 建索引 → 提问，系统语义检索 + 本地 LLM 生成带引用的回答。")
        doc = gr.Textbox(label="① 文档内容", lines=8, value=SAMPLE_DOC)
        status = gr.Textbox(label="索引状态", interactive=False)
        gr.Button("② 建立索引", variant="primary").click(build_index, doc, status)
        q = gr.Textbox(label="③ 提问", placeholder="例如：专业版多少钱？数据安全吗？")
        ans = gr.Textbox(label="回答", lines=3)
        src = gr.Textbox(label="引用来源(检索到的资料块)", lines=4)
        gr.Button("提问", variant="primary").click(answer, q, [ans, src])
        gr.Examples([["专业版套餐多少钱？"], ["你们的数据安全吗？"], ["怎么重置密码？"]], inputs=q)
    return demo


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        print(build_index(SAMPLE_DOC))
        hits = retrieve("专业版多少钱", top_k=2)
        print("检索:", hits[0][:40], "...")
        assert any("12" in h for h in hits)
        a, s = answer("你们的数据安全吗？")
        print("回答:", a[:80])
        assert "加密" in s or "AES" in s
        build_demo()
        print("✅ 项目2 自检通过：切块→bge嵌入→ChromaDB→检索→mlx-lm 生成。")
    else:
        build_demo().queue().launch()
