"""
================================================================================
 Agent Course · Chapter 3 · 案例：Gala 特工 Alfred（嘉宾信息 Agentic RAG，可运行）
================================================================================
 场景(呼应课程 Gala Agent)：韦恩豪宅办派对，管家 Alfred 要随时回答“某某嘉宾是谁/什么背景/
 怎么联系”。做法：把嘉宾资料建成可检索知识库(bge 语义检索)，Alfred 收到问题 → 检索嘉宾 →
 基于检索到的资料回答(可溯源)。这就是 Agentic RAG 在“查资料型助理”上的落地。
 (框架版：把 search_guest 用 smolagents Tool 包起来 + CodeAgent 即可，见学习笔记。)
 跑：python3 chapter3_AgenticRAG案例.py
================================================================================
"""
_M = {}
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 嘉宾知识库(name / relation / description / email)
GUESTS = [
    "Ada Lovelace | 老友 | 数学家、程序设计先驱，被誉为第一位程序员，研究分析机。 | ada@lovelace.org",
    "Nikola Tesla | 合作者 | 发明家、电气工程师，无线能量与交流电专家。 | tesla@wardenclyffe.org",
    "Bruce Wayne | 主人 | 韦恩企业 CEO，本次派对的主人。 | bruce@wayne.enterprises",
    "Alan Turing | 老友 | 计算机科学与人工智能之父，破译 Enigma。 | alan@turing.uk",
]


def _dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


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


def search_guest(query):
    """检索工具：按名字/关系/描述找最相关的嘉宾资料。"""
    if "gv" not in _M:
        _M["gv"] = _embed(GUESTS)
    sims = (_embed([query], is_query=True) @ _M["gv"].T)[0]
    return GUESTS[int(sims.argmax())], float(sims.max())


def llm(p, n=100):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    t = _M["t"].apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
    return generate(_M["m"], _M["t"], prompt=t, max_tokens=n, verbose=False).strip()


def alfred_answer(question):
    """Alfred：检索嘉宾资料 → 基于资料回答(可溯源)。"""
    guest, score = search_guest(question)
    ans = llm(f"你是管家 Alfred。根据下面嘉宾资料回答主人的问题。\n资料：{guest}\n问题：{question}\n回答：")
    return {"问题": question, "命中嘉宾": guest.split("|")[0].strip(), "相似度": round(score, 2), "回答": ans}


if __name__ == "__main__":
    for q in ["跟我说说 Ada Lovelace 是谁", "谁研究无线能量？我想和他聊聊",
              "破译 Enigma 的那位嘉宾背景如何"]:
        r = alfred_answer(q)
        print("─" * 62)
        print(f"👤 主人：{r['问题']}\n  🔍 命中嘉宾：{r['命中嘉宾']}（相似度{r['相似度']}）\n  🎩 Alfred：{r['回答'][:90]}")
    assert search_guest("第一位程序员是谁")[0].startswith("Ada")
    assert search_guest("无线能量专家")[0].startswith("Nikola")
    print("\n✅ Chapter3 案例跑通：Gala 特工 Alfred 用语义检索嘉宾资料 + mlx-lm 生成回答(可溯源)。")
