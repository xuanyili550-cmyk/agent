EMBED_CKPT = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
import sys
import torch

CORPUS = [
    "You can load a dataset offline by using a local cache without internet.",
    "To fine-tune a model, use the Trainer API together with TrainingArguments.",
    "FAISS builds an index that lets you quickly find similar embeddings.",
    "Tokenizers convert text into input_ids that the model can process.",
    "Padding makes all sequences in a batch the same length.",
    "The attention mask tells the model which tokens to ignore.",
    "Semantic search finds documents by meaning, not by exact keywords.",
    "Streaming lets you use a dataset that is larger than your memory.",
    "Mean pooling averages token vectors to produce a sentence embedding.",
    "You can deploy a model as an OpenAI-compatible API server.",
]

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def title(n, text):
    print("\n" + "=" * 72)
    print(f"  第 {n} 关：{text}")
    print("=" * 72)


def load_embedder():
    from transformers import AutoTokenizer,AutoModel
    tok=AutoTokenizer.from_pretrained(EMBED_CKPT)
    model=AutoModel.from_pretrained(EMBED_CKPT).to(pick_device())
    model.eval()
    def get_embeddings(text_list):
        enc=tok(text_list,padding=True,truncation=True, return_tensors="pt")
        enc={k:v.to(pick_device()) for k ,v in enc.items()}
        with torch.no_grad():
            out=model(**enc)
        return out.last_hidden_state[:,0]
    return tok,model,get_embeddings

def case_01():
    title(1, "数据集处理三招：from_dict + filter + map + rename")
    from datasets import Dataset
    ds = Dataset.from_dict({
        "review": ["great drug, helped a lot", "no effect at all", "", "side effects were bad"],
        "rating": [9, 3, 5, 2],
    })
    print("原始：", ds.num_rows, "行")
    ds=ds.filter(lambda x:len(x["review"])>0)
    print("filter 去空后：", ds.num_rows, "行")
    ds=ds.map(lambda x:{"n_words":len(x["review"].split())})
    print("map 加词数列：", ds.column_names)
    ds=ds.rename_column("rating","score")
    print("rename 后列名：", ds.column_names)
    print("一条样本：", ds[0])
    print("👉 filter 筛行、map 变换/加列、rename 改列名——数据清洗三板斧。")

def case_02():
    title(2, "文本嵌入：把一句话变成 768 维向量")
    tok, model, get_embeddings = load_embedder()
    text = "How can I load a dataset offline?"
    emb=get_embeddings(text)
    print(f"句子：{text}")
    print(f"嵌入向量 shape：{tuple(emb.shape)}  （1 句 × 768 维）")
    print(f"向量前 5 个数：{emb[0][:5].tolist()}")
    print("👉 语义相近的句子，向量也相近——这是语义搜索的基础。")

def case_03():
    title(3, "手动语义搜索：嵌入 → 点积 → top-k")
    tok, model, get_embeddings = load_embedder()
    corpus_emb=get_embeddings(CORPUS)
    query = "How do I use a dataset without internet?"
    q_emb=get_embeddings([query])
    scores=(q_emb@corpus_emb.T)[0]
    topk=torch.topk(scores,3)
    print(f"问题：{query}\n最相关的 3 条：")
    for score, idx in zip(topk.values, topk.indices):
        print(f"  [{score.item():.1f}] {CORPUS[idx.item()]}")
    print("👉 注意 top1 是'load a dataset offline'——用词不同(offline vs without internet)，")
    print("   但语义匹配上了。这就是语义搜索。")

def case_04():
    title(4, "FAISS 索引检索：add_faiss_index + get_nearest_examples")
    from datasets import Dataset
    tok, model, get_embeddings = load_embedder()
    ds = Dataset.from_dict({"text": CORPUS})
    ds=ds.map(lambda x:{"embeddings":get_embeddings(x["text"]).cpu().numpy()[0]})
    ds.add_faiss_index("embeddings")
    query = "way to search text by meaning"
    q_emb=get_embeddings([query]).cpu().numpy()
    scores, samples =ds.get_nearest_examples("embeddings",q_emb,k=2)
    print(f"问题：{query}\n最相关的 3 条（已按最近排序，第1条最相关）：")
    for score, text in zip(scores, samples["text"]):
        print(f"  [{score:.1f}] {text}")
    print("⚠️ 注意：FAISS 默认用 L2 距离，分数【越小越近】(和第3关点积越大越相似相反)，")
    print("   但 get_nearest_examples 已帮你按最近排好序，第1条就是最相关的。")
    print("👉 add_faiss_index 一行建索引，get_nearest_examples 一行检索。")
    print("   真实项目里语料几万几百万条，faiss 也能秒查——这就是工业级语义搜索。")

def case_05():
    title(5, "★语义搜索 vs 关键词搜索：换个说法还能不能搜到")
    tok, model, get_embeddings = load_embedder()
    corpus_emb = get_embeddings(CORPUS)
    query = "process data that does not fit in RAM"
    print(f"问题：{query}")
    q_words=set(query.lower().split())
    kw_hits=[d for d in CORPUS if q_words & set(d.lower().split())]
    print(f"\n[关键词搜索] 命中 {len(kw_hits)} 条：")
    for d in kw_hits:
        print(f"   {d}")
    print("   （注意：'RAM' 'fit' 这些词语料里没有，关键词很可能漏掉正确答案）")
    scores =(get_embeddings([query])@corpus_emb.T)[0]
    best=CORPUS[torch.argmax(scores).item()]
    print(f"\n[语义搜索] top1：{best}")
    print("👉 语义搜索理解'放不进RAM'≈'比内存大'，即使一个关键词都不重合也能命中。")

def case_06():
    title(6, "综合：迷你语义搜索引擎（RAG 检索雏形）")
    from datasets import Dataset
    tok, model, get_embeddings = load_embedder()
    ds = Dataset.from_dict({"text": CORPUS})
    ds = ds.map(lambda x: {'embeddings': get_embeddings(x["text"]).cpu().numpy()[0]})
    ds = ds.add_faiss_index(column="embeddings")

    def search(query, k=2):
        q_emb = get_embeddings([query]).cpu().numpy()
        scores, samples = ds.get_nearest_examples("embeddings", q_emb, k=k)
        return list(zip(scores, samples["text"]))

    questions = [
        "how to train a model on my own data",
        "make batches the same length",
        "serve a model behind an API",
    ]
    for q in questions:
        print(f"\n❓ {q}")
        for score, text in search(q, k=2):
            print(f"   [{score:.1f}] {text}")
    print("\n👉 这就是 RAG 的前半段：把用户问题→检索最相关文档。")
    print("   后半段接一个 LLM，把'检索到的文档 + 问题'一起给它生成答案，就是完整 RAG。")
    print("   （你作品集项目2就是把这个做成带界面、能上线的应用）")


CASES = {1: case_01, 2: case_02, 3: case_03, 4: case_04, 5: case_05, 6: case_06}

MENU = """\
用法：
  python3 Chapter5_案例闯关_语义搜索实战.py <关号>   跑单关，如 3 / 5 / 6
  python3 Chapter5_案例闯关_语义搜索实战.py all       跑全部

关卡列表（3~6 是语义搜索 = RAG 检索核心）：
  1  数据集处理三招     2  文本嵌入          3  手动语义搜索
  4  FAISS 索引检索     5  语义 vs 关键词     6  综合:迷你搜索引擎
"""


def run_choice(choice: str):
    choice = choice.strip().lower()
    if choice == "all":
        for n in sorted(CASES):
            CASES[n]()
        return
    try:
        n = int(choice)
    except ValueError:
        print(f"没有第 {choice} 关，请输入 1~6 或 all。")
        return
    if n not in CASES:
        print(f"没有第 {choice} 关，请输入 1~6 或 all。")
        return
    CASES[n]()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        run_choice(args[0])
    else:
        print(MENU)
        while True:
            choice = input("请输入关号（1~6，或 all 全部，回车/q 退出）：")
            if choice.strip().lower() in ("", "q", "quit", "exit"):
                print("已退出。")
                break
            run_choice(choice)
            print()
