"""
================================================================================
 中文版 · 语义检索 / RAG（用中文嵌入模型 bge-small-zh）
================================================================================
 为什么用中文嵌入：英文嵌入(all-MiniLM)对中文语义把握差；BAAI 的 bge 系列是中文最常用的强嵌入。
 两个中文特有细节(和英文版不同)：
   ① bge 用 CLS 池化(取 [:,0])，不是 mean 池化；
   ② bge 检索时建议给“查询”加前缀提示，能提升召回(文档不用加)。
 (对应英文版 ../分章项目/Ch5_语义搜索引擎.py、../案例/案例2)
 跑：python3 中文_语义检索RAG.py     (首次下 bge-small-zh ~95MB)
================================================================================
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ckpt = "BAAI/bge-small-zh-v1.5"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModel.from_pretrained(ckpt).eval()
# bge 官方建议：检索时给“查询”加这个前缀(文档不加)，提升召回
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

FAQ = [
    "重置密码：去 设置 > 安全 > 重置密码，按邮件链接操作。",
    "上传照片闪退：请升级到 App v3.2 或更高版本，该问题已修复。",
    "重复扣款：核实后 3-5 个工作日内退还。",
    "联系人工：在聊天里输入 'agent'，或工作日 9-18 点拨打热线。",
    "存储空间：免费账户 5GB，升级 Pro 得 1TB。",
]


def embed(texts, is_query=False):
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]     # 查询加前缀(bge 中文检索的技巧)
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state[:, 0]     # ★bge 用 CLS 池化(取第0个token)，不是 mean
    return F.normalize(out, p=2, dim=1)                # 归一化 → 点积=余弦


faq_vecs = embed(FAQ)                                  # 文档不加前缀
for query in ["我忘记登录密码了怎么办",                 # ~ 重置密码
              "照片老是传不上去还闪退",                  # ~ 崩溃
              "免费的能存多少东西"]:                     # ~ 存储
    sims = (embed([query], is_query=True) @ faq_vecs.T)[0]
    best = int(sims.argmax())
    print(f"问: {query}\n  → [{sims[best]:.2f}] {FAQ[best]}")

assert "密码" in FAQ[int((embed(["忘记密码"], is_query=True) @ faq_vecs.T)[0].argmax())]
print("\n✅ 中文语义检索跑通(bge-small-zh，CLS池化 + 查询前缀)。换说法也能命中。")
# print("面试：Q 中文检索为什么不用英文嵌入? A 语义空间不对齐,中文检索要用中文/多语言嵌入(bge/text2vec)。")
# print("      Q bge 和 sentence-transformers 池化区别? A bge 常用 CLS 池化,MiniLM 用 mean 池化,别搞混。")
