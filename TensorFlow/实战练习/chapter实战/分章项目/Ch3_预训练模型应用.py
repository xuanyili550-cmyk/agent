"""
================================================================================
 分章项目 · Ch3 · 使用预训练模型（贴 HF：fill-mask + 模型头家族 + 特征抽取）
================================================================================
 HF 课程里“直接用 Hub 上预训练模型”的核心，用可运行代码复现：
   ① MLM 完形填空：掩码语言模型(BERT 类)的看家能力，手写版看清 pipeline 内部怎么定位 mask/取 top-k。
   ② MLM vs CLM：双向遮词填空(BERT) vs 单向预测下一个词(GPT)——同一句话两种模型给的“下一步”不同。
   ③ 模型头家族(AutoModelForX)：同一个基座 + 不同的“头”= 不同任务(填空/分类/生成/取特征)。
   ④ 特征抽取：AutoModel(无头)输出隐藏状态，可当句/词向量(RAG、聚类的地基，见 Ch5)。
   完整章节材料见 ../../../Chapter 3/。
 跑：python3 Ch3_预训练模型应用.py
================================================================================
"""
import torch
from transformers import (AutoTokenizer, AutoModel, AutoModelForMaskedLM,
                          AutoModelForCausalLM)

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# ① MLM 完形填空(手写：定位 [MASK] → 该位置 logits → softmax → top-k)
# ==============================================================================
def fill_mask():
    # print("=" * 70, "\n① MLM 完形填空(BERT 类，双向)\n" + "=" * 70)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased").to(DEV).eval()
    for sentence in [f"The capital of France is {tok.mask_token}.",
                     f"Machine learning is a subfield of artificial {tok.mask_token}."]:
        enc = tok(sentence, return_tensors="pt").to(DEV)
        with torch.no_grad():
            logits = model(**enc).logits
        mask_pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
        probs = torch.softmax(logits[0, mask_pos], dim=-1)[0]
        top = torch.topk(probs, 5)
        print(f"  {sentence}")
        print("   top5:", [(tok.decode([t]), round(float(s), 3)) for t, s in zip(top.indices, top.values)])


# ==============================================================================
# ② MLM vs CLM：同一句“下一步”，两类模型给的不一样
# ==============================================================================
def mlm_vs_clm():
    # print("\n" + "=" * 70, "\n② MLM(双向填空) vs CLM(单向续写)\n" + "=" * 70)
    # CLM：GPT 类只能看左边，预测“下一个词”
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    clm = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEV).eval()
    enc = tok("The capital of France is", return_tensors="pt").to(DEV)
    with torch.no_grad():
        next_logits = clm(**enc).logits[0, -1]          # 只看最后一个位置=预测下一个词
    top = torch.topk(torch.softmax(next_logits, -1), 5)
    print("  CLM(distilgpt2) 续写 'The capital of France is' →",
          [tok.decode([t]).strip() for t in top.indices])
    # print("  区别：MLM 用两边上下文填中间的空(适合理解/编码)；CLM 只用左边预测右边(适合生成)。")


# ==============================================================================
# ③ 模型头家族：同一基座 + 不同的头 = 不同任务
# ==============================================================================
def head_zoo():
    # print("\n" + "=" * 70, "\n③ AutoModelForX 头家族(同基座不同头)\n" + "=" * 70)
    zoo = [
        ("AutoModel", "无头，输出隐藏状态", "取特征/句向量(RAG、聚类)"),
        ("AutoModelForMaskedLM", "MLM 头", "完形填空(BERT 预训练任务)"),
        ("AutoModelForCausalLM", "LM 头", "文本生成/对话(GPT 类)"),
        ("AutoModelForSequenceClassification", "分类头", "情感/意图/NLI"),
        ("AutoModelForTokenClassification", "逐token分类头", "NER/词性标注(见 Ch7)"),
    ]
    print(f"  {'AutoClass':<38}{'加的头':<18}{'典型用途'}")
    for cls, head, use in zoo:
        print(f"  {cls:<38}{head:<18}{use}")
    # print("  同一个 bert-base 权重，换不同 AutoModelForX 就是换任务的“头”，主干(表示)复用。")


# ==============================================================================
# ④ 特征抽取：AutoModel(无头) 输出隐藏状态 → 句向量
# ==============================================================================
def feature_extraction():
    # print("\n" + "=" * 70, "\n④ 特征抽取(AutoModel 无头 → 句向量)\n" + "=" * 70)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").to(DEV).eval()
    texts = ["I love cats.", "I adore kittens.", "The stock market crashed."]
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt").to(DEV)
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state              # [batch, seq, dim] 每个token一个向量
    mask = enc["attention_mask"].unsqueeze(-1).float()
    sent_vec = (hidden * mask).sum(1) / mask.sum(1)          # mean 池化(避开 PAD)→ 句向量
    sent_vec = torch.nn.functional.normalize(sent_vec, dim=1)
    print(f"  隐藏状态形状={tuple(hidden.shape)}  句向量维度={sent_vec.shape[1]}")
    print(f"  '爱猫' vs '爱小猫' 余弦 = {float(sent_vec[0] @ sent_vec[1]):.3f}（近）")
    print(f"  '爱猫' vs '股市崩盘' 余弦 = {float(sent_vec[0] @ sent_vec[2]):.3f}（远）")
    # print("  无头 AutoModel 的隐藏状态就是“表示/特征”，语义搜索(Ch5)/RAG 都建在它上面。")


if __name__ == "__main__":
    print(f">>> 设备={DEV}\n")
    fill_mask()
    mlm_vs_clm()
    head_zoo()
    feature_extraction()
    print("\n✅ Ch3 跑通：MLM 填空 → MLM/CLM 对比 → 模型头家族 → 特征抽取。")
    # print("面试：Q MLM 和 CLM 区别? Q AutoModel 和 AutoModelForX 区别? Q 怎么用预训练模型取句向量?"
          # " (见 ../面试高频题库.py)")
