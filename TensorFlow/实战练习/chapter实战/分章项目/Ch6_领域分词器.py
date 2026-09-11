"""
================================================================================
 分章项目 · Ch6 · 分词器（贴 HF Ch6：领域重训 + 快速分词器 + 三大算法 + 流水线）
================================================================================
 HF 课程 Ch6 讲透分词器，用可运行代码复现四块：
   ① 领域重训：通用分词器对特定领域(代码)切得碎、序列长、推理贵；在领域语料上重训，
      让高频术语整块成 token，序列显著变短(=省显存/延迟/成本，真金白银)。
   ② 快速分词器的超能力：offset_mapping(每个 token 对应原文哪段) + word_ids(子词属于第几个词)——
      NER/QA 全靠它把子词对齐回原文(见 Ch7)。
   ③ 三大子词算法：BPE(GPT) / WordPiece(BERT) / Unigram(T5/XLNet) 的区别。
   ④ 分词四步流水线：normalization → pre-tokenization → model(子词切分) → post-processing。
   完整章节材料见 ../../../Chapter 6/。
 跑：python3 Ch6_领域分词器.py     # 在真实代码语料上重训，量化压缩率
================================================================================
"""
import pandas as pd
from transformers import AutoTokenizer


# ==============================================================================
# ① 领域重训分词器：在真实 Python 代码语料上重训一份词表
# ==============================================================================
def train_domain_tokenizer():
    # print("=" * 70, "\n① 领域重训分词器(代码语料)\n" + "=" * 70)
    df = pd.read_parquet(
        "hf://datasets/code-search-net/code_search_net/python/train-00000-of-00001.parquet")
    texts = df["whole_func_string"].dropna().sample(n=4000, random_state=42).tolist()
    eval_texts = df["whole_func_string"].dropna().sample(n=300, random_state=7).tolist()

    base = AutoTokenizer.from_pretrained("gpt2")
    def corpus():                                        # 生成器惰性喂入，重训词表(算法结构复用 gpt2=BPE)
        for i in range(0, len(texts), 1000):
            yield texts[i:i + 1000]
    domain = base.train_new_from_iterator(corpus(), vocab_size=20000)

    def avg_tokens(t):
        return sum(len(ids) for ids in
                   t(eval_texts, truncation=True, max_length=2000)["input_ids"]) / len(eval_texts)
    a, b = avg_tokens(base), avg_tokens(domain)
    print(f"  通用 gpt2   : {a:.1f} tokens/函数")
    print(f"  领域分词器  : {b:.1f} tokens/函数")
    print(f"  ⇒ 序列缩短 {(a-b)/a*100:.1f}%  (等比例省推理算力/显存/成本)")
    sample = "def compute_loss(self, predictions, targets):\n    return self.criterion(predictions, targets)"
    print("\n  gpt2 :", base.tokenize(sample)[:14], "...")
    print("  领域 :", domain.tokenize(sample)[:14], "...")
    return base


# ==============================================================================
# ② 快速分词器的超能力：offset_mapping + word_ids
# ==============================================================================
def fast_tokenizer_powers():
    # print("\n" + "=" * 70, "\n② 快速分词器：offset_mapping + word_ids(子词对齐的关键)\n" + "=" * 70)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")   # fast tokenizer
    enc = tok("Hugging Face is based in Brooklyn", return_offsets_mapping=True)
    toks = tok.convert_ids_to_tokens(enc["input_ids"])
    print(f"  {'token':<10}{'word_id':<9}{'offset(原文字符区间)'}")
    for t, wid, off in zip(toks, enc.word_ids(), enc["offset_mapping"]):
        print(f"  {t:<10}{str(wid):<9}{off}")
    # print("  word_ids：子词属于原文第几个词(None=特殊符)；offset：token 对应原文的字符区间。")
    # print("  NER/QA 就靠这俩把模型的 token 级预测，映射回原文里的完整词/span。")


# ==============================================================================
# ③ 三大子词算法 & ④ 分词四步流水线
# ==============================================================================
def algorithms_and_pipeline():
    # print("\n" + "=" * 70, "\n③ 三大子词算法  &  ④ 分词四步流水线\n" + "=" * 70)
    # print("  ③ 子词算法：")
    for name, how, who in [
        ("BPE",       "从字符起，反复合并最高频的相邻对", "GPT/GPT-2/RoBERTa"),
        ("WordPiece", "按“合并后能最大化语料似然”选合并", "BERT/DistilBERT"),
        ("Unigram",   "从大词表起，逐步删掉贡献小的子词", "T5/XLNet/ALBERT"),
    ]:
        print(f"     {name:<10}{how:<26}{who}")
    # print("  共同目标：在“字符级(序列太长)”和“词级(词表爆炸、OOV)”之间折中——子词。")
    # print("\n  ④ 分词四步流水线(fast tokenizer 内部)：")
    # print("     normalization(小写/去重音/清洗) → pre-tokenization(按空格/标点粗切) →")
    # print("     model(BPE/WordPiece/Unigram 切子词) → post-processing(加 [CLS]/[SEP] 等特殊符)")


if __name__ == "__main__":
    train_domain_tokenizer()
    fast_tokenizer_powers()
    algorithms_and_pipeline()
    print("\n✅ Ch6 跑通：领域重训(省序列) → 快速分词器 offset/word_ids → 三大算法 → 四步流水线。")
    # print("面试：Q 领域词汇被切碎怎么办? Q BPE/WordPiece/Unigram 区别? Q offset_mapping 有什么用?"
          # " (见 ../面试高频题库.py 三)")
