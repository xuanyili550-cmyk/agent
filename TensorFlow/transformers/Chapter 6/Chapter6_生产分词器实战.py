"""
================================================================================
 Chapter 6 · 生产级分词器实战（真实语料训练领域分词器 + 评估 + 打包 + 基准）
================================================================================
 “能写进简历/能上线”的版本，不是小语料手写算法演示：
   · 真实语料(IMDB 影评)训练一个领域自适应分词器，不是几句玩具句子
   · 量化评估：领域分词器 vs 通用 gpt2 在同一批文本上的“压缩率(平均 token 数)”
     —— token 越少 = 推理越便宜越快，这是训练领域分词器的直接收益
   · 生产闭环：训练 → 保存 → 重新加载(往返一致) → 打包成 PreTrainedTokenizerFast → 基准测速

 为什么企业要自训分词器：通用分词器对特定领域(代码/医疗/法律/某语言)常把词切得很碎，
 序列变长 → 显存/延迟/成本都上去。领域分词器让高频术语整块成 token，直接省钱提速。

 用法：
   python3 Chapter6_生产分词器实战.py                # 真训练(小子集，~1-2 分钟)
   python3 Chapter6_生产分词器实战.py --docs 20000    # 更大语料，压缩收益更明显
================================================================================
"""

import argparse
import os
import tempfile
import time
from dataclasses import dataclass


@dataclass
class Config:
    # 领域=Python 代码：通用 gpt2 对代码(缩进/下划线命名)切得很碎，正是自训分词器的用武之地
    corpus_dataset: str = "code-search-net/code_search_net"
    dataset_config: str = "python"
    text_column: str = "whole_func_string"     # 每条=一个完整 Python 函数源码
    base_tokenizer: str = "gpt2"               # 拿它的算法结构来重训词表
    vocab_size: int = 20000
    n_docs: int = 8000                         # 训练用文档数(子集，可调大)
    n_eval: int = 500                          # 评估用文档数
    batch: int = 1000                          # 喂给训练器的批大小
    output_dir: str = "code-tokenizer"


def load_corpus(cfg):
    from datasets import load_dataset
    ds = load_dataset(cfg.corpus_dataset, cfg.dataset_config, split="train").shuffle(seed=42)
    col = cfg.text_column
    # 新版 datasets 的 ds[col] 是 Column 对象，分词器要 list[str]，这里显式转 list
    train_txt = list(ds.select(range(min(cfg.n_docs, len(ds))))[col])
    eval_txt = list(ds.select(range(cfg.n_docs, cfg.n_docs + cfg.n_eval))[col])
    return train_txt, eval_txt


def get_training_corpus(texts, batch):
    """生成器惰性喂入：每次吐一批，省内存(生产处理大语料的标准做法)。"""
    for i in range(0, len(texts), batch):
        yield texts[i:i + batch]


def avg_tokens(tokenizer, texts):
    """平均每篇文档被切成多少 token —— 越少说明这个分词器对该领域越“高效”。"""
    total = sum(len(ids) for ids in
                tokenizer(texts, truncation=True, max_length=2000)["input_ids"])
    return total / len(texts)


def main(cfg):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    print(f">>> 语料={cfg.corpus_dataset}  基座={cfg.base_tokenizer}  目标词表={cfg.vocab_size}")
    train_txt, eval_txt = load_corpus(cfg)
    print(f">>> 训练文档 {len(train_txt)} 篇，评估文档 {len(eval_txt)} 篇")

    # --- 1) 训练领域分词器：复用 gpt2 的算法结构，在影评语料上重学词表 ---
    base = AutoTokenizer.from_pretrained(cfg.base_tokenizer)
    t0 = time.time()
    domain_tok = base.train_new_from_iterator(
        get_training_corpus(train_txt, cfg.batch), vocab_size=cfg.vocab_size)
    print(f">>> 训练完成，用时 {time.time()-t0:.1f}s，新词表 {len(domain_tok)} 个 token")

    # --- 2) 量化评估：领域分词器 vs 通用 gpt2 的压缩率(平均 token 数，越低越好) ---
    a_generic = avg_tokens(base, eval_txt)
    a_domain = avg_tokens(domain_tok, eval_txt)
    save = (a_generic - a_domain) / a_generic * 100
    print("\n>>> 压缩率评估(同一批代码上，平均每篇函数的 token 数)：")
    print(f"    通用 gpt2      : {a_generic:8.1f} tokens/篇")
    print(f"    领域分词器     : {a_domain:8.1f} tokens/篇")
    print(f"    ⇒ 序列缩短 {save:.1f}%  (等比例省推理算力/显存/成本)")

    # 具体看一段代码切法差异(代码最能体现领域分词器的优势)
    sample = "def compute_loss(self, predictions, targets):\n    return self.criterion(predictions, targets)"
    print("\n>>> 同一段代码切法对比：")
    print("    gpt2  :", base.tokenize(sample))
    print("    领域  :", domain_tok.tokenize(sample))

    # --- 3) 保存 → 重新加载(往返一致性，生产必须保证) ---
    domain_tok.save_pretrained(cfg.output_dir)
    reloaded = AutoTokenizer.from_pretrained(cfg.output_dir)
    assert reloaded(sample)["input_ids"] == domain_tok(sample)["input_ids"]
    print(f"\n>>> 已保存到 {cfg.output_dir}/ 并验证重载后编码完全一致 ✅")

    # --- 4) 打包成 PreTrainedTokenizerFast（就是它现在的类型，可直接配 AutoModel 用） ---
    assert domain_tok.is_fast and isinstance(reloaded, PreTrainedTokenizerFast)
    print(">>> 类型:", type(reloaded).__name__, "(快速分词器，可直接喂给 AutoModel/Trainer)")

    # --- 5) 基准测速：快速分词器批量编码吞吐(生产关心的指标) ---
    bench = eval_txt[:200]
    t0 = time.time(); _ = domain_tok(bench, truncation=True, max_length=512)
    dt = time.time() - t0
    print(f"\n>>> 基准：批量编码 {len(bench)} 篇用 {dt*1000:.0f}ms "
          f"({len(bench)/dt:.0f} 篇/秒)  —— 快速分词器(Rust)才有的吞吐")
    print("\n✅ 生产分词器闭环跑通：真实语料训练 → 评估压缩率 → 保存重载 → 打包 → 基准。")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=8000, help="训练文档数(越多压缩收益越明显)")
    p.add_argument("--vocab", type=int, default=20000)
    args = p.parse_args()
    main(Config(n_docs=args.docs, vocab_size=args.vocab))
