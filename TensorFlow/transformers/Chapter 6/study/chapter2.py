import argparse
import time
from dataclasses import dataclass

@dataclass
class Config:
    corpus_dataset: str = "code-search-net/code_search_net"
    dataset_config: str = "python"
    text_column: str = "whole_func_string"  # 每条=一个完整 Python 函数源码
    base_tokenizer: str = "gpt2"  # 拿它的算法结构来重训词表
    vocab_size: int = 20000
    n_docs: int = 8000  # 训练用文档数(子集，可调大)
    n_eval: int = 500  # 评估用文档数
    batch: int = 1000  # 喂给训练器的批大小
    output_dir: str = "code-tokenizer"

def load_corpus(cfg):
    from datasets import load_dataset
    ds=load_dataset(cfg.corpus_dataset,cfg.dataset_config,split='train').shuffle(seed=42)
    col=cfg.text_column
    train_txt = list(ds.select(range(min(cfg.n_docs, len(ds))))[col])
    eval_txt = list(ds.select(range(cfg.n_docs, cfg.n_docs + cfg.n_eval))[col])
    return train_txt, eval_txt

def get_training_corpus(text,batch):
    for i in range(0,len(text),batch):
        yield text[i:i+batch]

def avg_tokens(tokenizer,texts):
    total=sum(len(ids) for ids in tokenizer(texts,truncation=True, max_length=2000)["input_ids"])
    return total/len(texts)
def main(cfg):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    train_txt, eval_txt = load_corpus(cfg)
    base = AutoTokenizer.from_pretrained(cfg.base_tokenizer)
    t0 = time.time()
    domain_tok = base.train_new_from_iterator(
        get_training_corpus(train_txt, cfg.batch), vocab_size=cfg.vocab_size)
    a_generic = avg_tokens(base, eval_txt)
    a_domain = avg_tokens(domain_tok, eval_txt)
    save = (a_generic - a_domain) / a_generic * 100
    sample = "def compute_loss(self, predictions, targets):\n    return self.criterion(predictions, targets)"
    domain_tok.save_pretrained(cfg.output_dir)
    reloaded = AutoTokenizer.from_pretrained(cfg.output_dir)
    assert reloaded(sample)["input_ids"] == domain_tok(sample)["input_ids"]
    assert domain_tok.is_fast and isinstance(reloaded, PreTrainedTokenizerFast)
    bench = eval_txt[:200]
    t0 = time.time(); _ = domain_tok(bench, truncation=True, max_length=512)
    dt = time.time() - t0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=8000, help="训练文档数(越多压缩收益越明显)")
    p.add_argument("--vocab", type=int, default=20000)
    args = p.parse_args()
    main(Config(n_docs=args.docs, vocab_size=args.vocab))
