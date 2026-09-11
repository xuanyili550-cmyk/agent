"""
================================================================================
 Chapter 7 案例闯关 · 主要 NLP 任务实战（6 关，每关都真跑一个任务）
================================================================================
 用法：
   python3 Chapter7_案例闯关_主要NLP任务实战.py          # 看菜单
   python3 Chapter7_案例闯关_主要NLP任务实战.py 2         # 只跑第 2 关
   python3 Chapter7_案例闯关_主要NLP任务实战.py all        # 全部(会下多个模型，慢)
 也可在 PyCharm 直接点运行 → 按提示输入关号。

 关卡地图（每关用“尽量小”的模型，做真实推理，Mac 上能跑）：
   1  NER 命名实体识别    pipeline 识别人名/地名/机构名
   2  MLM 完形填空        给 [MASK] 填词，看 top-5 候选
   3  翻译 en→中文        Marian(opus-mt) 英译中
   4  摘要 Summarization  t5-small 把长文压成一句
   5  文本/代码生成       distilgpt2 续写
   6  QA 抽取式问答       distilbert-squad 从 context 抠答案

 说明：
   · 每关首次会下载对应模型(见各关注释的大致体积)，之后走缓存。
   · 建议一次只跑一关(下一个模型)，别用 all(会把 6 个模型都下下来)。
   · 没装 transformers/torch 会打印提示并跳过，不影响其它关。
================================================================================
"""

import sys


def title(n, text):
    print("\n" + "=" * 72 + f"\n  第 {n} 关：{text}\n" + "=" * 72)


def need(*mods):
    """检查依赖，缺了就返回 False 并提示。"""
    import importlib
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if missing:
        print(f"  需要：pip install {' '.join(missing)}")
        return False
    return True


def device_id():
    """pipeline 的 device 参数：有 CUDA 用 0，其它(含苹果 mps)让 pipeline 自己挑。"""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return -1


def pick_device():
    """给“直接调模型”用的设备字符串(cuda/mps/cpu)。"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# 注意：transformers v5 移除了 translation / summarization / question-answering 这几个
# 便捷 pipeline，所以第 3/4/6 关改成“直接加载模型 + generate()/取 logits”，更贴近底层。


# ==============================================================================
# 第 1 关：NER 命名实体识别（识别人名/地名/机构名）
# ==============================================================================
def case1_ner():
    title(1, "NER 命名实体识别（~430MB 模型）")
    if not need("transformers", "torch"):
        return
    from transformers import pipeline
    # aggregation_strategy="simple"：把 S/##yl/##va/##in 这些子词合并成完整实体 'Sylvain'
    ner = pipeline("token-classification",
                   model="huggingface-course/bert-finetuned-ner",
                   aggregation_strategy="simple", device=device_id())
    text = "My name is Sylvain and I work at Hugging Face in Brooklyn."
    print("  句子：", text)
    for ent in ner(text):
        print(f"   {ent['entity_group']:5} {ent['word']!r:16} 置信度={ent['score']:.3f}")
    print("  要点：模型给每个 token 打 BIO 标签，aggregation 再把同一实体的子词拼回原词。")


# ==============================================================================
# 第 2 关：MLM 完形填空（给 [MASK] 填词）
# ==============================================================================
def case2_fill_mask():
    title(2, "MLM 完形填空（~268MB 模型）")
    if not need("transformers", "torch"):
        return
    from transformers import pipeline
    filler = pipeline("fill-mask", model="distilbert-base-uncased", device=device_id())
    for text in ["This movie is really [MASK].", "Paris is the capital of [MASK]."]:
        print(f"\n  句子：{text}")
        for p in filler(text, top_k=5):
            print(f"   {p['token_str']:12} 概率={p['score']:.3f}")
    print("\n  要点：MLM 双向看上下文预测被遮的词；这就是 BERT 的预训练任务。")


# ==============================================================================
# 第 3 关：翻译 en→中文（Marian / opus-mt）
# ==============================================================================
def case3_translation():
    title(3, "翻译 英译中（~300MB 模型，需要 sentencepiece）")
    if not need("transformers", "torch", "sentencepiece"):
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    ckpt = "Helsinki-NLP/opus-mt-en-zh"
    dev = pick_device()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(dev).eval()
    for en in ["Hugging Face is a company based in New York.",
               "This plugin lets you translate web pages automatically."]:
        inputs = tok(en, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=128)      # 解码器逐词生成中文
        print(f"  EN: {en}")
        print(f"  ZH: {tok.decode(out[0], skip_special_tokens=True)}\n")
    print("  要点：翻译是 seq2seq(编码器读英文→解码器 generate 生成中文)，训练时目标句用 text_target 编码。")


# ==============================================================================
# 第 4 关：摘要 Summarization（t5-small）
# ==============================================================================
def case4_summarization():
    title(4, "摘要 Summarization（t5-small ~240MB，需要 sentencepiece）")
    if not need("transformers", "torch", "sentencepiece"):
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    dev = pick_device()
    tok = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(dev).eval()
    article = (
        "The tower is 324 metres tall, about the same height as an 81-storey building, "
        "and the tallest structure in Paris. Its base is square, measuring 125 metres on "
        "each side. During its construction, the Eiffel Tower surpassed the Washington "
        "Monument to become the tallest man-made structure in the world, a title it held "
        "for 41 years until the Chrysler Building in New York City was finished in 1930.")
    # ★T5 靠“任务前缀”区分任务：摘要就在正文前加 "summarize: "
    inputs = tok("summarize: " + article, return_tensors="pt", truncation=True).to(dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=40, min_length=10, num_beams=4)
    print("  原文长度:", len(article), "字符")
    print("  摘要:", tok.decode(out[0], skip_special_tokens=True))
    print("  要点：T5 把所有任务都当“文本→文本”，摘要时在正文前加 'summarize:' 前缀。")


# ==============================================================================
# 第 5 关：文本/代码生成（因果语言模型 distilgpt2）
# ==============================================================================
def case5_generation():
    title(5, "文本生成（因果LM distilgpt2 ~350MB）")
    if not need("transformers", "torch"):
        return
    from transformers import pipeline, set_seed
    gen = pipeline("text-generation", model="distilgpt2", device=device_id())
    set_seed(42)
    for prompt in ["Machine learning is", "def add(a, b):"]:
        out = gen(prompt, max_new_tokens=25, num_return_sequences=1)[0]["generated_text"]
        print(f"  提示: {prompt!r}\n  续写: {out!r}\n")
    print("  要点：因果LM 只看左边、逐词预测下一个；这就是 GPT 家族“写下去”的原理。")


# ==============================================================================
# 第 6 关：QA 抽取式问答（distilbert-squad）
# ==============================================================================
def case6_qa():
    title(6, "QA 抽取式问答（~260MB 模型，手写取答案）")
    if not need("transformers", "torch"):
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    ckpt = "distilbert-base-cased-distilled-squad"
    dev = pick_device()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt).to(dev).eval()
    context = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
               "Paris, France. It is named after the engineer Gustave Eiffel, whose company "
               "designed and built the tower. It was completed in 1889.")
    print("  上下文：", context, "\n")
    for q in ["Who is the Eiffel Tower named after?",
              "When was the Eiffel Tower completed?",
              "Where is the Eiffel Tower?"]:
        inputs = tok(q, context, return_tensors="pt", return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping")[0]
        with torch.no_grad():
            out = model(**{k: v.to(dev) for k, v in inputs.items()})
        # 取起点/终点分数最高的 token，再用 offset 把 token 区间换回原文字符子串
        start = int(out.start_logits.argmax())
        end = int(out.end_logits.argmax())
        answer = context[offsets[start][0]:offsets[end][1]]
        print(f"  Q: {q}\n  A: {answer!r} (token 区间[{start},{end}])\n")
    print("  要点：抽取式 QA 预测答案的 start/end token，再用 offset 换回原文子串。")


CASES = {
    1: case1_ner, 2: case2_fill_mask, 3: case3_translation,
    4: case4_summarization, 5: case5_generation, 6: case6_qa,
}


def menu():
    print(__doc__)
    print("请输入关号(1-6) / all，然后回车：", end="")
    try:
        return input().strip()
    except EOFError:
        return "1"


def run(choice):
    if choice == "all":
        for n in sorted(CASES):
            CASES[n]()
    elif choice.isdigit() and int(choice) in CASES:
        CASES[int(choice)]()
    else:
        print(f"没有第 {choice} 关，可选：{sorted(CASES)} 或 all")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else menu())
