"""
================================================================================
 Chapter 3 案例闯关 · 使用预训练模型（6 关，全部能在 Mac 上跑）
================================================================================
 用法：
   python3 Chapter3_案例闯关_预训练模型实战.py         # 看菜单
   python3 Chapter3_案例闯关_预训练模型实战.py 1        # 只跑第 1 关
   python3 Chapter3_案例闯关_预训练模型实战.py all       # 全部
 也可在 PyCharm 直接点运行 → 按提示输入关号。

 关卡地图：
   1  fill-mask 初体验     用预训练模型做完形填空
   2  多语言对比           英/中/法 三个模型各做一次完形填空
   3  ★mask token 的坑     bert 用[MASK]、camembert 用<mask>，用错就崩
   4  Auto 类 vs 专用类     两种加载方式，证明结果完全一致
   5  手动 fill-mask       不用 pipeline，手写：分词→模型→找mask→softmax→topk [结合Ch1]
   6  综合：探测模型的“知识” 用完形填空看模型学到了什么常识

 说明：优先用你已下载的模型(bert-base-uncased/chinese、camembert-base)，跑得快。
================================================================================
"""

import sys
import torch


def title(n, text):
    print("\n" + "=" * 72)
    print(f"  第 {n} 关：{text}")
    print("=" * 72)


# ==============================================================================
# 第 1 关：fill-mask 初体验
# ==============================================================================
def case_01():
    title(1, "fill-mask 初体验：预训练模型天生会完形填空")
    from transformers import pipeline

    fm = pipeline("fill-mask", model="bert-base-uncased")
    # bert 用 [MASK] 作为掩码符号
    text = "The capital of France is [MASK]."
    print(f"句子：{text}\n模型猜被挖的词（top 5）：")
    for r in fm(text):
        print(f"  {r['token_str']:12s} 概率 {r['score']:.3f} → {r['sequence']}")
    print("👉 没微调、没训练，直接就能填。因为 fill-mask 是 BERT 预训练时的原生技能。")


# ==============================================================================
# 第 2 关：多语言对比
# ==============================================================================
def case_02():
    title(2, "多语言对比：不同语言要用对应语言的模型")
    from transformers import pipeline

    tasks = [
        ("英文", "bert-base-uncased", "Paris is the [MASK] of France."),
        ("中文", "bert-base-chinese", "北京是中国的[MASK]。"),
        ("法语", "camembert-base",    "Le camembert est <mask> :)"),
    ]
    for lang, model, text in tasks:
        fm = pipeline("fill-mask", model=model)
        top = fm(text)[0]
        print(f"  [{lang}] {text}")
        print(f"       → 最佳答案：{top['token_str']}（{top['score']:.3f}）")
    print("👉 注意每个模型的 mask 符号不同：英/中的 bert 用 [MASK]，法语 camembert 用 <mask>。")


# ==============================================================================
# 第 3 关：mask token 的坑
# ==============================================================================
def case_03():
    title(3, "★mask token 的坑：写死符号会崩，应动态取")
    from transformers import pipeline

    fm = pipeline("fill-mask", model="bert-base-uncased")
    # ★ 稳妥做法：从 tokenizer 动态取 mask 符号，不写死
    mask = fm.tokenizer.mask_token
    print(f"bert-base-uncased 的 mask 符号是：{mask!r}")

    text = f"Machine learning is a subfield of {mask} intelligence."
    print(f"用动态 mask 拼句子：{text}")
    print(f"  → {fm(text)[0]['token_str']}（{fm(text)[0]['score']:.3f}）")

    # 演示用错符号的后果
    print("\n如果给 bert 用 <mask>（camembert 的符号）会怎样：")
    try:
        fm("Paris is the <mask> of France.")
        print("  （某些版本能容错，但很多会直接报错）")
    except Exception as e:
        print(f"  ❌ 报错：{type(e).__name__}: {str(e)[:80]}")
    print("👉 结论：永远用 tokenizer.mask_token，别把 [MASK]/<mask> 写死。")


# ==============================================================================
# 第 4 关：Auto 类 vs 专用类
# ==============================================================================
def case_04():
    title(4, "Auto 类 vs 专用类：两种加载方式，结果一致")
    from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                              BertTokenizer, BertForMaskedLM)

    ckpt = "bert-base-uncased"
    # 写法A：专用类（写死 Bert 架构）
    tok_a = BertTokenizer.from_pretrained(ckpt)
    model_a = BertForMaskedLM.from_pretrained(ckpt)
    # 写法B：Auto 类（自动判断架构）★推荐
    tok_b = AutoTokenizer.from_pretrained(ckpt)
    model_b = AutoModelForMaskedLM.from_pretrained(ckpt)

    print("专用类加载的模型类型：", type(model_a).__name__)
    print("Auto 类加载的模型类型：", type(model_b).__name__)
    print("→ 是同一个类：", type(model_a).__name__ == type(model_b).__name__)
    print("👉 结论一致。但 Auto 类换 checkpoint 不用改代码，所以优先用 Auto。")
    print("   （把 ckpt 换成 camembert-base，AutoModelForMaskedLM 自动加载 camembert）")


# ==============================================================================
# 第 5 关：手动 fill-mask（不用 pipeline，看清底层）[结合 Ch1]
# ==============================================================================
def case_05():
    title(5, "手动 fill-mask：分词→模型→找mask位置→softmax→topk")
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    ckpt = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForMaskedLM.from_pretrained(ckpt)
    model.eval()

    text = f"The weather today is very {tok.mask_token}."
    # ① 分词（Ch1）
    inputs = tok(text, return_tensors="pt")
    # ② 找到 [MASK] 在序列里的位置
    mask_pos = (inputs["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=True)[0].item()
    print(f"句子：{text}")
    print(f"[MASK] 在第 {mask_pos} 个 token 位置")
    # ③ 模型前向，拿到该位置对整个词表的打分(logits)
    with torch.no_grad():
        logits = model(**inputs).logits          # [1, 序列长, 词表大小]
    mask_logits = logits[0, mask_pos]             # 只取 mask 位置那一行
    # ④ softmax → 概率，取 top5（Ch1 的 softmax + argmax 思路）
    probs = torch.softmax(mask_logits, dim=-1)
    top = torch.topk(probs, 5)
    print("手动算出的 top5 候选：")
    for score, tid in zip(top.values, top.indices):
        print(f"  {tok.decode([tid]):12s} 概率 {score.item():.3f}")
    print("👉 这就是 pipeline('fill-mask') 内部干的事，和 Ch1 的手动推理一个套路。")


# ==============================================================================
# 第 6 关：综合 —— 用完形填空探测模型学到的“知识”
# ==============================================================================
def case_06():
    title(6, "综合：用 fill-mask 探测预训练模型学到了什么常识")
    from transformers import pipeline

    fm = pipeline("fill-mask", model="bert-base-uncased")
    mask = fm.tokenizer.mask_token

    probes = [
        f"The sun rises in the {mask}.",              # 常识：east
        f"Water is made of hydrogen and {mask}.",     # 常识：oxygen
        f"A doctor works in a {mask}.",               # 常识：hospital
        f"The opposite of hot is {mask}.",            # 常识：cold
    ]
    print("模型没被专门教过这些，但预训练时从海量文本里“悟”到了：")
    for text in probes:
        top = fm(text)[0]
        print(f"  {text:48s} → {top['token_str']}（{top['score']:.2f}）")
    print("\n👉 完形填空不只是填词——它是一扇窗，能看到模型预训练时学到的世界常识。")
    print("   这也解释了为什么预训练模型能当各种下游任务的‘底座’(Ch2 微调的基础)。")


# ------------------------------------------------------------------------------
# 调度器
# ------------------------------------------------------------------------------
CASES = {1: case_01, 2: case_02, 3: case_03, 4: case_04, 5: case_05, 6: case_06}

MENU = """\
用法：
  python3 Chapter3_案例闯关_预训练模型实战.py <关号>   跑单关，如 1 / 3 / 6
  python3 Chapter3_案例闯关_预训练模型实战.py all       跑全部

关卡列表：
  1  fill-mask 初体验     2  多语言对比          3  mask token 的坑
  4  Auto 类 vs 专用类    5  手动 fill-mask(Ch1) 6  综合:探测模型的知识
"""


def run_choice(choice: str):
    choice = choice.strip().lower()
    if choice == "all":
        for n in sorted(CASES):
            CASES[n]()
    else:
        try:
            CASES[int(choice)]()
        except (ValueError, KeyError):
            print(f"没有第 {choice} 关，请输入 1~6 或 all。")


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
