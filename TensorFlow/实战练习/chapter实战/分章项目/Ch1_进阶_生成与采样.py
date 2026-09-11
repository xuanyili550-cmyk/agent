"""
================================================================================
 分章项目 · Ch1 进阶 · 文本生成的解码策略与采样参数（推理部署的核心旋钮）
================================================================================
 “优化推理部署”里最常被问、也最实用的一块：同一个模型，解码策略/采样参数不同，
 生成质量天差地别。本文件用真模型(distilgpt2)把这些旋钮一次讲清、可运行：
   ① 三种解码策略：贪婪(greedy) vs 采样(sampling) vs 束搜索(beam)——各自适合什么。
   ② 采样参数：temperature / top_k / top_p（核采样）——控制“随机性/多样性”。
   ③ 抑制重复：repetition_penalty / no_repeat_ngram_size——治“复读机”。
   ④ GenerationConfig：把一堆生成参数打包成一个对象(生产里统一管理)。
   ⑤ 流式输出：TextIteratorStreamer + 线程——边生成边吐字(打字机效果，体验关键)。
   完整章节材料见 ../../../Chapter 1/优化推理部署_学习笔记.py。
 跑：python3 Ch1_进阶_生成与采样.py
================================================================================
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "distilgpt2"
tok = AutoTokenizer.from_pretrained(CKPT)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(CKPT).to(DEV).eval()
PROMPT = "In the future, artificial intelligence will"


def gen(**kw):
    enc = tok(PROMPT, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")


# ==============================================================================
# ① 三种解码策略
# ==============================================================================
def decoding_strategies():
    # ① 解码策略：贪婪 vs 采样 vs 束搜索
    torch.manual_seed(0)
    # print("  [贪婪 greedy]   每步取概率最高的词(确定、稳，但易重复呆板)：")
    print("    →", gen(max_new_tokens=30, do_sample=False)[:90])
    # print("  [采样 sampling] 按概率分布抽样(多样、有创造力，但可能跑题)：")
    print("    →", gen(max_new_tokens=30, do_sample=True, temperature=0.9, top_p=0.95)[:90])
    # print("  [束搜索 beam]   同时保留 k 条候选选整体最优(适合翻译/摘要这种有‘正确答案’的)：")
    print("    →", gen(max_new_tokens=30, num_beams=4, do_sample=False, early_stopping=True)[:90])


# ==============================================================================
# ② 采样参数：temperature / top_k / top_p
# ==============================================================================
def sampling_params():
    # ② 采样参数：temperature / top_k / top_p(核采样)
    torch.manual_seed(0)
    print("  temperature 低(0.3)=更保守聚焦：", gen(max_new_tokens=25, do_sample=True, temperature=0.3)[:80])
    torch.manual_seed(0)
    print("  temperature 高(1.3)=更发散冒险：", gen(max_new_tokens=25, do_sample=True, temperature=1.3)[:80])
    torch.manual_seed(0)
    print("  top_k=10  只在概率最高的10个词里抽：", gen(max_new_tokens=25, do_sample=True, top_k=10)[:80])
    torch.manual_seed(0)
    print("  top_p=0.9 只在累计概率90%的核里抽：", gen(max_new_tokens=25, do_sample=True, top_p=0.9)[:80])
    # 记忆：temperature 调整体随机性；top_k 固定候选数；top_p 按累计概率动态截断(最常用)。


# ==============================================================================
# ③ 抑制重复
# ==============================================================================
def anti_repeat():
    # ③ 抑制重复：repetition_penalty / no_repeat_ngram_size
    print("  不加抑制(易复读)     :", gen(max_new_tokens=40, do_sample=False)[:90])
    print("  repetition_penalty=1.3:", gen(max_new_tokens=40, do_sample=False, repetition_penalty=1.3)[:90])
    print("  no_repeat_ngram_size=3:", gen(max_new_tokens=40, do_sample=False, no_repeat_ngram_size=3)[:90])
    # print("  前者对‘出现过的词’降概率；后者硬禁止任何 3-gram 重复出现。")


# ==============================================================================
# ④ GenerationConfig：把生成参数打包成对象
# ==============================================================================
def generation_config():
    # ④ GenerationConfig(统一管理生成参数)
    cfg = GenerationConfig(max_new_tokens=30, do_sample=True, temperature=0.8,
                           top_p=0.9, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
    enc = tok(PROMPT, return_tensors="pt").to(DEV)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(**enc, generation_config=cfg)
    print("  用 cfg 一次传入：", tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()[:90])
    # print("  生产里把生成参数存成 GenerationConfig(可随模型保存/版本化)，别到处散落魔法数字。")


# ==============================================================================
# ⑤ 流式输出：TextIteratorStreamer + 线程
# ==============================================================================
def streaming():
    # ⑤ 流式输出(TextIteratorStreamer，边生成边吐字)
    from transformers import TextIteratorStreamer
    from threading import Thread
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    enc = tok(PROMPT, return_tensors="pt").to(DEV)
    # generate 放到子线程跑，主线程从 streamer 迭代拿增量文本(否则 generate 会阻塞)
    Thread(target=model.generate, kwargs=dict(
        **enc, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id, streamer=streamer)).start()
    # print("  流式：", end="", flush=True)
    pieces = 0
    for text in streamer:                                      # 每来一小段就立刻显示
        print(text, end="", flush=True); pieces += 1
    print(f"\n  (共 {pieces} 次增量推送——这就是 ChatGPT 那种打字机效果的底层，体验远好于一次性等完。)")


if __name__ == "__main__":
    # 设备/模型见文件顶部；下面五个演示各自打印真实生成结果
    decoding_strategies()
    sampling_params()
    anti_repeat()
    generation_config()
    streaming()
    print("\n✅ Ch1 进阶跑通：解码策略 → 采样参数 → 抑制重复 → GenerationConfig → 流式输出。")
    # 面试：greedy/beam/sampling 各适合什么? top_k vs top_p? 怎么治复读机? 流式怎么实现? (见 ../面试高频题库.py)
