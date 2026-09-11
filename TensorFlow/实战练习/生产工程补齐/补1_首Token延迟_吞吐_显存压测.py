"""
================================================================================
 生产工程补齐 · 补1 · 推理指标压测：首Token延迟(TTFT) + 吞吐(tokens/s) + 显存
================================================================================
 补上「显存/首Token延迟/吞吐 三大核心指标」这一块——面试和上线都要会量这三个数。
   ① TTFT(Time To First Token)：从请求到吐出第 1 个 token 的耗时——决定「用户多久看到反应」。
   ② 吞吐(decode tokens/s)：首 token 之后的稳定生成速度——决定「一次回答多久生成完」。
   ③ 批处理吞吐：把多条请求一起算(batching)，总 tokens/s 远高于逐条——GPU 利用率的关键。
   ④ 显存峰值：torch 查当前/峰值占用——判断卡够不够、能开多大 batch。
 为什么这么测：TTFT 靠「流式 streamer 记第一片的时间」；吞吐靠「token 数 / 时间」；
   批处理提速靠「一批 vs 逐条」对比。生产里 TTFT 和吞吐是两个独立优化目标(前者靠预填充/KV Cache，
   后者靠连续批处理)。
 依赖：transformers + torch（本机已装）。用小模型 distilgpt2，CPU/MPS 都能跑。
 跑：python3 补1_首Token延迟_吞吐_显存压测.py     （注：按用户要求本文件未在本机执行，仅作真实可跑代码）
================================================================================
"""
import time
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "distilgpt2"
tok = AutoTokenizer.from_pretrained(CKPT)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(CKPT).to(DEV).eval()
PROMPT = "In the future, artificial intelligence will"


def mem_mb():
    """当前进程占用的模型显存(MB)，按设备取对应 API。"""
    if DEV == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    if DEV == "mps":
        return torch.mps.current_allocated_memory() / 1024**2
    return 0.0  # CPU 无独立显存概念


# ==============================================================================
# ① 单条：TTFT + 稳定吞吐（靠流式 streamer 记录「第一片」的时刻）
# ==============================================================================
def bench_single(max_new_tokens=64):
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    enc = tok(PROMPT, return_tensors="pt").to(DEV)
    t0 = time.perf_counter()
    Thread(target=model.generate, kwargs=dict(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.eos_token_id, streamer=streamer)).start()

    first_t = None
    pieces = 0
    for _ in streamer:                       # 每来一片增量
        if first_t is None:
            first_t = time.perf_counter()    # 记下第一片 = TTFT 终点
        pieces += 1
    end = time.perf_counter()

    ttft = (first_t - t0) * 1000             # ms
    decode_tps = pieces / (end - first_t) if end > first_t else 0.0
    # 说明：streamer 的「片」约等于 token 数(BPE 下大体成立)，作近似吞吐。
    return {"TTFT_ms": round(ttft, 1), "decode_tokens_per_s": round(decode_tps, 1), "gen_tokens": pieces}


# ==============================================================================
# ② 批处理吞吐：一批 N 条一起算 vs 逐条，对比总 tokens/s（GPU 利用率的关键）
# ==============================================================================
def bench_batch(batch=8, max_new_tokens=32):
    prompts = [PROMPT] * batch
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)

    # 批处理：一次 generate 出 batch 条
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    dt_batch = time.perf_counter() - t0
    new_tokens = batch * max_new_tokens
    batch_tps = new_tokens / dt_batch

    # 逐条：循环 batch 次
    t0 = time.perf_counter()
    with torch.no_grad():
        for p in prompts:
            e = tok(p, return_tensors="pt").to(DEV)
            model.generate(**e, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    dt_seq = time.perf_counter() - t0
    seq_tps = new_tokens / dt_seq

    return {"batch_tokens_per_s": round(batch_tps, 1), "sequential_tokens_per_s": round(seq_tps, 1),
            "speedup": round(batch_tps / seq_tps, 2) if seq_tps else None}


def main():
    single = bench_single()
    batch = bench_batch()
    # single -> {'TTFT_ms': ~, 'decode_tokens_per_s': ~, 'gen_tokens': 64}
    # batch  -> {'batch_tokens_per_s': ~, 'sequential_tokens_per_s': ~, 'speedup': >1}
    # 显存峰值 mem_mb() -> MPS/CUDA 上为模型占用(MB)，CPU 为 0
    assert single["TTFT_ms"] >= 0 and single["gen_tokens"] > 0
    print(f"✅ 补1 跑通：TTFT={single['TTFT_ms']}ms  单条吞吐={single['decode_tokens_per_s']}tok/s  "
          f"批处理提速×{batch['speedup']}  显存={mem_mb():.0f}MB")
    # 面试：Q TTFT 和吞吐为什么是两个独立指标? A TTFT 靠预填充/KV Cache 优化，吞吐靠连续批处理;
    #      Q 批处理为什么提吞吐? A 一次算多条 GPU 不空转; Q 怎么量显存峰值? A torch.*_memory_allocated。


if __name__ == "__main__":
    main()
