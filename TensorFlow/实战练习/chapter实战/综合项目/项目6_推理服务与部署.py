"""
================================================================================
 综合项目6 · 推理服务与优化部署（整合 Ch1 + Ch3 + Ch9）
================================================================================
 把模型高效地“服务化”出去。本地演示“批处理提速”这个最核心的推理优化，再给出上云部署全套。
 整合哪些章、为什么用：
   [Ch1 推理优化] 批处理(batching)：把多条请求一起算，GPU 利用率高、吞吐大——本文件真跑对比。
   [Ch3 vLLM]     生产用 vLLM(PagedAttention + Continuous Batching)，吞吐再高一个量级(见 ../生产架构/生产01)。
   [Ch9 服务化]   FastAPI 把模型包成 HTTP API；Docker/K8s 上线(见 ../生产架构/部署)。

 本地跑：python3 项目6_推理服务与部署.py     # 真跑“逐条 vs 批处理”吞吐对比
================================================================================
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def main():
    dev = pick_device()
    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(dev).eval()
    texts = ["This movie is great!", "Terrible service.", "I love it.",
             "Not good at all.", "Amazing product.", "Waste of money."] * 16   # 96 条

    # ---- [Ch1] 逐条推理(每次一条，GPU 频繁启停、利用率低) ----
    t0 = time.time()
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True).to(dev)
        with torch.no_grad():
            _ = model(**enc).logits
    seq_time = time.time() - t0

    # ---- [Ch1] 批处理推理(一次一批，padding 对齐，GPU 一次算完) ----
    t0 = time.time()
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            _ = model(**enc).logits
    batch_time = time.time() - t0

    print(f">>> 设备={dev}  {len(texts)} 条文本推理：")
    print(f"    逐条(batch=1) : {seq_time*1000:6.0f} ms  ({len(texts)/seq_time:.0f} 条/秒)")
    print(f"    批处理(batch=32): {batch_time*1000:6.0f} ms  ({len(texts)/batch_time:.0f} 条/秒)")
    print(f"    ⇒ 批处理提速 {seq_time/batch_time:.1f}× （这就是为什么生产必须批处理）")
    print("\n✅ 推理优化跑通：批处理是最基础也最有效的提速手段;LLM 生成用 vLLM 连续批处理更强。")


# ==============================================================================
# 生产：把模型服务化 + 上云（代码见 ../生产架构/，这里给要点）
# ==============================================================================
# 一、LLM 生成服务(不是分类)：别自己写生成循环，用 vLLM(OpenAI 兼容)：
#     python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8001
#     → PagedAttention(KV Cache 分页) + Continuous Batching(连续批处理),吞吐远超原生。
# 二、FastAPI 网关(鉴权/限流/路由/日志) → vLLM。见 ../生产架构/生产01。
# 三、优化手段(显存/延迟/成本)：
#     · 量化 INT8/INT4(AWQ/GPTQ/FP8)：权重变小,更小卡跑大模型。
#     · 张量并行 tensor-parallel：一层切多卡,放下超大模型。
#     · KV cache / prefix caching：相同前缀复用,省重复计算。
#     · 请求缓存(Redis)、模型路由(简单请求走小模型)、批处理。
# 四、部署：Docker 打包 → docker-compose(单机) / K8s(集群,探针+HPA自动扩缩)。见 ../生产架构/部署。
# 五、监控：延迟 p50/p95/p99、吞吐 QPS、GPU 利用率/显存、错误率、每请求 token 数/成本。

# ==============================================================================
# 面试题
# ==============================================================================
# Q: vLLM 为什么快？两大核心？(见面试题库 七.推理部署)
# Q: 批处理为什么能提速？动态 vs 连续批处理区别？
# A: 一次算多条,GPU 不空转、吞吐高。静态批处理要等凑满一批;连续批处理(vLLM)是每步动态把不同请求的
#    token 拼一起,新请求随到随进、完成的随时退出,GPU 几乎不空转。
# Q: 模型放不下单卡怎么部署？(见面试题库 七)
# Q: 训练和推理为什么分离？(见面试题库 七)
# Q: 怎么降低推理成本？(见面试题库 八)

if __name__ == "__main__":
    main()
