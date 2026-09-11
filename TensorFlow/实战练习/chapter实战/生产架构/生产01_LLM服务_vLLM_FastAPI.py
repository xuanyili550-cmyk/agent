"""
================================================================================
 生产01 · LLM 推理服务：vLLM(OpenAI 兼容) + FastAPI 网关
================================================================================
 ⚠ 需要 GPU 服务器，本机跑不了；这是“真上线怎么搭”的架构参考代码。
   本地能跑的小版本(mlx-lm 采样等)见 Chapter 1/优化推理部署_案例闯关.py。

 架构：
   客户端 ──HTTP──▶ FastAPI 网关(鉴权/限流/日志) ──▶ vLLM 服务(OpenAI 兼容) ──▶ GPU
 为什么这么分：
   · 为什么用 vLLM：它有 PagedAttention(KV Cache 分页,显存利用率高) + Continuous Batching
     (连续批处理,把不同请求的 token 动态拼一起算),吞吐比原生 transformers 高一个量级。
   · 为什么要 FastAPI 网关而不直连 vLLM：网关做鉴权/限流/审计/多后端路由/统一日志,
     把这些“横切关注点”和推理引擎解耦,vLLM 只管高效生成。
   · 为什么 OpenAI 兼容：vLLM 暴露 /v1/chat/completions,前端/SDK 无缝切换(不锁定厂商)。

 —— 一、起 vLLM 服务(在 GPU 机器上，命令行) ——
   pip install vllm
   python -m vllm.entrypoints.openai.api_server \\
       --model Qwen/Qwen2.5-7B-Instruct \\
       --served-model-name qwen \\
       --gpu-memory-utilization 0.90 \\     # 用满显存(留 10% 余量)
       --max-model-len 8192 \\
       --max-num-seqs 256 \\                # 最多并发多少条序列
       --tensor-parallel-size 1 \\          # 多卡时=卡数(张量并行)
       --port 8001
   → 起来后就是一个 OpenAI 兼容服务：POST http://gpu-host:8001/v1/chat/completions
   (中国云上同理；也可用 LMDeploy: lmdeploy serve api_server Qwen/... )
================================================================================
"""
import os
import time
import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001/v1/chat/completions")
API_KEYS = set(os.getenv("API_KEYS", "demo-key-123").split(","))   # 生产用密钥管理/网关鉴权
MODEL = os.getenv("MODEL", "qwen")

app = FastAPI(title="LLM 推理网关")

# 极简内存限流(生产用 Redis 做分布式限流：每 key 每分钟 N 次)
_rate = {}
RATE_LIMIT_PER_MIN = 60


class ChatRequest(BaseModel):
    messages: list          # [{"role","content"}]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False


def check_auth(authorization: str):
    # 生产：从 Authorization: Bearer <key> 取密钥并校验(可接 API 网关/JWT)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "缺少 Bearer token")
    key = authorization.split(" ", 1)[1]
    if key not in API_KEYS:
        raise HTTPException(401, "无效的 API key")
    return key


def check_rate_limit(key):
    now = int(time.time() // 60)
    bucket = _rate.setdefault((key, now), 0)
    if bucket >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "超出限流(每分钟上限)")
    _rate[(key, now)] += 1


@app.get("/health")
def health():
    return {"status": "ok"}          # K8s 存活/就绪探针打这个


@app.post("/v1/chat")
async def chat(req: ChatRequest, authorization: str = Header(None)):
    key = check_auth(authorization)
    check_rate_limit(key)
    # 网关把请求转发给 vLLM(OpenAI 兼容)，可在此加：多后端路由、缓存、审计日志、脱敏
    payload = {"model": MODEL, "messages": req.messages,
               "temperature": req.temperature, "max_tokens": req.max_tokens,
               "stream": req.stream}
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(VLLM_URL, json=payload)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"上游 vLLM 出错: {e}")
    data = r.json()
    return {"reply": data["choices"][0]["message"]["content"],
            "usage": data.get("usage")}


# —— 二、启动网关(在能连到 vLLM 的机器上) ——
#   uvicorn 生产01_LLM服务_vLLM_FastAPI:app --host 0.0.0.0 --port 8000 --workers 4
#   调用：curl -X POST http://localhost:8000/v1/chat \
#         -H "Authorization: Bearer demo-key-123" \
#         -d '{"messages":[{"role":"user","content":"你好"}]}'
#
# —— 三、生产要点 ——
#   · 流式：req.stream=True 时用 StreamingResponse 把 vLLM 的 SSE 透传给前端(体验好)。
#   · 扩展：vLLM 服务和网关都无状态 → K8s 里各自多副本,前面挂负载均衡;GPU 紧张就加 vLLM 副本。
#   · 缓存：相同 prompt 结果可缓存(Redis),省 GPU。
#   · 监控：记录每次请求的延迟/token 数/错误,打到 Prometheus;p99 延迟和 GPU 利用率是关键指标。
#   · 成本：按并发/吞吐选卡和 tensor-parallel-size;量化(AWQ/GPTQ/FP8)可在更小卡上跑大模型。

if __name__ == "__main__":
    print(__doc__)
    print(">>> 这是架构参考。真跑：先在 GPU 机起 vLLM，再 uvicorn 起本网关(见文件内命令)。")
