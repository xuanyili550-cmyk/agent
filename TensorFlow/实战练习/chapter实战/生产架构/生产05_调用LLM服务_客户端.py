"""
================================================================================
 生产05 · 调用已部署的 LLM 服务：OpenAI 兼容客户端 / HF InferenceClient / 流式 / 结构化输出
================================================================================
 ⚠ 需要一个【已部署的 LLM 服务】(vLLM/TGI/云端 API)，本机没有服务不跑；真实生产客户端代码。
   服务端怎么起见 生产01(vLLM+FastAPI 网关)；本文件是【客户端侧】怎么调它。
   本地能跑的小生成(采样/流式)见 ../分章项目/Ch1_进阶_生成与采样.py(用小模型 distilgpt2)。

 三种主流调用方式(都是 GPU 服务的标准客户端写法)：
   ① OpenAI 兼容客户端：vLLM/TGI 都暴露 /v1/chat/completions，用官方 openai SDK 直连(不锁厂商)。
   ② HF InferenceClient：连 HF Inference Endpoints / TGI / 各推理提供商，chat_completion/text_generation。
   ③ 结构化输出：temperature=0 + JSON 约束 + json.loads 兜底，把 LLM 当“可编程组件”。
 依赖：pip install openai huggingface_hub
================================================================================
"""
import os
import json

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")   # vLLM OpenAI 兼容端点
API_KEY = os.getenv("LLM_API_KEY", "EMPTY")                         # vLLM 本地不校验，云端填真 key
MODEL = os.getenv("LLM_MODEL", "qwen")


# ==============================================================================
# ① OpenAI 兼容客户端：一次性返回
# ==============================================================================
def chat_openai(question):
    """用官方 openai SDK 连 vLLM 的 OpenAI 兼容端点(生产最常用；换云端只改 base_url/key)。"""
    from openai import OpenAI                                      # 惰性导入
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120, max_retries=2)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "你是简洁的助手。"},
                  {"role": "user", "content": question}],
        temperature=0.7, max_tokens=512,
    )
    return resp.choices[0].message.content


# ==============================================================================
# ② OpenAI 兼容客户端：流式(打字机效果)
# ==============================================================================
def chat_openai_stream(question):
    """stream=True：服务端 SSE 逐块推送，客户端边收边显示(前端体验关键)。"""
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)
    stream = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": question}],
        temperature=0.7, max_tokens=512, stream=True,
    )
    pieces = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        pieces.append(delta)
        print(delta, end="", flush=True)                          # 实时打印增量
    return "".join(pieces)


# ==============================================================================
# ③ HF InferenceClient：连 TGI / HF Endpoints / 各推理提供商
# ==============================================================================
def chat_inference_client(question):
    """huggingface_hub.InferenceClient：统一接口连 HF Endpoints/TGI/第三方 provider。"""
    from huggingface_hub import InferenceClient                   # 惰性导入
    client = InferenceClient(base_url=BASE_URL, token=os.getenv("HF_TOKEN"))
    out = client.chat_completion(
        messages=[{"role": "user", "content": question}],
        max_tokens=512, temperature=0.7,
    )
    return out.choices[0].message.content


def generate_inference_client(prompt):
    """text_generation 原语(非 chat)：直接续写，可拿 token 级细节。"""
    from huggingface_hub import InferenceClient
    client = InferenceClient(base_url=BASE_URL, token=os.getenv("HF_TOKEN"))
    return client.text_generation(prompt, max_new_tokens=256, temperature=0.7,
                                  repetition_penalty=1.1, stop=["\n\n"])


# ==============================================================================
# ④ 结构化输出：把 LLM 当“可编程组件”(生产抽取/分类必用)
# ==============================================================================
def extract_json(text):
    """temperature=0(要确定性) + 强约束提示 + json.loads 兜底。生产做信息抽取/分类的标准姿势。"""
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "只输出 JSON，不要多余文字。"},
            {"role": "user", "content":
             f"从下面文本抽取 {{name, company, sentiment}} 三个字段，输出 JSON：\n{text}"},
        ],
        temperature=0,                                            # ★确定性输出，别采样
        response_format={"type": "json_object"},                  # ★vLLM/OpenAI 支持的 JSON 模式
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)                                    # 兜底：解析失败要能降级/重试
    except json.JSONDecodeError:
        return {"_raw": raw, "_error": "not valid json"}


NOTES = """
 生产要点：
  · 服务端换了(vLLM→云端 OpenAI/通义/DeepSeek)，客户端只改 base_url + api_key，代码不动(OpenAI 兼容的价值)。
  · 健壮性：timeout + max_retries + 指数退避；上游 5xx/限流要有降级(排队/切备用模型)。
  · 流式：SSE 透传给前端；注意 delta.content 可能为 None(要 or "")。
  · 结构化：temperature=0 + response_format/guided decoding；仍要 json.loads 兜底 + 重试。
  · 成本/延迟：缓存相同请求(Redis)、简单请求路由到小模型、批量请求合并。
  · 中国云同理：阿里 通义(DashScope 兼容 OpenAI)、DeepSeek、火山方舟、硅基流动都给 OpenAI 兼容端点。
"""

if __name__ == "__main__":
    print(__doc__)
    print("=== 生产要点 ===", NOTES)
    print(">>> 真实客户端代码，本机没有 LLM 服务不跑。真跑步骤：")
    print("    1) 先按 生产01 在 GPU 机起 vLLM(OpenAI 兼容，端口 8001)")
    print("    2) export LLM_BASE_URL=http://<gpu-host>:8001/v1  LLM_MODEL=qwen")
    print("    3) python3 -c 'import 生产05_调用LLM服务_客户端 as m; print(m.chat_openai(\"你好\"))'")
    print("    本地小生成(distilgpt2)见 ../分章项目/Ch1_进阶_生成与采样.py")
