"""
================================================================================
 vLLM 生产实战参考（★本文件不在 Mac 上运行，仅供手敲学习 / 上 GPU 后照着操作）
================================================================================
 ⚠️ 为什么不运行？
    vLLM 只能在 Linux + NVIDIA GPU(CUDA) 上跑，你的 M4 Mac 装不了（会话开头已确认）。
    本文件是「生产环境怎么写 vLLM、怎么部署操作」的参考。等你有 GPU 环境
    （云服务器 / Colab GPU / 公司机器）时，照着敲、照着命令操作即可。
    想在 Mac 上真跑“等价概念”，去隔壁 Chapter 1/优化推理部署_案例闯关.py（用 mlx-lm）。

 先决条件（在 Linux + NVIDIA GPU 机器上）：
    pip install vllm
    # 验证有 GPU：nvidia-smi

 六个生产场景，每个都给「怎么写代码 + 怎么操作部署」：
    场景1  离线批量推理          一次性处理海量 prompt（数据清洗/打标/评测）
    场景2  部署 OpenAI 兼容服务   vllm serve，最主流的生产上线方式
    场景3  高并发调优            引擎参数怎么调（显存/并发/吞吐）
    场景4  结构化输出(JSON)       强制模型输出合法 JSON（数据抽取管线）
    场景5  流式输出              聊天 UI 的打字机效果
    场景6  Docker + 多卡部署      生产运维：容器化 + 张量并行跑大模型

 术语（vLLM 特有）：
    PagedAttention     vLLM 的显存管理黑科技，KV cache 分页，显存利用率接近 100%
    continuous batching 连续批处理，谁生成完谁下车、空位立刻上新请求，GPU 不空转
    tensor parallel    张量并行，把一个大模型切到多张卡上跑
    guided decoding    引导解码，约束输出格式（如强制合法 JSON）
================================================================================
"""

# ==============================================================================
# 场景 1：离线批量推理（offline batch inference）
#   适用：一次性处理海量文本——给数据集打标、批量摘要、离线评测、生成训练数据。
#   特点：不用起服务，脚本跑完即止；vLLM 自动做连续批处理，吞吐极高。
# ==============================================================================
# ---- ① 怎么写 ----
from vllm import LLM, SamplingParams

# LLM(...)：加载模型并初始化推理引擎（相当于 mlx 的 load）
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",  # HuggingFace 上的模型名或本地路径
    gpu_memory_utilization=0.90,        # 允许占用 90% 显存：越高并发越大，太高易 OOM
    max_model_len=4096,                 # 单条最大上下文长度(token)，按显存和需求设
    dtype="auto",                       # 精度：auto/half(fp16)/bfloat16
)

# SamplingParams：生成参数（和 Chapter 1 学的采样参数一一对应）
sampling_params = SamplingParams(
    temperature=0.7,        # 随机性：0=确定，越高越发散
    top_p=0.95,             # 核采样：只从累计概率前 95% 的词里选
    top_k=50,               # 只从概率最高的 50 个词里选（0=不限）
    max_tokens=256,         # 最多生成多少新 token
    presence_penalty=0.0,   # 出现过就惩罚 → 鼓励新话题
    frequency_penalty=0.0,  # 按出现次数惩罚 → 抑制刷屏
    stop=["\n\n", "###"],   # 遇到这些字符串就停
)

# 一次喂一大批 prompt，vLLM 内部自动批处理（这就是它比裸 transformers 快的关键）
prompts = [
    "用一句话介绍杭州。",
    "把这句话翻译成英文：今天天气很好。",
    "给'量子计算'写一句科普。",
    # ……实际生产里这里可能是几万、几十万条
]
outputs = llm.generate(prompts, sampling_params)

# 取结果：每个 output 对应一个 prompt
for out in outputs:
    prompt = out.prompt
    text = out.outputs[0].text          # outputs[0] 是第一个候选（n=1 时就它）
    print(f"输入：{prompt}\n输出：{text}\n")

# 也可以用 chat 接口（自动套聊天模板，推荐用于 instruct 模型）：
messages = [
    {"role": "system", "content": "你是简洁的中文助手。"},
    {"role": "user", "content": "什么是连续批处理？一句话。"},
]
chat_outputs = llm.chat(messages, sampling_params)
print(chat_outputs[0].outputs[0].text)

# ---- ② 怎么操作 ----
#   # 在 Linux + GPU 机器上：
#   pip install vllm
#   python offline_batch.py         # 直接跑本脚本即可，跑完退出
#   # 想省显存跑更大模型，调低 gpu_memory_utilization 或减小 max_model_len


# ==============================================================================
# 场景 2：部署 OpenAI 兼容 API 服务（最主流的生产上线方式）
#   适用：把模型做成一个 HTTP 服务，前端/其他服务用 OpenAI 客户端直接调。
#   优势：客户端代码和调 OpenAI 官方一模一样；vLLM 自动处理高并发。
# ==============================================================================
# ---- ① 怎么操作：一条命令起服务 ----
#   vllm serve Qwen/Qwen2.5-7B-Instruct \
#       --port 8000 \
#       --gpu-memory-utilization 0.90 \
#       --max-model-len 4096 \
#       --api-key sk-my-secret            # 可选：给服务加个鉴权 key
#
#   起来后访问 http://localhost:8000/v1 ，接口与 OpenAI 完全兼容。

# ---- ② 怎么写：客户端调用（Python，用官方 openai 库）----
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",   # 指向你的 vLLM 服务
    api_key="sk-my-secret",                # 与 --api-key 一致；没设就随便填
)
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",      # 与 serve 的模型名一致
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
    temperature=0.7,
    top_p=0.95,
    max_tokens=200,
)
print(resp.choices[0].message.content)

# ---- ③ 怎么操作：也能用 curl 直接测（运维排查常用）----
#   curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer sk-my-secret" \
#     -d '{
#       "model": "Qwen/Qwen2.5-7B-Instruct",
#       "messages": [{"role": "user", "content": "你好"}],
#       "max_tokens": 100
#     }'


# ==============================================================================
# 场景 3：高并发调优（引擎参数怎么调）
#   适用：线上 QPS 高，要榨干 GPU 吞吐、又不能 OOM 崩服务。
#   这些参数决定“能同时扛多少请求”。
# ==============================================================================
# ---- 怎么操作：serve 时加这些参数 ----
#   vllm serve Qwen/Qwen2.5-7B-Instruct \
#       --gpu-memory-utilization 0.95 \      # 显存吃满一点 → 更多并发（留点余量防OOM）
#       --max-num-seqs 256 \                 # 最多同时处理 256 条序列（并发上限）
#       --max-num-batched-tokens 8192 \      # 一个 batch 里所有序列 token 总数上限
#       --max-model-len 4096 \               # 单条上下文上限；调小可省显存换更高并发
#       --enable-chunked-prefill \           # 长 prompt 分块预填充，降低长请求对延迟的冲击
#       --swap-space 4                       # CPU 交换空间(GB)，显存不够时临时换出
#
# 调优直觉：
#   * 想要高吞吐(throughput) → 调大 max-num-seqs / max-num-batched-tokens
#   * 显存不够 / OOM → 调小 max-model-len、gpu-memory-utilization，或加 swap-space
#   * 长文本请求多 → 开 enable-chunked-prefill
#   * 用 nvidia-smi / vLLM 日志观察显存占用和吞吐，逐步调

# ---- 怎么写：在自己的异步应用里内嵌引擎（进阶）----
#   from vllm import AsyncLLMEngine, AsyncEngineArgs
#   engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
#       model="Qwen/Qwen2.5-7B-Instruct",
#       gpu_memory_utilization=0.9,
#       max_num_seqs=256,
#   ))
#   # 然后用 async for 消费 engine.generate(...) 的流式结果
#   # 大多数情况直接用 vllm serve 就够了，不用自己写异步引擎。


# ==============================================================================
# 场景 4：结构化输出 JSON（生产数据抽取管线）
#   适用：从文本抽字段入库、function calling、需要“保证输出是合法 JSON”的场景。
#   vLLM 的 guided decoding 能在解码时约束输出，100% 符合你给的 schema。
# ==============================================================================
# ---- ① 怎么写：离线方式 ----
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "skills"],
}
guided = GuidedDecodingParams(json=json_schema)     # 用 JSON Schema 约束
params = SamplingParams(temperature=0.0, max_tokens=200, guided_decoding=guided)

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
out = llm.generate("提取信息：张三，28岁，会 Python 和 Go。", params)
print(out[0].outputs[0].text)   # 保证是合法 JSON：{"name":"张三","age":28,"skills":[...]}

# ---- ② 怎么写：服务方式（OpenAI 客户端 + extra_body）----
#   resp = client.chat.completions.create(
#       model="Qwen/Qwen2.5-7B-Instruct",
#       messages=[{"role": "user", "content": "提取：李四，30岁，会 Java。"}],
#       extra_body={"guided_json": json_schema},   # vLLM 扩展参数，走 extra_body 传
#   )
# 除了 guided_json，还支持 guided_regex（正则约束）、guided_choice（限定选项）。


# ==============================================================================
# 场景 5：流式输出（聊天 UI 的打字机效果）
#   适用：面向用户的聊天产品，要“秒见首字、逐字蹦”，降低等待焦虑。
# ==============================================================================
# ---- 怎么写：服务 + OpenAI 客户端 stream=True ----
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-my-secret")
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "写一个关于星空的短故事"}],
    max_tokens=300,
    temperature=0.8,
    stream=True,                       # 关键：开启流式
)
for chunk in stream:                   # 一块一块到达
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)   # 实时打印
print()


# ==============================================================================
# 场景 6：Docker 部署 + 多卡张量并行（生产运维）
#   适用：正式上线，用容器统一环境；模型太大单卡放不下时切到多张卡。
# ==============================================================================
# ---- 怎么操作：Docker 一键起服务（官方镜像）----
#   docker run --gpus all \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \   # 挂载模型缓存，避免重复下载
#       -p 8000:8000 \
#       vllm/vllm-openai:latest \
#       --model Qwen/Qwen2.5-7B-Instruct \
#       --gpu-memory-utilization 0.90
#
# ---- 怎么操作：多卡张量并行跑大模型（如 72B）----
#   vllm serve Qwen/Qwen2.5-72B-Instruct \
#       --tensor-parallel-size 4 \       # 把模型切到 4 张 GPU 上（要正好有 4 张卡）
#       --gpu-memory-utilization 0.90 \
#       --max-model-len 8192
#   # tensor-parallel-size = 你有几张卡就写几；单卡放不下的大模型靠它才能跑起来
#
# ---- 生产上线检查清单 ----
#   □ --api-key 加鉴权，别裸奔
#   □ 用 nvidia-smi 确认显存没爆、GPU 利用率健康
#   □ 压测（如 vllm 自带 benchmark 或 locust）确定 max-num-seqs 等参数
#   □ 前面挂个反向代理(nginx)做限流/HTTPS
#   □ 监控吞吐、延迟、错误率；日志留存


# ==============================================================================
# 附：Mac 上想真跑怎么办？
# ==============================================================================
# 本文件在你 Mac 上跑不了（无 CUDA）。两条路：
#   1) 学概念 + 动手 → Chapter 1/优化推理部署_案例闯关.py（用 mlx-lm，Mac 能跑，
#      概念与 vLLM 一一对应：采样参数、流式、批处理、OpenAI 服务）
#   2) 真跑 vLLM → 上云 GPU：
#      - Google Colab（免费 T4）：新建 notebook，运行时选 GPU，!pip install vllm
#      - Runpod / Lambda / AWS 等租 GPU 服务器，按本文件的命令操作
# 你在 mlx 上练熟的 OpenAI 客户端代码，切到 vLLM 服务一行都不用改。
