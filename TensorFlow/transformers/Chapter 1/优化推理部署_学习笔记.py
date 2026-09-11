"""
================================================================================
 优化推理部署 (Optimized Inference Deployment) —— 系统学习笔记
================================================================================
 配套 HuggingFace 课程「Chapter 1 / 优化推理部署」章节。
 本文件是「阅读 + 手敲」用的参考资料，结构如下：

   第 0 部分  三大框架是什么、怎么选、你的 Mac 能跑哪个
   第 1 部分  采样参数详解（temperature / top_p / top_k / 各种 penalty ...）
   第 2 部分  部署与内存参数详解（gpu_memory_utilization / block_size ...）
   第 3 部分  四种客户端调用方式（InferenceClient / OpenAI / llama.cpp / vLLM）
   第 4 部分  四个生产案例（可直接对照手敲）

 术语速记：
   token（词元）  模型处理的最小单位，一个中文字 ≈ 1~2 token，一个英文词 ≈ 1 token
   logits        模型最后一层输出的“原始分数”，未归一化；经 softmax 才变成概率
   推理 (inference) 用训练好的模型做预测/生成，区别于训练 (training)
   吞吐 (throughput) 单位时间处理的 token 数，衡量“服务能扛多少量”
   延迟 (latency)   单条请求从发出到拿到结果的耗时，衡量“用户等多久”
================================================================================
"""

# ==============================================================================
# 第 0 部分：三大框架是什么、怎么选、你的 Mac 能跑哪个
# ==============================================================================
#
# 课程讲了三套“把模型部署成服务”的框架，定位不同：
#
#  ┌────────────┬──────────────────────────┬──────────────────┬─────────────┐
#  │ 框架        │ 定位                      │ 硬件要求          │ 你的 M4 Mac │
#  ├────────────┼──────────────────────────┼──────────────────┼─────────────┤
#  │ vLLM       │ 高并发生产级推理引擎       │ Linux + NVIDIA   │ ✗ 装不了     │
#  │            │ 核心：PagedAttention +    │ GPU (CUDA)       │ (无 CUDA)   │
#  │            │ Continuous Batching       │                  │             │
#  ├────────────┼──────────────────────────┼──────────────────┼─────────────┤
#  │ TGI        │ HuggingFace 官方生产方案   │ Linux + GPU      │ ✗ 用 Docker │
#  │ (Text-Gen- │ 通常用 Docker 起服务       │ (Docker)         │  也需 GPU   │
#  │  Inference)│                           │                  │             │
#  ├────────────┼──────────────────────────┼──────────────────┼─────────────┤
#  │ llama.cpp  │ 轻量、跨平台、量化推理     │ CPU / 任意 GPU    │ ✓ 可跑      │
#  │            │ 用 GGUF 量化模型           │ (含 Apple Metal) │  (含 Metal) │
#  └────────────┴──────────────────────────┴──────────────────┴─────────────┘
#
# 关键结论：
#   * vLLM / TGI 是 Linux + NVIDIA GPU 的东西。学它们的“概念和 API”很有用，
#     但要真跑，得上云 GPU（Colab / Runpod / AWS）。本机装不了不是配置问题。
#   * 你的 M4 Mac 上能真跑的等价物：
#       - llama.cpp / llama-cpp-python   （课程里就有）
#       - mlx-lm  （Apple 官方，已在你机器上验证过：167 tok/s）
#     它们都能起一个 “OpenAI 兼容” 的本地服务，API 跟 vLLM 几乎一样。
#
# 为什么这些框架比“裸 transformers”快？三个核心优化，务必理解：
#
#   1) Continuous Batching（连续批处理）
#      裸 transformers：一批请求必须等最慢的那条生成完才能返回，GPU 大量空转。
#      连续批处理：谁生成完谁立刻“下车”，空出的槽位马上让新请求“上车”，
#      GPU 始终满载 → 吞吐量成倍提升。这是生产服务化的第一性能来源。
#
#   2) PagedAttention（分页注意力，vLLM 首创）
#      KV Cache（注意力的键值缓存）是显存大户。传统做法给每条序列预留一整块
#      连续显存，浪费严重。PagedAttention 借鉴操作系统“虚拟内存分页”，把
#      KV Cache 切成固定大小的 block（见 block_size 参数），按需分配，
#      显存利用率接近 100% → 同样显存能塞下更多并发请求。
#
#   3) Quantization（量化）
#      把权重从 FP16(16bit) 压到 INT8/INT4，显存和带宽需求大幅下降，
#      精度损失很小。llama.cpp 的 GGUF（如 Q4_K_M = 4bit）就是干这个的。


# ==============================================================================
# 第 1 部分：采样参数详解（决定“生成出来的文字长什么样”）
# ==============================================================================
#
# 生成是逐 token 进行的：每一步，模型对词表里每个 token 打分(logits)，
# 下面这些参数控制“如何从这些分数里挑下一个 token”。理解它们 = 掌握生成质量。
#
# ------------------------------------------------------------------------------
# 【1】temperature  温度  —— 控制随机性 / 创造性
# ------------------------------------------------------------------------------
#   取值：0.0 ~ 2.0，默认常见 0.7~1.0
#   原理：把 logits 除以 temperature 再做 softmax。
#     - 越低(→0)：分布越“尖”，几乎总选概率最高的词 → 确定、保守、可复现
#     - 越高(→2)：分布越“平”，低概率词也有机会 → 发散、有创意、也更容易胡说
#   经验：
#     - 事实问答 / 代码 / 结构化抽取：0.0 ~ 0.3
#     - 常规对话：0.7
#     - 写作 / 头脑风暴：0.8 ~ 1.2
#
# ------------------------------------------------------------------------------
# 【2】top_p  核采样 (nucleus sampling)  —— 按“累计概率”截断候选
# ------------------------------------------------------------------------------
#   取值：0.0 ~ 1.0，默认常见 0.9~1.0
#   原理：把候选 token 按概率从高到低累加，只保留累计概率达到 p 的那批，
#         其余全部丢弃，再在保留的里面采样。
#   例：top_p=0.95 → 只从“最有可能的、加起来占 95% 概率”的词里选。
#   特点：候选集大小是“动态的”——模型很确定时候选很少，不确定时候选很多。
#
# ------------------------------------------------------------------------------
# 【3】top_k  —— 按“数量”截断候选
# ------------------------------------------------------------------------------
#   取值：正整数，如 50；设 0 表示不限制
#   原理：每步只保留概率最高的 k 个 token，其余丢弃，再采样。
#   与 top_p 区别：top_k 是固定数量，top_p 是固定概率质量。二者常一起用。
#
# ------------------------------------------------------------------------------
# 【4】max_tokens / max_new_tokens  —— 最多生成多少个新 token
# ------------------------------------------------------------------------------
#   注意命名差异（很容易踩坑）：
#     - vLLM SamplingParams / OpenAI 客户端：max_tokens
#     - HF InferenceClient / TGI 的 text_generation：max_new_tokens
#     - llama.cpp Python (Llama)：max_tokens
#   含义都一样：生成的新内容上限，不含输入 prompt。防止无限生成、控制成本。
#
# ------------------------------------------------------------------------------
# 【5】min_tokens / min_new_tokens  —— 最少生成多少（防止太短就停）
# ------------------------------------------------------------------------------
#   在达到这个数量前，模型即使想输出结束符(EOS)也不许停。
#   用于强制“至少写这么长”。
#
# ------------------------------------------------------------------------------
# 【6】重复控制三兄弟  —— 防止模型车轱辘话来回说
# ------------------------------------------------------------------------------
#   repetition_penalty  重复惩罚（TGI / llama.cpp，llama.cpp 里叫 repeat_penalty）
#       取值 >1.0 生效，如 1.1；对“已出现过”的 token 的 logits 做除法惩罚，
#       出现过就更难再被选中。1.0=不惩罚，越大越强（>1.3 容易伤语义）。
#
#   frequency_penalty  频率惩罚（OpenAI / vLLM）
#       取值 -2.0~2.0，如 0.5；按 token“出现的次数”线性惩罚，
#       出现越多次，越被压制 → 抑制高频词刷屏。
#
#   presence_penalty   存在惩罚（OpenAI / vLLM）
#       取值 -2.0~2.0，如 0.5；只要 token“出现过”就惩罚（不看次数），
#       → 鼓励模型引入新话题/新词，增加多样性。
#
#   no_repeat_ngram_size  n-gram 防重复（TGI）
#       如 3 → 禁止任何长度为 3 的词组(trigram)重复出现，硬性去重。
#
#   → 记法：repetition_penalty 是“老框架”的做法；OpenAI 生态用
#           frequency_penalty + presence_penalty 这一对。别混用。
#
# ------------------------------------------------------------------------------
# 【7】stop / stop_sequences  —— 遇到这些字符串就停止生成
# ------------------------------------------------------------------------------
#   命名差异：
#     - OpenAI / vLLM / llama.cpp：stop=["\n\n", "###"]
#     - TGI / InferenceClient：stop_sequences=["\n\n", "###"]
#   典型用途：对话模板里用 stop=["<|im_end|>"] 让模型说完一轮就停，
#             不要把下一轮的角色标记也脑补出来。
#
# ------------------------------------------------------------------------------
# 【8】其它常见开关
# ------------------------------------------------------------------------------
#   do_sample=True/False   （TGI）True=按上面参数随机采样；False=贪心，永远选最高分
#   details=True           （InferenceClient）返回额外信息（每个 token 的概率等）
#   stream=True            流式输出：边生成边一段段返回，做“打字机”效果（案例2）
#   ignore_eos=False       （vLLM）True=忽略结束符，强行生成到 max_tokens
#   skip_special_tokens=True（vLLM）输出里去掉 <|im_end|> 这类特殊标记
#   seed                   随机种子，固定它 + temperature 一起可让结果可复现
#
# ------------------------------------------------------------------------------
# 一句话搭配口诀：
#   temperature 定“敢不敢乱来”，top_p/top_k 定“从多大池子里选”，
#   penalty 系列定“别啰嗦”，max/min/stop 定“写多长、何时收”。
# ==============================================================================


# ==============================================================================
# 第 2 部分：部署与内存参数详解（决定“服务能扛多大量”）
# ==============================================================================
#
# 这些不是控制“文字长什么样”，而是控制“引擎怎么用显存/CPU、并发多高”。
#
# ---- vLLM：LLM(...) / AsyncEngineArgs(...) ----
#   model                    模型名或本地路径
#   gpu_memory_utilization   0~1，如 0.85 = 允许 vLLM 占用 85% 显存。
#                            越高能塞越多并发，但太高易 OOM（显存溢出崩溃）。
#   max_num_batched_tokens   一个 batch 里所有序列 token 数上限，如 8192。
#                            调大 → 吞吐高、显存压力大。
#   max_num_seqs             同时并发处理的序列条数上限，如 256。
#   block_size               PagedAttention 每个 KV block 的 token 数，如 16。
#                            分页粒度，一般用默认即可。
#
# ---- llama.cpp：Llama(...) ----
#   model_path               GGUF 量化模型文件路径，如 xxx.Q4_K_M.gguf
#   n_ctx                    上下文窗口大小(token)，如 4096。prompt+生成不能超它。
#   n_threads                CPU 线程数，如 8。纯 CPU 推理时越多越快(到核数为止)。
#   n_gpu_layers             放到 GPU 上的层数。0=纯 CPU；在 Mac 上可用 Metal 加速；
#                            大模型显存不够时，只放前 N 层到 GPU，其余留 CPU（卸载）。
#
# ---- TGI（Docker 启动参数，命令行）----
#   --model-id               模型
#   --max-batch-total-tokens 一个 batch 总 token 上限（≈vLLM 的 max_num_batched_tokens）
#   --max-input-length       单条输入最大 token 数
#   --shm-size 1g            共享内存大小（多进程通信用）
#
# ---- llama.cpp server（命令行）常用 ----
#   -c 2048            上下文大小
#   --threads 4        CPU 线程
#   --n-gpu-layers 32  GPU 层数
#   --mlock            锁定内存防止被系统换出到磁盘（降低卡顿）
#   --cont-batching    开启连续批处理（生产必开）
# ==============================================================================


# ==============================================================================
# 第 3 部分：四种客户端调用方式（同一个服务，四种“打电话”姿势）
# ==============================================================================
# 前提：先有一个在跑的服务，监听某端口（vLLM 常见 8000，TGI/llama.cpp 常见 8080）。
# 下面四种客户端连的是“同一个 HTTP 服务”，只是库不同、参数名略有差异。

# ---- 3.1 HuggingFace InferenceClient（TGI 原生风格，参数名带 _new_/_sequences）
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

# (a) 纯文本生成
resp = client.text_generation(
    "tell me a story",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    stop_sequences=["\n\n"],
    details=True,          # 返回细节（含各 token 概率）
)
print(resp.generated_text)

# (b) 对话格式（自动套聊天模板）
resp = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=200,
    temperature=0.7,
    top_p=0.95,
)
print(resp.choices[0].message.content)


# ---- 3.2 OpenAI 官方客户端（最通用：vLLM / TGI / llama.cpp / mlx 都兼容）
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",   # 注意结尾 /v1
    api_key="not-needed",                  # 本地服务通常不校验，随便填
)

resp = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",  # 本地服务里这个字段常可随便填
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    top_p=0.95,
    frequency_penalty=0.5,   # OpenAI 生态用这一对来控重复
    presence_penalty=0.5,
    max_tokens=200,
)
print(resp.choices[0].message.content)


# ---- 3.3 llama.cpp 原生库（llama-cpp-python，最细粒度、可纯本地无服务）
from llama_cpp import Llama

llm = Llama(
    model_path="smollm2-1.7b-instruct.Q4_K_M.gguf",
    n_ctx=4096,        # 上下文窗口
    n_threads=8,       # CPU 线程
    n_gpu_layers=0,    # 0=纯 CPU；Mac 上可设 -1 尽量用 Metal
)

# 注意：原生库要自己按模型的对话模板拼 prompt（这里是 ChatML 格式）
prompt = (
    "<|im_start|>system\nYou are a creative storyteller.<|im_end|>\n"
    "<|im_start|>user\nWrite a creative story<|im_end|>\n"
    "<|im_start|>assistant\n"
)
out = llm(
    prompt,
    max_tokens=200,
    temperature=0.8,
    top_p=0.95,
    repeat_penalty=1.1,           # 注意：这里叫 repeat_penalty
    stop=["<|im_end|>"],
)
print(out["choices"][0]["text"])


# ---- 3.4 vLLM 原生 Python 接口（Linux+GPU 才能真跑，这里作为“读懂”对象）
# from vllm import LLM, SamplingParams
#
# llm = LLM(
#     model="HuggingFaceTB/SmolLM2-360M-Instruct",
#     gpu_memory_utilization=0.85,
#     max_num_batched_tokens=8192,
#     max_num_seqs=256,
#     block_size=16,
# )
# sampling_params = SamplingParams(
#     temperature=0.8,
#     top_p=0.95,
#     max_tokens=100,
#     presence_penalty=1.1,
#     frequency_penalty=1.1,
#     stop=["\n\n", "###"],
# )
# outputs = llm.generate("Write a creative story", sampling_params)   # 注意：generate 与 print 要分两行！
# print(outputs[0].outputs[0].text)
#
# 原始笔记里 `outputs = llm.generate(...) print(...)` 挤在一行是语法错误，
# 手敲时务必拆成两行。


# ==============================================================================
# 第 4 部分：四个生产案例（对照手敲）
# ==============================================================================
# 说明：案例 1~3 是标准“连 OpenAI 兼容服务”的生产写法，vLLM/TGI/llama.cpp/mlx
#       起的服务都能用；案例 4 是你 Mac 本机就能真跑通的完整闭环。
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# 生产案例 1：高并发客服机器人（低温度、稳定、带系统提示词与超时/重试）
#   场景：企业客服，要求回答稳定、可控、别乱发挥；并发高，需超时与重试兜底。
# ------------------------------------------------------------------------------
from openai import OpenAI

def build_client():
    return OpenAI(
        base_url="http://localhost:8000/v1",   # 生产里换成你的服务地址
        api_key="not-needed",
        timeout=30.0,          # 单请求超时(秒)，防止一条卡死拖垮线程
        max_retries=2,         # 失败自动重试次数
    )

SYSTEM_PROMPT = (
    "你是某电商平台的客服助手。只依据公司政策回答，"
    "语气礼貌简洁；不确定的信息不要编造，引导用户联系人工。"
)

def answer_customer(client, question: str) -> str:
    resp = client.chat.completions.create(
        model="qwen2.5",                       # 本地服务此字段常可随意
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.2,       # 低温 → 稳定、少幻觉（客服核心诉求）
        top_p=0.9,
        max_tokens=256,
        frequency_penalty=0.3, # 轻微抑制重复
        presence_penalty=0.0,
        stop=["用户:", "客服:"],  # 防止模型自问自答续写多轮
    )
    return resp.choices[0].message.content.strip()

# 用法：
# client = build_client()
# print(answer_customer(client, "我的订单还没发货，怎么办？"))


# ------------------------------------------------------------------------------
# 生产案例 2：流式输出的写作助手（stream=True，打字机效果，边生成边显示）
#   场景：面向 C 端的写作/聊天产品，用户体验要求“秒回、逐字蹦”，降低等待焦虑。
# ------------------------------------------------------------------------------
from openai import OpenAI

def stream_story(prompt: str):
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    stream = client.chat.completions.create(
        model="qwen2.5",
        messages=[
            {"role": "system", "content": "你是富有想象力的中文故事作家。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,       # 高温 → 更有创意
        top_p=0.95,
        presence_penalty=0.6,  # 鼓励引入新意象，别老用同一批词
        max_tokens=500,
        stream=True,           # 关键：开启流式
    )
    full = []
    for chunk in stream:                       # 逐块到达
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)   # 实时打印，不换行、立即刷新
            full.append(delta)
    print()
    return "".join(full)

# 用法：
# stream_story("写一个关于深海灯塔守护者的短故事")


# ------------------------------------------------------------------------------
# 生产案例 3：结构化信息抽取（极低温度 + JSON 输出 + stop，做数据管线）
#   场景：把非结构化文本（简历/工单/评论）抽成 JSON 入库。要求可解析、可复现。
# ------------------------------------------------------------------------------
import json
from openai import OpenAI

EXTRACT_SYSTEM = (
    "你是信息抽取引擎。只输出 JSON，不要任何解释文字。"
    "字段：name(姓名), years(工作年限,整数), skills(技能数组)。"
)

def extract_resume(text: str) -> dict:
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    resp = client.chat.completions.create(
        model="qwen2.5",
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0.0,       # 抽取任务：0 温度，追求确定可复现
        top_p=1.0,
        max_tokens=300,
        stop=["```"],          # 防止模型把 JSON 包进 markdown 代码块后继续啰嗦
        # 若服务支持，可加 response_format={"type": "json_object"} 强制 JSON
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 生产里要有兜底：记录原文、告警、进人工复核队列
        return {"_error": "invalid_json", "_raw": raw}

# 用法：
# print(extract_resume("张三，从事后端开发8年，精通 Python、Go 和 Kubernetes。"))


# ------------------------------------------------------------------------------
# 生产案例 4：本机可跑的完整闭环（mlx-lm 起 OpenAI 兼容服务 → Python 调用）
#   场景：Apple Silicon 边缘/本地部署，无需 GPU 云，隐私数据不出本机。
#   ★ 这个在你的 M4 Mac 上已验证可真跑（vLLM 的本地平替）。
#
#   第一步：终端里先启动服务（保持这个终端开着）：
#       python3 -m mlx_lm server \
#           --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
#           --port 8080
#
#   第二步：另开一个终端/脚本，用 OpenAI 客户端连它：
# ------------------------------------------------------------------------------
from openai import OpenAI

def local_chat(user_msg: str) -> str:
    client = OpenAI(
        base_url="http://localhost:8080/v1",   # mlx server 默认端口 8080
        api_key="not-needed",
    )
    resp = client.chat.completions.create(
        model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        messages=[
            {"role": "system", "content": "你是一个简洁的中文助手。"},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        top_p=0.95,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    # 确认 mlx server 已在 8080 跑起来后，运行本文件即可看到真实输出
    print(local_chat("用一句话解释什么是连续批处理"))
