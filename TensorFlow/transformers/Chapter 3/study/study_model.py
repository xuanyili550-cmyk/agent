import asyncio
from vllm import LLM, SamplingParams

llm=LLM(
    model='Qwen/Qwen2.5-7B-Instruct',
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    dtype="auto",
)
sampling_params = SamplingParams(
    temperature=0.7,        # 随机性：0=确定，越高越发散
    top_p=0.95,             # 核采样：只从累计概率前 95% 的词里选
    top_k=50,               # 只从概率最高的 50 个词里选（0=不限）
    max_tokens=256,         # 最多生成多少新 token
    presence_penalty=0.5,   # 出现过就惩罚 → 鼓励新话题
    frequency_penalty=0.5,  # 按出现次数惩罚 → 抑制刷屏
    stop=["\n\n", "###"],   # 遇到这些字符串就停
)
prompts = [
    "用一句话介绍杭州。",
    "把这句话翻译成英文：今天天气很好。",
    "给'量子计算'写一句科普。",
    # ……实际生产里这里可能是几万、几十万条
]
outputs = llm.generate(prompts,sampling_params)

for out in outputs:
    prompt = out.prompt
    text = out.outputs[0].text          # outputs[0] 是第一个候选（n=1 时就它）
    print(f"输入：{prompt}\n输出：{text}\n")
messages = [
    {"role": "system", "content": "你是简洁的中文助手。"},
    {"role": "user", "content": "什么是连续批处理？一句话。"},
]
chat_outputs = llm.chat(messages, sampling_params)
print(chat_outputs[0].outputs[0].text)

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

from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.utils import random_uuid

engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
       model="Qwen/Qwen2.5-7B-Instruct",
       gpu_memory_utilization=0.9,
       max_num_seqs=256,
   ))
async def generate_text(prompt: str):
    request_id = random_uuid()

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=200,
    )

    # 3. engine.generate() 返回异步生成器
    results_generator = engine.generate(
        prompt,
        sampling_params,
        request_id,
    )

    # 4. 持续消费流式结果
    async for request_output in results_generator:
        # 当前已经生成的完整文本
        text = request_output.outputs[0].text

        print(text)


# 5. 启动异步程序
async def main():
    prompt = "请介绍一下 Transformer 的核心思想。"

    await generate_text(prompt)
if __name__ == "__main__":
    asyncio.run(main())

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
guided=GuidedDecodingParams(json=json_schema)
params=SamplingParams(temperature=0.0, max_tokens=200, guided_decoding=guided)
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
out = llm.generate("提取信息：张三，28岁，会 Python 和 Go。", params)
print(out[0].outputs[0].text)

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