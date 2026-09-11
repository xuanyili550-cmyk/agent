from huggingface_hub import InferenceClient
from openai import OpenAI
from llama_cpp import Llama
#from vllm import LLM, SamplingParams
client=InferenceClient(model='http://localhost:8080')
response=client.text_generation(
    'tell me a story',
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    details=True,
    stop_sequences=[]
)
print(response.generated_text)
response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].message.content)

#openai
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

response=client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a story"},
    ],
    max_tokens=100,
    temperature=0.7,
    top_p=0.95
)

print(response.choices[0].message.content)

#使用 InferenceClient 实现灵活的文本生成
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

# Raw text generation
response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    details=True,
)
print(response.generated_text)

#使用 OpenAI 客户端
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,  # Higher for more creativity
)
print(response.choices[0].message.content)


#使用推理客户端：
client = InferenceClient(model="http://localhost:8080/v1", token="sk-no-key-required")

response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1,
    details=True,
)
print(response.generated_text)

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key-required")

response = client.chat.completions.create(
    model="smollm2-1.7b-instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    # 温度越高，创意性越强
    temperature=0.8,
# 核心采样概率
    top_p=0.95,
# 减少频繁出现的标记的重复
    frequency_penalty=0.5,
# 通过惩罚已存在的标记来减少重复
    presence_penalty=0.5,
# 最大生成长度
    max_tokens=200,
)
print(response.choices[0].message.content)

#使用 llama.cpp 的原生库来获得更多控制权

llm=Llama(
    model_path='smollm2-1.7b-instruct.Q4_K_M.gguf',
    n_ctx = 4096,  # 上下文窗口大小
    n_threads = 8,  # CPU 线程数
    n_gpu_layers = 0,  # GPU 层数（0 = 仅 CPU）
)
# 根据模型预期格式格式化提示
prompt = """<|im_start|>system
You are a creative storyteller.
<|im_end|>
<|im_start|>user
Write a creative story
<|im_end|>
<|im_start|>assistant
"""# 生成具有精确参数控制的响应

output=llm(
    prompt,
    max_tokens=200,
    temperature=0.8,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["<|im_end|>"]
)
print(output["choices"][0]['text'])


#使用 vLLM 的高级功能，可以使用 InferenceClient

client = InferenceClient(model="http://localhost:8000/v1")

response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    max_tokens=200,
    top_p=0.95,
)
print(response.choices[0].message.content)

response = client.text_generation(
    "Write a creative story about space exploration",
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.95,
    details=True,
)
print(response.generated_text)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "system", "content": "You are a creative storyteller."},
        {"role": "user", "content": "Write a creative story"},
    ],
    temperature=0.8,
    top_p=0.95,
    max_tokens=200,
)
print(response.choices[0].message.content)

# #vLLM 还提供了一个具有细粒度控制功能的原生 Python 接口
# llm=LLM(
#     model= "HuggingFaceTB/SmolLM2-360M-Instruct" ,
#     gpu_memory_utilization= 0.85 ,
#     max_num_batched_tokens=8192,
#     max_num_seqs=256,
#     block_size=16,
# )
# # 配置采样参数
# sampling_params = SamplingParams(
#     temperature= 0.8 ,   # 温度越高，创意越多
#     top_p= 0.95 ,   # 考虑前 95% 的概率质量
#     max_tokens= 100 ,   # 最大长度
#     presence_penalty= 1.1 ,   # 减少重复
#     frequency_penalty= 1.1 ,   # 减少重复
#     stop=[ "\n\n" , "###" ],   # 停止序列
# ) # 生成文本
#
# prompt = "Write a creative story"
# outputs = llm.generate(prompt, sampling_params) print (outputs[ 0 ].outputs[ 0 ].text) # 用于聊天式交互
# chat_prompt = [
#     {"role": "system", "content": "You are a creative storyteller."},
#     {"role": "user", "content": "Write a creative story"},
# ]
# formatted_prompt = llm.get_chat_template()(chat_prompt)   # 使用模型的聊天模板
# outputs = llm.generate(formatted_prompt, sampling_params) print (outputs[ 0 ].outputs[ 0 ].text)

# 先进发电控制
# 令牌选择和抽样
# 文本生成过程涉及每一步选择下一个词元。此选择过程可通过各种参数进行控制：
#
# 原始 Logits：每个标记的初始输出概率
# 温度：控制选择的随机性（温度越高，创造性越强）
# Top-p（核心）采样：筛选出概率质量占比 X% 的前几个标记。
# 前k个筛选：将选择范围限制在最有可能的k个词元。

client.generate(
     "编写一个创意故事" ,
    temperature= 0.8 ,   # 温度越高，创意越多
    top_p= 0.95 ,   # 考虑前95%的概率
    top_k= 50 ,   # 考虑前50个词元
    max_new_tokens= 100 ,   # 最大长度
    repetition_penalty= 1.1 ,   # 减少重复
)
# 通过 OpenAI API 兼容性
response = client.completions.create(
    model= "smollm2-1.7b-instruct" ,   # 模型名称（llama.cpp 服务器可以是任何字符串）
    prompt= "编写一个创意故事" ,
    temperature= 0.8 ,   # 温度越高，创意越多
    top_p= 0.95 ,   # 考虑前 95% 的概率质量
    frequency_penalty= 1.1 ,   # 减少重复
    presence_penalty= 0.1 ,   # 减少重复
    max_tokens= 100 ,   # 最大长度
) # 通过 llama-cpp-python 直接访问
output = llm( "编写一个创意故事" ,
    temperature= 0.8 ,
    top_p= 0.95 ,
    top_k= 50 ,
    max_tokens= 100 ,
    repeat_penalty= 1.1 ,
)

# 控制重复
# 这两个框架都提供了防止重复文本生成的方法：
#
# <hfoption value="tgi" label="TGI">
client.generate(
     "编写多样化的文本" ,
    repetition_penalty= 1.1 ,   # 对重复的词元进行惩罚
    no_repeat_ngram_size= 3 ,   # 防止三元组重复
)

# 通过 OpenAI API
response = client.completions.create(
    model= "smollm2-1.7b-instruct" ,
    prompt= "编写一段多样化的文本" ,
    frequency_penalty= 1.1 ,   # 对频繁出现的词元进行惩罚
    presence_penalty= 0.8 ,   # 对已出现的词元进行惩罚
) # 通过直接调用库
output = llm( "编写一段多样化的文本" ,
    repeat_penalty= 1.1 ,   # 对重复出现的词元进行惩罚
    frequency_penalty= 0.5 ,   # 对重复出现的词元进行额外惩罚
    presence_penalty= 0.5 ,   # 对已出现的词元进行额外惩罚
)
#
# #</hfoption> <hfoption value="vllm" label="vLLM">
# params = SamplingParams(
#     presence_penalty= 0.1 ,   # 对标记出现频率进行惩罚
#     frequency_penalty= 0.1 ,   # 对标记频率进行惩罚
# )

#您可以控制生成长度并指定何时停止
client.generate(
    "Generate a short paragraph",
    max_new_tokens=100,
    min_new_tokens=10,
    stop_sequences=["\n\n", "###"],
)

# 通过 OpenAI API
response = client.completions.create(
    model="smollm2-1.7b-instruct",
    prompt="Generate a short paragraph",
    max_tokens=100,
    stop=["\n\n", "###"],
)

# 通过直接使用库
output = llm("Generate a short paragraph", max_tokens=100, stop=["\n\n", "###"])
#
# params = SamplingParams(
#     max_tokens=100,
#     min_tokens=10,
#     stop=["###", "\n\n"],
#     ignore_eos=False,
#     skip_special_tokens=True,
# )

# 内存管理
# 这两个框架都采用了先进的内存管理技术，以实现高效的推理。
# 使用内存优化部署 Docker
# docker run --gpus all -p 8080:80 \
#     --shm-size 1g \
#     ghcr.io/huggingface/text-generation-inference:latest \
#     --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct \
#     --max-batch-total-tokens 8192 \
#     --max-input-length 4096
#
# llama.cpp 使用了量化和优化的内存布局：
# # 启用内存优化的服务器
# ./server \
#     -m smollm2-1.7b-instruct.Q4_K_M.gguf \
#     --host 0.0.0.0 \
#     --port 8080 \
#     -c 2048 \                # 上下文大小
#     --threads 4 \            # CPU 线程数
#     --n-gpu-layers 32 \      # 为更大的模型使用更多 GPU 层
#     --mlock \                # 锁定内存以防止交换
#     --cont-batching          # 启用连续批处理

# 对于GPU无法处理的巨大模型，可以使用CPU卸载：
# ./server \
#     -m smollm2-1.7b-instruct.Q4_K_M.gguf \
#     --n-gpu-layers 20 \      # 将前 20 层保留在 GPU 上
#     --threads 8              # 使用更多 CPU 线程处理 CPU 层

# #vLLM 使用 PagedAttention 实现最佳内存管理：
# from vllm.engine.arg_utils import AsyncEngineArgs
#
# engine_args = AsyncEngineArgs(
#     model= "HuggingFaceTB/SmolLM2-1.7B-Instruct" ,
#     gpu_memory_utilization= 0.85 ,
#     max_num_batched_tokens= 8192 ,
#     block_size= 16 ,
# )
#
# llm = LLM(engine_args=engine_args)