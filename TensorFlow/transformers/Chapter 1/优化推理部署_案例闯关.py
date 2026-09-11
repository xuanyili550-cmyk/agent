"""
================================================================================
 优化推理部署 案例闯关 · 用 mlx-lm 在 Mac 上实践 vLLM 的核心概念（6 关）
================================================================================
 ⚠️ 为什么用 mlx-lm 而不是 vLLM？
    vLLM 需要 Linux + NVIDIA GPU，你的 M4 Mac 装不了（本会话开头已确认）。
    mlx-lm 是 Apple 官方推理框架，能在 Mac 上真跑，且概念与 vLLM 一一对应：
    采样参数、流式输出、批处理吞吐、OpenAI 兼容服务——学到的都能平移到 vLLM。

 用法：
   python3 优化推理部署_案例闯关.py         # 看菜单
   python3 优化推理部署_案例闯关.py 1        # 只跑第 1 关
   python3 优化推理部署_案例闯关.py all       # 全部
 也可在 PyCharm 直接点运行 → 按提示输入关号。

 关卡地图（括号里是对应的 vLLM 概念）：
   1  本地生成初体验      load + generate           (对应 vLLM 的 LLM.generate)
   2  采样参数对照        temperature/top_p 控制     (对应 vLLM SamplingParams)
   3  流式输出            stream_generate 打字机      (对应 OpenAI/vLLM stream=True)
   4  批处理吞吐          批量 vs 逐条计时            (对应 vLLM 连续批处理的收益)
   5  起 OpenAI 兼容服务   子进程起服务 + 客户端调用   (对应 vLLM serve + OpenAI API)
   6  综合:本地聊天助手    加载一次多轮对话            (一个能跑的本地部署雏形)

 模型：mlx-community/Qwen2.5-0.5B-Instruct-4bit（4bit量化小模型，你已下载）
================================================================================
"""

import sys
import time

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def title(n, text):
    print("\n" + "=" * 72)
    print(f"  第 {n} 关：{text}")
    print("=" * 72)


# ==============================================================================
# 第 1 关：本地生成初体验
# ==============================================================================
def case_01():
    title(1, "本地生成初体验（对应 vLLM 的 LLM.generate）")
    from mlx_lm import load, generate

    print("加载模型中……")
    model, tok = load(MODEL)     # 类似 vLLM 的 LLM(model=...)
    prompt = "Explain what an LLM is in one sentence."
    out = generate(model, tok, prompt, max_tokens=50, verbose=False)
    print(f"提问：{prompt}")
    print(f"回答：{out.strip()}")
    print("👉 一次加载、一行生成。vLLM 里是 llm.generate(prompt, sampling_params)，同理。")


# ==============================================================================
# 第 2 关：采样参数对照（temperature / top_p）
# ==============================================================================
def case_02():
    title(2, "采样参数对照：temperature 控制随机性（对应 SamplingParams）")
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    model, tok = load(MODEL)
    prompt = "Write one creative sentence about the ocean:"

    def run(temp, top_p=0.95):
        mx.random.seed(0)   # 固定随机种子，让差异只来自被改的参数（控制变量，同 Ch1 第13关）
        sampler = make_sampler(temp=temp, top_p=top_p)
        return generate(model, tok, prompt, max_tokens=40, sampler=sampler).strip()

    print("【temp=0.0】贪心/确定（每次一样，最保守）：")
    print("  ", run(0.0))
    print("【temp=0.8】适度发散：")
    print("  ", run(0.8))
    print("【temp=1.5】高度发散/大胆（也更可能跑偏）：")
    print("  ", run(1.5))
    print("👉 temp、top_p 就是 vLLM SamplingParams 里的同名参数，作用完全一致。")


# ==============================================================================
# 第 3 关：流式输出（打字机效果）
# ==============================================================================
def case_03():
    title(3, "流式输出：边生成边显示（对应 OpenAI/vLLM stream=True）")
    from mlx_lm import load, stream_generate

    model, tok = load(MODEL)
    prompt = "Tell me a very short story about a robot:"
    print(f"提问：{prompt}\n回答（逐字蹦出来）：")
    for chunk in stream_generate(model, tok, prompt, max_tokens=60):
        print(chunk.text, end="", flush=True)   # 实时打印，不换行、立即刷新
    print("\n👉 流式让用户‘秒见首字’，体验更好。vLLM/OpenAI 里就是 stream=True。")


# ==============================================================================
# 第 4 关：批处理吞吐（连续批处理的收益）
# ==============================================================================
def case_04():
    title(4, "批处理吞吐：批量 vs 逐条（对应 vLLM 连续批处理的核心收益）")
    from mlx_lm import load, generate, batch_generate

    model, tok = load(MODEL)
    prompts_text = [
        "Say hello in English.",
        "Say hello in French.",
        "Say hello in Spanish.",
        "Say hello in German.",
    ]

    # 逐条生成（模拟“没有批处理”）
    t0 = time.perf_counter()
    for p in prompts_text:
        generate(model, tok, p, max_tokens=20)
    seq_time = time.perf_counter() - t0

    # 批量生成（一次喂一批，GPU 利用率更高）
    prompt_ids = [tok.encode(p) for p in prompts_text]
    t0 = time.perf_counter()
    batch_generate(model, tok, prompt_ids, max_tokens=20, verbose=False)
    batch_time = time.perf_counter() - t0

    print(f"逐条生成 {len(prompts_text)} 条耗时：{seq_time:.2f}s")
    print(f"批量生成 {len(prompts_text)} 条耗时：{batch_time:.2f}s")
    print(f"批量快了约 {seq_time / batch_time:.1f} 倍")
    print("👉 这就是 vLLM 最大的性能来源：连续批处理让 GPU 不空转，吞吐成倍提升。")


# ==============================================================================
# 第 5 关：起一个 OpenAI 兼容服务并调用（vLLM serve 的等价物）
# ==============================================================================
def case_05():
    title(5, "起 OpenAI 兼容服务 + 客户端调用（对应 vLLM serve + OpenAI API）")
    import subprocess
    import urllib.request

    port = 8080
    print(f"在后台启动 mlx-lm 服务（端口 {port}）……")
    # 相当于 vLLM: python -m vllm.entrypoints.openai.api_server --model ...
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server", "--model", MODEL, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # 轮询等待服务就绪
        up = False
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=1)
                up = True
                break
            except Exception:
                time.sleep(1)
        if not up:
            print("❌ 服务启动超时"); return
        print("✅ 服务已就绪，用 OpenAI 客户端调用（和调 vLLM 一模一样的代码）：\n")

        from openai import OpenAI
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is continuous batching? One sentence."},
            ],
            temperature=0.7, top_p=0.95, max_tokens=60,
        )
        print("  用户：What is continuous batching? One sentence.")
        print("  助手：", resp.choices[0].message.content.strip())
        print("\n👉 关键：客户端代码对 vLLM / mlx / TGI 通用——只要服务是 OpenAI 兼容的。")
    finally:
        proc.terminate()
        proc.wait()
        print("（服务已关闭）")


# ==============================================================================
# 第 6 关：综合 —— 本地聊天助手（加载一次，多轮对话）
# ==============================================================================
def case_06():
    title(6, "综合：本地聊天助手（加载一次，模型带上下文多轮对话）")
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(MODEL)
    sampler = make_sampler(temp=0.7, top_p=0.95)

    # 模拟一段多轮对话：把历史拼进 chat template，模型就能“记住”上下文
    messages = [{"role": "system", "content": "You are a helpful, concise assistant."}]
    turns = [
        "My name is Alex.",
        "What is 12 times 8?",
        "What did I say my name was?",   # 考验它是否记住上下文
    ]
    for user_msg in turns:
        messages.append({"role": "user", "content": user_msg})
        # apply_chat_template：把多轮历史拼成模型认识的格式（含角色标记）
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        reply = generate(model, tok, prompt, max_tokens=40, sampler=sampler).strip()
        messages.append({"role": "assistant", "content": reply})
        print(f"  用户：{user_msg}")
        print(f"  助手：{reply}\n")
    print("👉 把对话历史一起喂进去，模型就有了‘记忆’。这就是一个本地部署的聊天雏形，")
    print("   换成 vLLM 服务后，把 messages 发给 /v1/chat/completions 即可，逻辑不变。")


# ------------------------------------------------------------------------------
# 调度器
# ------------------------------------------------------------------------------
CASES = {1: case_01, 2: case_02, 3: case_03, 4: case_04, 5: case_05, 6: case_06}

MENU = """\
用法：
  python3 优化推理部署_案例闯关.py <关号>   跑单关，如 1 / 4 / 5
  python3 优化推理部署_案例闯关.py all       跑全部

关卡列表（都用 mlx-lm 在 Mac 上真跑，概念对应 vLLM）：
  1  本地生成初体验     2  采样参数对照(temp)   3  流式输出
  4  批处理吞吐         5  起 OpenAI 兼容服务   6  综合:本地聊天助手
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
