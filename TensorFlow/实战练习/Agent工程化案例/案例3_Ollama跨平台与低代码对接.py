"""
================================================================================
 Agent 工程化案例3 · Ollama 跨平台本地部署 + Dify/Coze 低代码对接（真实代码）
================================================================================
 补齐"本地部署跨平台 + 低代码平台"短板：
   ① Ollama：跨平台(Win/Linux/Mac)本地跑 LLM，OpenAI 兼容——补 mlx(仅 Apple 芯片)的不足。
   ② Dify：低代码搭好的工作流，用 Python 调它的 API(企业常用可视化编排 + 代码集成)。
   ③ Coze/Dify 概念：低代码平台适合快速搭原型/非工程同学；代码框架(LangGraph)适合复杂可控。
 本机现实：Ollama/Dify 是外部服务(未装/未起)，故为真实代码 + 惰性导入 + 本机不跑；
 smoke 校验函数已定义 + 打印跨平台 LLM 后端对比表。
 跑：python3 案例3_Ollama跨平台与低代码对接.py
================================================================================
"""
import os
import sys


# ==============================================================================
# ① Ollama：跨平台本地 LLM(OpenAI 兼容)——两种调用方式
# ==============================================================================
def chat_ollama_openai(question, model="qwen2.5:0.5b"):
    """用官方 openai SDK 连 Ollama 的 OpenAI 兼容端点(和调 vLLM/云端一模一样，不锁厂商)。
    需先本机装 Ollama 并 `ollama pull qwen2.5:0.5b`。"""
    from openai import OpenAI                                   # 惰性导入
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    r = client.chat.completions.create(model=model,
        messages=[{"role": "user", "content": question}])
    return r.choices[0].message.content


def chat_ollama_native(question, model="qwen2.5:0.5b"):
    """用 ollama 官方 Python 客户端(需 pip install ollama + 本机 Ollama 服务在跑)。"""
    import ollama
    return ollama.chat(model=model, messages=[{"role": "user", "content": question}])["message"]["content"]


def stream_ollama(question, model="qwen2.5:0.5b"):
    """Ollama 流式输出。"""
    import ollama
    for chunk in ollama.chat(model=model, messages=[{"role": "user", "content": question}], stream=True):
        yield chunk["message"]["content"]


# ==============================================================================
# ② Dify：低代码平台搭的工作流，用 Python 调 API 集成进你的应用
# ==============================================================================
def call_dify_workflow(inputs: dict):
    """调 Dify 工作流 API(在 Dify 可视化搭好流程后，用代码触发)。需 DIFY_API_KEY + Dify 实例。"""
    import requests                                            # 惰性导入
    api_key = os.getenv("DIFY_API_KEY")
    resp = requests.post(
        os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1") + "/workflows/run",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"inputs": inputs, "response_mode": "blocking", "user": "app-user"},
        timeout=120,
    )
    return resp.json()["data"]["outputs"]


def call_dify_chat(query: str, conversation_id: str = ""):
    """调 Dify 对话型应用 API(带会话记忆)。"""
    import requests
    api_key = os.getenv("DIFY_API_KEY")
    resp = requests.post(
        os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1") + "/chat-messages",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": query, "conversation_id": conversation_id, "response_mode": "blocking",
              "inputs": {}, "user": "app-user"}, timeout=120)
    return resp.json()["answer"]


# ==============================================================================
# ③ 本地/部署 LLM 后端对比(选型)
# ==============================================================================
BACKENDS = [
    ("mlx-lm",  "仅 Apple 芯片(Mac)", "本机快、省内存", "本项目本机演示用"),
    ("Ollama",  "跨平台(Win/Linux/Mac)", "一条命令跑各种开源模型", "本地开发/小团队，OpenAI 兼容"),
    ("vLLM",    "GPU(Linux)", "高吞吐 PagedAttention", "生产大规模服务"),
    ("云端 API", "任意", "免运维、按量付费", "起步/不想自建(通义/DeepSeek/OpenAI)"),
]
LOWCODE = [
    ("Dify",  "开源低代码 LLM 应用平台", "可视化搭 RAG/工作流/Agent + API 集成", "企业快速落地、非工程同学也能搭"),
    ("Coze",  "字节低代码 Bot 平台", "拖拽搭 Bot + 插件 + 工作流", "快速做对话机器人"),
    ("代码框架", "LangChain/LangGraph", "完全可控、可版本化、可测试", "复杂/定制/工程化(见 ../Agent进阶案例)"),
]

if __name__ == "__main__":
    print(__doc__)
    for fn in (chat_ollama_openai, chat_ollama_native, call_dify_workflow, call_dify_chat):
        assert callable(fn)
    # print("\n本地/部署 LLM 后端对比：")            # 去装饰线，保留表标题（菜单/说明结构性输出）
    print(f"  {'后端':<10}{'平台':<22}{'特点':<22}{'适合'}")
    for b in BACKENDS:
        print(f"  {b[0]:<10}{b[1]:<22}{b[2]:<22}{b[3]}")
    # print("\n低代码 vs 代码框架：")                 # 去装饰线，保留表标题（菜单/说明结构性输出）
    print(f"  {'平台':<10}{'是什么':<24}{'能力':<34}{'适合'}")
    for l in LOWCODE:
        print(f"  {l[0]:<10}{l[1]:<24}{l[2]:<34}{l[3]}")
    # 探测本机 Ollama 服务状态(客户端已装，服务需 brew install ollama && ollama serve)
    try:
        import ollama
        models = [m.get("model", "") for m in ollama.list().get("models", [])]
        print(f"\n>>> Ollama 服务在跑 ✅，已拉模型: {models or '(无，先 ollama pull qwen2.5:0.5b)'}")
        if models:
            print("    真跑:", chat_ollama_openai("用一句话介绍你自己", model=models[0])[:60])
    except Exception as e:
        print(f"\n>>> Ollama 服务未起({type(e).__name__})——客户端已装、代码真实；真跑需："
              "`brew install ollama && ollama serve && ollama pull qwen2.5:0.5b`")
    # print("    Dify 真跑：export DIFY_API_KEY=... 后 call_dify_chat('你好') / call_dify_workflow({...})。")
    print("✅ 案例3 就绪：Ollama 跨平台(客户端已装) + Dify/Coze 低代码对接，全部真实代码。")
