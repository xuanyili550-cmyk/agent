"""
================================================================================
 Agent 工程化案例 · 补齐"部署/联网/全栈/低代码"短板（对标 didilili 的工程落地）
================================================================================
 接 ../Agent进阶案例(NL2SQL/多智能体/混合检索)之后，这里补"上线工程化"的几块：
   案例1_联网搜索Agent.py                DuckDuckGo【真联网】搜索 + LangGraph + mlx 综合带来源回答(本机真跑)
   案例2_FastAPI_WebSocket全栈服务/      FastAPI + WebSocket【实时进度+流式回答】+ Dockerfile + docker-compose(app+ollama)
                                        TestClient 本机真测 WebSocket 通过；`docker compose up` 一键部署
   案例3_Ollama跨平台与低代码对接.py      Ollama 跨平台本地 LLM(补 mlx 仅 Mac) + Dify/Coze 低代码对接(真实代码)

 —— 至此，相比 didilili/ai-agents-from-zero 补齐的清单 ——
   ✅ LangGraph 真状态机/多智能体   ✅ NL2SQL   ✅ 企业级混合检索(向量+BM25+RRF+重排)
   ✅ 联网搜索(DuckDuckGo 真跑)     ✅ 全栈 FastAPI+WebSocket+Docker Compose 一键部署
   ✅ Ollama 跨平台本地部署        ✅ Dify/Coze 低代码对接(参考)
   (联网也可换 Tavily；Docker Compose 里已放 Ollama 作 Linux 上的 LLM 后端替代 mlx。)

 —— 本机真跑 vs 需外部服务 ——
   真跑：案例1(需联网) / 案例2(TestClient 真测 WebSocket)
   参考(需外部服务)：案例2 的 docker compose(需 Docker) / 案例3 的 Ollama(需装)+Dify(需 key)——均真实代码。
 跑：python3 案例1_联网搜索Agent.py "问题"   ·   python3 案例2_.../app.py smoke   ·   python3 案例3_....py
================================================================================
"""
CASES = [
    ("案例1_联网搜索Agent", "DuckDuckGo真联网+LangGraph+mlx", "本机真跑(需联网)", "深度研究/联网"),
    ("案例2_FastAPI_WebSocket全栈服务", "FastAPI+WebSocket流式+Docker Compose", "TestClient真测", "全栈部署"),
    ("案例3_Ollama跨平台与低代码对接", "Ollama+Dify/Coze", "真实代码(需外部服务)", "跨平台部署/低代码"),
]
if __name__ == "__main__":
    print(__doc__)
    print(f" {'案例':<32}{'技术':<40}{'运行':<20}{'补的短板'}")
    for n, t, r, g in CASES:
        print(f" {n:<32}{t:<40}{r:<20}{g}")
