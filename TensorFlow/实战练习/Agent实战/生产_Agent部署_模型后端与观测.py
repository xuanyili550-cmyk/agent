"""
================================================================================
 Agent 实战 · 生产 · Agent 上线：模型后端(vLLM/HF) + GradioUI + 可观测(Langfuse) + 分享
================================================================================
 ⚠ 需要 LLM 服务(自建 vLLM 或 HF Token) + (可观测)Langfuse 账号，本机不跑；真实生产 Agent 代码。
   工具沿用前面案例(中文 NLP / 检索 / MCP)；本文件聚焦“把 Agent 真正部署出去 + 看得见”。
   本地能跑的工具自检见 案例2/3/4；LLM 服务端怎么起见 ../chapter实战/生产05。

 生产 Agent 五件事(都是真实代码，惰性导入，本机不执行)：
   ① 模型后端：接自建 vLLM(OpenAIServerModel，不锁厂商) 或 HF InferenceClientModel。
   ② 组装：add_base_tools 加内置工具、planning_interval 定期规划、max_steps 兜底防死循环。
   ③ 上线 UI：GradioUI(agent).launch() 一键网页；或挂 FastAPI(见 ../chapter实战/分章项目/Ch9_进阶)。
   ④ 可观测：OpenTelemetry + Langfuse(SmolagentsInstrumentor)——看每步 推理/工具调用/token/延迟/成本。
   ⑤ 分享/版本：agent.push_to_hub / from_hub，把 Agent(工具+提示+配置)打包版本化。
 跑：python3 生产_Agent部署_模型后端与观测.py     # 只打印架构与步骤(本机无服务不跑)
================================================================================
"""
import os


# ==============================================================================
# ① + ② 生产 Agent：接自建 vLLM(OpenAI 兼容) + 内置工具 + 规划 + 兜底步数
# ==============================================================================
def build_production_agent():
    """接自建 vLLM(生产01/05 起的 OpenAI 兼容服务)；换 HF/云端只改 model 那一行。"""
    from smolagents import CodeAgent, OpenAIServerModel, DuckDuckGoSearchTool
    # 复用前面章节的工具(中文 NLP)：
    from 案例2_NLP能力Agent_中文客服 import get_sentiment, extract_entities, search_faq

    model = OpenAIServerModel(
        model_id=os.getenv("LLM_MODEL", "qwen"),
        api_base=os.getenv("LLM_BASE_URL", "http://gpu-host:8001/v1"),   # 自建 vLLM
        api_key=os.getenv("LLM_API_KEY", "EMPTY"),
    )
    agent = CodeAgent(
        tools=[get_sentiment, extract_entities, search_faq, DuckDuckGoSearchTool()],
        model=model,
        add_base_tools=True,        # 加内置基础工具(如 python 执行、最终答案)
        planning_interval=3,        # 每 3 步做一次“规划”(复杂任务更稳)
        max_steps=8,                # 兜底：最多 8 步，防死循环
        verbosity_level=1,
    )
    return agent


def build_agent_hf():
    """备选：接 HF Inference(需 HF Token)。"""
    from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool
    return CodeAgent(tools=[DuckDuckGoSearchTool()],
                     model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"))


# ==============================================================================
# ③ 上线 UI：GradioUI 一键网页
# ==============================================================================
def serve_ui():
    from smolagents import GradioUI
    agent = build_production_agent()
    GradioUI(agent).launch()        # 起网页；也可 GradioUI(agent).launch(share=True) 出临时公网链接


# ==============================================================================
# ④ 可观测：OpenTelemetry + Langfuse（看每步推理/工具调用/token/延迟）
# ==============================================================================
def enable_observability():
    """在创建/运行 Agent 前调用一次，之后每次 agent.run 都会自动上报到 Langfuse。"""
    # 环境变量：LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    from langfuse import get_client
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    SmolagentsInstrumentor().instrument()       # ★埋点：smolagents 的每步自动生成 trace
    lf = get_client()
    if lf.auth_check():
        pass
        # print("Langfuse 已连接：Agent 的每步推理/工具调用/token/延迟都会上报，可在面板看轨迹与成本。")
    return lf


# ==============================================================================
# ⑤ 分享 / 版本化
# ==============================================================================
def publish(agent, repo_id):
    """把 Agent(工具+提示+配置)打包推到 Hub；别人 from_hub 一行复用。"""
    agent.push_to_hub(repo_id)
    from smolagents import CodeAgent
    return CodeAgent.from_hub(repo_id, trust_remote_code=True)


NOTES = """
 生产要点：
  · 模型后端：自建 vLLM(OpenAIServerModel) 最省成本可私有化；起步用 HF/云端(InferenceClientModel)快。
  · 防失控：max_steps 兜底、planning_interval 规划、工具做输入校验、危险操作要“人在环”确认。
  · 可观测是刚需：Agent 是多步黑盒，没 trace 出问题查不动。Langfuse/OpenTelemetry 看每步 + token/成本。
  · 安全：工具能执行真实操作 → 最小权限、审计日志、限流；Sampling/代码执行要沙箱。
  · 部署：GradioUI 快速演示；生产挂 FastAPI(Ch9_进阶 mount_gradio_app) + K8s(生产/部署) + HPA。
  · 评估：任务成功率、平均步数、工具调用正确率、端到端延迟/成本；难例回归集持续跑。
"""

if __name__ == "__main__":
    print(__doc__)
    print("=== 生产要点 ===", NOTES)
    # print(">>> 真实生产 Agent 代码，本机无 LLM 服务/Langfuse 不跑。真跑步骤：")
    # print("    1) 按 ../chapter实战/生产05 起 vLLM(OpenAI 兼容)；export LLM_BASE_URL=http://<host>:8001/v1")
    # print("    2) (可选)配 LANGFUSE_* 环境变量 → enable_observability()")
    # print("    3) python3 -c 'import 生产_Agent部署_模型后端与观测 as m; m.serve_ui()'  # 起 Agent 网页")
    # print("    本地工具自检见 案例2/3/4。")
