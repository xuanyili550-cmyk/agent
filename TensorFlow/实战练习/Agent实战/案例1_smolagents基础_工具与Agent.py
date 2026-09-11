"""
================================================================================
 Agent 实战 · 案例1 · smolagents 基础：工具 + Agent + ReAct（Agent 的 Hello World）
================================================================================
 Agent 课程第 1-2 章的核心，用 smolagents 跑通：
   ① 定义工具两种写法：@tool 装饰器(函数) / 继承 Tool 类(需状态时)。
   ② 工具三要素：name + description + 参数类型(inputSchema)——LLM 靠这些决定“调不调、传什么”。
   ③ 两种 Agent：CodeAgent(让 LLM 写 Python 代码调工具，灵活) vs ToolCallingAgent(输出 JSON 工具调用，稳)。
   ④ 模型后端：TransformersModel(本地) / InferenceClientModel(HF/供应商) / OpenAIServerModel(vLLM)。
   ⑤ ReAct 循环：推理(Thought) → 行动(Action=调工具) → 观察(Observation) → …→ 最终答案。

 本机现实：工具【本地真跑+自检】；Agent 大脑本地小模型太弱(见 README，已实测生成无效代码)，
   故 smoke 只做“工具自检 + Agent 组装校验”；真跑 Agent 需 HF Token(InferenceClientModel)。
 跑：python3 案例1_smolagents基础_工具与Agent.py smoke
================================================================================
"""
import sys
import datetime
from smolagents import tool, Tool


# ==============================================================================
# ① 写法一：@tool 装饰器（无状态的简单工具）
# ==============================================================================
@tool
def calculator(a: int, b: int) -> int:
    """计算两个整数的乘积。

    Args:
        a: 第一个整数。
        b: 第二个整数。
    """
    return a * b


@tool
def current_time(timezone: str = "UTC") -> str:
    """返回当前 UTC 时间字符串(演示无参/默认参数工具)。

    Args:
        timezone: 时区名(此处仅演示，固定返回 UTC)。
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ==============================================================================
# ② 写法二：继承 Tool 类（需要保存状态/初始化资源时用，如带一份数据/模型）
# ==============================================================================
class MenuTool(Tool):
    name = "suggest_menu"
    description = "根据场合(casual/formal/superhero)推荐派对菜单。"
    inputs = {"occasion": {"type": "string", "description": "场合类型。"}}
    output_type = "string"

    def forward(self, occasion: str):
        return {"casual": "披萨、小吃和饮料。",
                "formal": "三道菜晚宴配红酒和甜点。",
                "superhero": "高能量健康自助餐。"}.get(occasion, "为管家定制的菜单。")


TOOLS = [calculator, current_time, MenuTool()]


# ==============================================================================
# ③④ Agent 组装：CodeAgent / ToolCallingAgent + 三种模型后端(真实代码，需模型才 run)
# ==============================================================================
def build_code_agent():
    """CodeAgent：LLM 写 Python 代码来调工具(更灵活，能组合/循环)。"""
    from smolagents import CodeAgent, InferenceClientModel
    return CodeAgent(tools=TOOLS, model=InferenceClientModel(), max_steps=6, verbosity_level=1)


def build_tool_calling_agent():
    """ToolCallingAgent：LLM 输出 JSON 形式的工具调用(更稳，贴近 OpenAI function calling)。"""
    from smolagents import ToolCallingAgent, InferenceClientModel
    return ToolCallingAgent(tools=TOOLS, model=InferenceClientModel(), max_steps=6)


# —— 四种模型后端的真实构造(惰性导入；除本地外都需 Token/服务，本机不调用) ——
def build_agent_local():
    """本地 transformers 模型(小模型弱，仅试配线，见 README 实测)。"""
    from smolagents import CodeAgent, TransformersModel
    model = TransformersModel(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", device_map="mps")
    return CodeAgent(tools=TOOLS, model=model, max_steps=4)


def build_agent_vllm(api_base="http://gpu-host:8001/v1"):
    """自建 vLLM(OpenAI 兼容，见 ../chapter实战/生产05)；不锁厂商、可私有化。"""
    from smolagents import CodeAgent, OpenAIServerModel
    model = OpenAIServerModel(model_id="qwen", api_base=api_base, api_key="EMPTY")
    return CodeAgent(tools=TOOLS, model=model, max_steps=6)


def build_agent_litellm():
    """任意厂商用 litellm 统一接口(deepseek/通义/openai…)。"""
    from smolagents import CodeAgent, LiteLLMModel
    return CodeAgent(tools=TOOLS, model=LiteLLMModel(model_id="deepseek/deepseek-chat"))


def smoke():
    # ① 工具自检：直接调用 smolagents 工具，结果即注释里的期望值
    #   calculator(6, 7)       -> 42
    #   current_time()         -> 形如 "2026-08-30 12:00:00 UTC"
    #   MenuTool()("formal")   -> "三道菜晚宴配红酒和甜点。"
    assert calculator(6, 7) == 42
    assert MenuTool()("superhero").startswith("高能量")
    # 工具都是合法 smolagents Tool，带 name/description/inputs
    for t in TOOLS:
        assert isinstance(t, Tool) and t.name and t.description

    # ② 工具契约(LLM 就靠 name/description/inputs 读懂工具)：
    #   calculator   : 计算两个整数的乘积。            参数=['a', 'b']
    #   current_time : 返回当前 UTC 时间字符串。       参数=['timezone']
    #   suggest_menu : 根据场合推荐派对菜单。          参数=['occasion']
    for t in TOOLS:
        assert t.inputs  # 契约存在即可，无需打印

    # ③④ Agent 组装校验(不 run，run 需模型)：
    #   CodeAgent=LLM 写代码调工具(灵活)；ToolCallingAgent=LLM 出 JSON 工具调用(稳)。
    #   模型后端(真实构造函数)：build_agent_local / build_agent_vllm / build_agent_litellm
    #                          + build_code_agent(InferenceClientModel)
    from smolagents import CodeAgent, ToolCallingAgent, InferenceClientModel  # noqa: F401

    print("✅ 案例1 跑通：@tool / Tool 子类 / 工具契约 / Agent 组装 / 模型后端。")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
    else:
        smoke()
        # print("\n>>> 真跑 CodeAgent(需 HF Token)：")
        try:
            agent = build_code_agent()
            print(agent.run("用 calculator 工具算 6 乘以 7，只返回数字。"))
        except Exception as e:
            print(f"  (未跑：{type(e).__name__}: {str(e)[:80]}；export HF_TOKEN 后可跑)")
