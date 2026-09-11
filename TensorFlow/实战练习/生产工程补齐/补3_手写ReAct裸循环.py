"""
================================================================================
 生产工程补齐 · 补3 · 手写 ReAct 裸循环（不靠框架，看清 思考→行动→观测 的完整逻辑）
================================================================================
 补上「Agent 优先手写 ReAct 循环」这一块。前面的 Agent 案例都用 smolagents/LangGraph 封装，
 这里把框架撕开，纯 Python 手写一遍 ReAct，看清它到底在循环什么：
   Thought(思考要不要用工具、用哪个) → Action(调工具+传参) → Observation(把工具结果读回) → 循环
   → 直到 Final Answer(拿够信息，给最终答案)。
 关键工程点(面试必问)：
   ① 工具调用 Schema：每个工具有 name/description/参数，"大脑"靠这些决定调谁、传什么。
   ② 输出解析：从 LLM 文本里正则抠出 `Action: 工具[入参]`——生产用 function-calling 更稳。
   ③ 失败兜底：解析失败/未知工具/超最大步数，都要有兜底(把错误喂回 or 直接收尾)，不能死循环烧钱。
 为可确定演示、免下模型，这里的"大脑" brain() 用规则实现；换成真 LLM 只需把 brain 换成
 调 mlx-lm / OpenAI 的一行(生产就这么做)。
 跑：python3 补3_手写ReAct裸循环.py   （注：按用户要求本文件未在本机执行，仅作真实可跑代码）
================================================================================
"""
import re

# ==============================================================================
# ① 工具集：每个工具 = 函数 + name + description（大脑靠 description 选工具）
# ==============================================================================
KB = {"退货政策": "7 天无理由退货，需保持商品完好。", "运费": "满 99 包邮，否则 10 元。"}


def calculator(expr: str) -> str:
    """算一个算术表达式，如 '12*7'。"""
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))   # 仅演示；生产要用安全解析器
    except Exception as e:
        return f"计算出错:{e}"


def kb_search(query: str) -> str:
    """在知识库里按关键词查一条资料。"""
    for k, v in KB.items():
        if k in query:
            return v
    return "未找到相关资料"


TOOLS = {
    "calculator": {"fn": calculator, "desc": "算术计算，入参是表达式字符串"},
    "kb_search": {"fn": kb_search, "desc": "查知识库(退货政策/运费)，入参是问题关键词"},
}


# ==============================================================================
# ② "大脑"：给定历史(scratchpad)，产出下一步 —— 规则版(可替换成真 LLM)
# ==============================================================================
def brain(question: str, scratchpad: str) -> str:
    """返回一步文本：'Action: 工具[入参]' 或 'Final Answer: ...'。
    生产版把这里换成：prompt = ReAct 模板 + 工具清单 + scratchpad；return llm(prompt)。"""
    obs = scratchpad
    if "退货" in question and "Observation" not in obs:
        return "Thought: 这是政策问题，查知识库。\nAction: kb_search[退货政策]"
    if any(c in question for c in "0123456789") and "Observation" not in obs:
        expr = re.sub(r"[^0-9+\-*/().]", "", question)
        return f"Thought: 需要算数。\nAction: calculator[{expr}]"
    # 已经有观测 → 收尾
    last = obs.strip().splitlines()[-1] if obs.strip() else ""
    ans = last.replace("Observation:", "").strip()
    return f"Final Answer: {ans}"


# ==============================================================================
# ③ ReAct 主循环：解析 Action → 执行 → 拼 Observation → 循环；带兜底
# ==============================================================================
ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*?)\]", re.S)


def react(question: str, max_steps: int = 5) -> str:
    scratchpad = ""
    for step in range(max_steps):
        thought = brain(question, scratchpad)
        scratchpad += thought + "\n"
        if "Final Answer:" in thought:
            return thought.split("Final Answer:", 1)[1].strip()
        m = ACTION_RE.search(thought)
        if not m:                                          # 解析失败兜底：喂回提示
            scratchpad += "Observation: 未解析到合法 Action，请用 Action: 工具[入参] 格式。\n"
            continue
        name, arg = m.group(1), m.group(2).strip()
        tool = TOOLS.get(name)
        if not tool:                                       # 未知工具兜底
            scratchpad += f"Observation: 没有名为 {name} 的工具。\n"
            continue
        obs = tool["fn"](arg)                              # 真调工具
        scratchpad += f"Observation: {obs}\n"
    return "（超过最大步数，兜底转人工/给出当前最佳答案）"   # 超步数兜底，绝不死循环


def main():
    a1 = react("退货政策是什么？")     # -> "7 天无理由退货，需保持商品完好。"
    a2 = react("帮我算 12*7 等于多少")  # -> "84"
    assert "退货" in a1 and a2 == "84"
    print(f"✅ 补3 跑通：手写 ReAct(思考→行动→观测) → 退货问答='{a1[:12]}…' 计算='{a2}'（含解析/未知工具/超步数兜底）")
    # 面试：Q ReAct 循环在循环什么? A 思考→调工具→读观测→再思考，直到 Final Answer;
    #      Q 为什么要最大步数+兜底? A 防解析错/工具失败导致死循环烧 token; Q 生产怎么更稳? A 用 function-calling 替代文本解析。


if __name__ == "__main__":
    main()
