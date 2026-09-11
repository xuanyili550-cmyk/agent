"""
================================================================================
 Agent Course · Chapter 1 · Agent 基础（学习笔记，可运行真代码）
================================================================================
 一句话：Agent = LLM(大脑) + Tools(手脚) + ReAct 循环(推理→行动→观察)。本章讲清这三件，
        并用本地 LLM(mlx-lm) 手写一个能跑的 ReAct 小循环(看清 Agent 内部到底怎么转)。
 内容：
   ① 为什么需要 Agent：LLM 只会“说”，不会“做/查实时信息”；给它工具 + 循环，它就能行动。
   ② Tool 三要素：name + description + 参数(类型) —— LLM 靠这些“读懂”工具、决定调不调、传什么。
   ③ ReAct 循环：Thought(想) → Action(调工具) → Observation(看结果) → …→ Final Answer。
   ④ 模型后端：本地(mlx-lm/Transformers) / HF(InferenceClient) / vLLM(OpenAI 兼容)。
 本机：用 mlx-lm(Qwen2.5-0.5B) 当大脑真跑；小模型可靠性有限，加规则兜底保证能看到完整循环。
 跑：python3 chapter1_Agent基础_学习笔记.py
================================================================================
"""
import re

_M = {}


# ==============================================================================
# ② Tool：把普通函数登记成“工具”(name/description/参数 = LLM 的说明书)
# ==============================================================================
class Tool:
    def __init__(self, name, description, func, args):
        self.name, self.description, self.func, self.args = name, description, func, args

    def spec(self):                                    # 给 LLM 看的说明
        return f"{self.name}({', '.join(self.args)}): {self.description}"

    def __call__(self, arg):
        return self.func(arg)


def _calculator(expr):
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr or ""):
        return "非法表达式"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return "计算失败"


def _weather(city):
    return f"{city}：晴，25°C（演示数据）"


TOOLS = {
    "calculator": Tool("calculator", "计算算术表达式(只含数字和 +-*/())", _calculator, ["expr"]),
    "get_weather": Tool("get_weather", "查询城市天气", _weather, ["city"]),
}


# ==============================================================================
# ④ 模型后端：本地 mlx-lm 当大脑
# ==============================================================================
def llm(prompt, max_tokens=60):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    text = _M["t"].apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return generate(_M["m"], _M["t"], prompt=text, max_tokens=max_tokens, verbose=False).strip()


# ==============================================================================
# ③ 手写 ReAct 循环：Thought → Action → Observation → Final Answer
# ==============================================================================
def _to_expr(q):
    """中文算式 → 符号算式：乘以/乘→*、加→+、减→-、除以/除→/，再只留算术字符。"""
    q = (q.replace("乘以", "*").replace("乘", "*").replace("加", "+")
         .replace("减", "-").replace("除以", "/").replace("除", "/"))
    return re.sub(r"[^\d+\-*/().]", "", q) or "1+1"


def _route_fallback(q):
    """规则兜底：小模型选错时按关键词兜底(保证能演示完整循环)。"""
    if re.search(r"\d.*[+\-*/].*\d|算|乘|加|减|除|等于", q):
        return "calculator", _to_expr(q)
    return "get_weather", re.sub(r"(天气|的|怎么样|如何|查|一下)", "", q).strip() or "北京"


def react(question, max_steps=2):
    tool_specs = "\n".join("  - " + t.spec() for t in TOOLS.values())
    print(f"\n❓ 问题：{question}")
    # —— Thought + Action：让 LLM 决定调哪个工具、传什么参数 ——
    pick = llm(f"你能用这些工具：\n{tool_specs}\n"
               f"用户问题：{question}\n只输出一行：工具名|参数", max_tokens=40)
    m = re.search(r"(\w+)\s*[|｜]\s*(.+)", pick)
    tool = m.group(1) if (m and m.group(1) in TOOLS) else None
    arg = m.group(2).strip() if m else ""
    if tool is None or not arg:
        tool, arg = _route_fallback(question)          # 规则兜底
        print(f"  🧠 Thought：(LLM 提议不可靠，规则兜底) 该用 {tool}")
    else:
        print(f"  🧠 Thought：该用 {tool} 处理")
    if tool == "calculator":
        arg = _to_expr(question)                       # 算术参数一律用规则抽取，保证算对(小模型算式不可靠)
    # —— Observation：真正执行工具 ——
    obs = TOOLS[tool](arg)
    print(f"  🔧 Action：{tool}({arg})  →  👀 Observation：{obs}")
    # —— Final Answer：让 LLM 根据观察给最终回答 ——
    final = llm(f"用户问：{question}\n工具返回：{obs}\n用一句话自然回答：", max_tokens=60)
    print(f"  ✅ Final Answer：{final}")
    return final, tool, obs


if __name__ == "__main__":
    print("=" * 70)
    print("① Agent = LLM(大脑) + Tools(手脚) + ReAct 循环(推理→行动→观察)")
    print("② 可用工具(LLM 靠这些说明书决定调不调)：")
    for t in TOOLS.values():
        print("   -", t.spec())
    print("=" * 70)
    print("③ 手写 ReAct 循环(mlx-lm 真跑)：")
    react("帮我算一下 128 乘以 39")
    react("北京今天天气怎么样")
    # 自检：算术走计算器、天气走天气
    assert react("算 12+8")[1] == "calculator"
    assert react("上海天气")[1] == "get_weather"
    print("\n✅ Chapter1 跑通：Tool 定义 + 手写 ReAct(Thought→Action→Observation→Answer)。")
    print("面试：Q Agent 三要素? Q ReAct 三步? Q LLM 靠什么决定调哪个工具?(name+description+参数)")
