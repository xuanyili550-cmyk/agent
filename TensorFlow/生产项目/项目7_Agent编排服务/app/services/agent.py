"""Agent:手写 ReAct 循环(规则大脑替代 LLM → 离线可跑) + 工具;backend=real 时接真 LLM。"""
import re
from ..core.exceptions import ValidationError

_SAFE = re.compile(r"[^0-9+*/().-]")          # 只保留数字与算术符(- 放末尾免转义)


def _calc(e):
    return str(eval(_SAFE.sub("", e), {"__builtins__": {}}, {}))


TOOLS = {"calc": _calc}


def run(query: str, max_steps: int = 5):
    if not query:
        raise ValidationError("query 不能为空")
    steps = []
    if any(c.isdigit() for c in query):          # 含数字 → 调计算器工具
        expr = _SAFE.sub("", query)
        steps.append({"action": "calc", "input": expr})
        obs = TOOLS["calc"](expr)
        steps.append({"observation": obs})
        return {"answer": f"计算结果:{obs}", "steps": steps, "route": "tool"}
    return {"answer": "我可以调用计算器等工具,试试问我算式。", "steps": steps, "route": "direct"}
