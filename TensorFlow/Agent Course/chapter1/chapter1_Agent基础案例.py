"""
================================================================================
 Agent Course · Chapter 1 · 案例：智能个人助理（多工具 ReAct，批处理，可运行）
================================================================================
 场景：一个个人助理 Agent，用户随口问什么(算账/天气/换算/查资料)，Agent 自己选工具处理。
 复用 Chapter1 学习笔记的 ReAct 思路，加更多工具 + 一批真实请求，看 Agent 怎么按问题选不同工具。
 本机：mlx-lm(Qwen2.5-0.5B) 当大脑 + 规则兜底(小模型可靠性有限)。
 跑：python3 chapter1_Agent基础案例.py
================================================================================
"""
import re

_M = {}


def _calc(expr):
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr or ""):
        return "非法表达式"
    try:
        return str(round(eval(expr, {"__builtins__": {}}, {}), 4))
    except Exception:
        return "计算失败"


def _weather(city):
    return f"{city}：多云转晴，22°C，适合出行（演示数据）"


def _convert(text):
    """简单单位换算：支持 '1 美元' / '10 公里'(演示汇率/换算)。"""
    m = re.search(r"([\d.]+)\s*(美元|美金|dollar|公里|km|磅|lb)", text, re.I)
    if not m:
        return "无法识别换算"
    n, unit = float(m.group(1)), m.group(2).lower()
    table = {"美元": ("人民币", 7.2), "美金": ("人民币", 7.2), "dollar": ("人民币", 7.2),
             "公里": ("英里", 0.621), "km": ("英里", 0.621), "磅": ("千克", 0.454), "lb": ("千克", 0.454)}
    to, rate = table[unit]
    return f"{n}{unit} ≈ {round(n * rate, 2)} {to}"


TOOLS = {
    "calculator": (_calc, "计算算术表达式"),
    "get_weather": (_weather, "查询城市天气"),
    "unit_convert": (_convert, "货币/长度/重量单位换算"),
}
SPEC = "\n".join(f"  - {n}: {d}" for n, (_, d) in TOOLS.items())


def llm(prompt, max_tokens=60):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    text = _M["t"].apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    return generate(_M["m"], _M["t"], prompt=text, max_tokens=max_tokens, verbose=False).strip()


def _fallback(q):
    if re.search(r"美元|美金|公里|km|磅|lb|换算|等于多少(人民币|英里|千克)", q, re.I):
        return "unit_convert", q
    if re.search(r"\d.*[+\-*/].*\d|算|乘|加|减|除", q):
        expr = (q.replace("乘以", "*").replace("乘", "*").replace("加", "+")
                .replace("减", "-").replace("除以", "/").replace("除", "/"))
        return "calculator", re.sub(r"[^\d+\-*/().]", "", expr) or "1+1"
    return "get_weather", re.sub(r"(天气|的|怎么样|如何|查|一下|今天)", "", q).strip() or "北京"


def assist(question):
    # 小模型选工具不可靠：以规则路由为准，LLM 提议仅在【与规则一致且参数合法】时采纳其参数。
    # (生产换大模型可让 LLM 主导 function calling，去掉此兜底。)
    pick = llm(f"可用工具：\n{SPEC}\n用户问题：{question}\n只输出一行：工具名|参数", max_tokens=40)
    fb_tool, fb_arg = _fallback(question)
    m = re.search(r"(\w+)\s*[|｜]\s*(.+)", pick)
    tool, arg = fb_tool, fb_arg
    if m and m.group(1) == fb_tool:                 # LLM 与规则选了同一个工具 → 采纳 LLM 的参数
        cand = m.group(2).strip()
        if not (fb_tool == "calculator" and not re.fullmatch(r"[\d\s+\-*/().]+", cand)):
            arg = cand or fb_arg
    obs = TOOLS[tool][0](arg)
    final = llm(f"用户问：{question}\n工具结果：{obs}\n用一句话回答：", max_tokens=50)
    return {"问题": question, "选用工具": tool, "工具结果": obs, "回答": final}


if __name__ == "__main__":
    requests = ["帮我算 (88+12)*3", "上海今天天气怎么样", "100 美元等于多少人民币", "5 公里是多少英里"]
    for q in requests:
        r = assist(q)
        print("─" * 60)
        print(f"👤 {r['问题']}\n  🔧 选用 {r['选用工具']} → {r['工具结果']}\n  🤖 {r['回答']}")
    # 自检：不同问题选对不同工具
    assert assist("算 6*7")["选用工具"] == "calculator"
    assert assist("100美元换人民币")["选用工具"] == "unit_convert"
    assert assist("北京天气")["选用工具"] == "get_weather"
    print("\n✅ Chapter1 案例跑通：个人助理 Agent 按问题自动选 计算/天气/换算 工具。")
