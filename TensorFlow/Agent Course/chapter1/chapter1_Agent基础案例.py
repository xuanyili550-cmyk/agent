"""
================================================================================
 Agent Course · Chapter 1 · 基础案例：真 ReAct 内核(工具注册表+步循环+路由,纯 python 可跑)
================================================================================
 Agent 内核三件套,这里都用真机制实现(纯 python,无需下模型):
   ① 工具注册表:@tool 从函数签名/docstring 自动生成 schema 并登记(同 function-calling 的工具清单);
   ② 步循环(ReAct):看历史 → 决策一个动作(调工具 or 收尾)→ 执行 → 观测写回 → 再来,带 max_steps 防死循环;
   ③ 路由:大脑按当前子任务选对应工具。复合问题("先算…再查…")会真的循环多步、连续调多个工具。
 大脑用规则实现(可靠、可复现);把真实 LLM 当大脑的写法见文末 🔴 函数(默认不执行)。
 跑：python3 chapter1_Agent基础案例.py
================================================================================
"""
import inspect
import re

REGISTRY = {}


def tool(fn):
    """迷你 function-calling 工具登记:从签名+docstring 自动生成 schema 并入表。"""
    sig = inspect.signature(fn)
    REGISTRY[fn.__name__] = {
        "fn": fn,
        "desc": (fn.__doc__ or "").strip().splitlines()[0],
        "params": list(sig.parameters),
    }
    return fn


@tool
def calculator(expr: str) -> str:
    """计算算术表达式(只允许数字与 + - * / ( ))。"""
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr or ""):
        return "非法表达式"
    try:
        return str(round(eval(expr, {"__builtins__": {}}, {}), 4))
    except Exception:
        return "计算失败"


@tool
def get_weather(city: str) -> str:
    """查询城市天气(演示数据)。"""
    return f"{city}多云转晴 22°C"


@tool
def unit_convert(text: str) -> str:
    """货币/长度/重量单位换算。"""
    m = re.search(r"([\d.]+)\s*(美元|美金|dollar|公里|km|磅|lb)", text, re.I)
    if not m:
        return "无法识别换算"
    n, unit = float(m.group(1)), m.group(2).lower()
    table = {"美元": ("人民币", 7.2), "美金": ("人民币", 7.2), "dollar": ("人民币", 7.2),
             "公里": ("英里", 0.621), "km": ("英里", 0.621), "磅": ("千克", 0.454), "lb": ("千克", 0.454)}
    to, rate = table[unit]
    return f"{n}{unit}≈{round(n * rate, 2)}{to}"


def route(clause):
    """路由(规则大脑):把一个子任务映射到 工具名+参数。真实系统这里换成 LLM 的 function calling。"""
    if re.search(r"美元|美金|dollar|公里|km|磅|lb|换算|等于多少", clause, re.I):
        return "unit_convert", clause
    if re.search(r"\d.*[+\-*/×x].*\d|算|乘|加|减|除", clause):
        expr = (clause.replace("乘以", "*").replace("乘", "*").replace("×", "*").replace("x", "*")
                .replace("加", "+").replace("减", "-").replace("除以", "/").replace("除", "/"))
        return "calculator", re.sub(r"[^\d+\-*/().]", "", expr) or "0"
    return "get_weather", re.sub(r"(的?天气|怎么样|如何|查|一下|今天|请问|帮我)", "", clause).strip() or "北京"


def agent(question, max_steps=8):
    """真步循环:把问题拆成子任务队列,每轮取一个 → 路由 → 调工具 → 观测入 trace,直到收尾或触顶。"""
    todo = [c for c in re.split(r"[,，；;]|再|然后|以及|and", question) if c.strip()]
    trace, obs_list, steps = [], [], 0
    while steps < max_steps:
        steps += 1
        if not todo:                                   # 无待办 → 决策"收尾"动作
            trace.append(("finish", "；".join(obs_list)))
            return {"问题": question, "步数": steps, "trace": trace,
                    "调用工具": [t for t, *_ in trace if t != "finish"],
                    "回答": "；".join(obs_list)}
        clause = todo.pop(0)                           # 取当前子任务
        tool_name, arg = route(clause)                 # ③ 路由
        obs = REGISTRY[tool_name]["fn"](arg)           # ① 经注册表真调用
        trace.append((tool_name, arg, obs))            # ② 观测写回历史(下轮据此继续/收尾)
        obs_list.append(obs)
    return {"问题": question, "步数": steps, "trace": trace, "回答": "(触发 max_steps 兜底)"}


if __name__ == "__main__":
    print("已注册工具:", {n: t["desc"] for n, t in REGISTRY.items()})
    for q in ["帮我算 (88+12)*3", "上海天气怎么样", "100 美元等于多少人民币",
              "先算 (88+12)*3 再查北京天气 然后 5 公里是多少英里"]:
        r = agent(q)
        print("─" * 64)
        print(f"👤 {q}")
        for row in r["trace"]:
            print(f"   🔧 {row}" if row[0] != "finish" else f"   ✅ 收尾: {row[1]}")

    # 自检:单任务路由正确 + 复合任务真的多步、连调 3 个工具
    assert agent("算 6*7")["调用工具"] == ["calculator"]
    assert agent("100美元换人民币")["调用工具"] == ["unit_convert"]
    assert agent("北京天气")["调用工具"] == ["get_weather"]
    multi = agent("先算 (88+12)*3 再查北京天气 然后 5 公里是多少英里")
    assert multi["调用工具"] == ["calculator", "get_weather", "unit_convert"]   # 一次任务连调 3 工具
    assert multi["步数"] == 4                                                    # 3 次工具 + 1 次收尾
    assert calculator("(88+12)*3") == "300"
    print("\n✅ 案例跑通:真工具注册表 + ReAct 步循环 + 路由;复合问题在一次任务里自动多步调多工具。")
    print("面试 Q：ReAct 循环为何一定要 max_steps 上限？"
          " A：大脑可能反复决策调工具却不收尾(观测不满足停机条件)导致死循环/烧钱,"
          "步数上限是硬性兜底,超限即中止并返回当前最优/降级结果。")


# 🔴 用真实 LLM 当大脑(需 mlx-lm + 下模型;默认不执行,仅参考) ————————————————
def llm_route(question, spec):  # pragma: no cover
    from mlx_lm import load, generate
    model, tok = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": f"可用工具:\n{spec}\n问题:{question}\n只输出一行:工具名|参数"}],
        add_generation_prompt=True)
    return generate(model, tok, prompt=prompt, max_tokens=40, verbose=False).strip()
