"""
================================================================================
 Agent Course · Chapter 2 · 案例：派对策划 Agent「Alfred」（多工具，mlx-lm 真跑）
================================================================================
 场景(呼应课程 Alfred 管家)：Alfred 帮韦恩豪宅策划派对。用户提各种需求(定菜单/找供应商/
 出主题/算预算)，Agent 自己选合适的工具处理。多工具选择 + mlx-lm 真跑 + 规则兜底。
 (框架版：把这些函数用 smolagents @tool 包起来 + CodeAgent(InferenceClientModel) 即可，见学习笔记。)
 跑：python3 chapter2_框架案例.py
================================================================================
"""
import re

_M = {}


def suggest_menu(occasion):
    """按场合推荐菜单。"""
    return {"正式": "三道菜晚宴配红酒", "休闲": "披萨小吃饮料",
            "超级英雄": "高能量健康自助餐"}.get(occasion.strip(), "管家定制菜单")


def catering(_):
    """找评分最高的餐饮供应商(演示)。"""
    return "Gotham Catering Co.（评分 4.9）"


def party_theme(category):
    """按类别出派对主题创意。"""
    return {"反派化装": "哥谭恶棍舞会：宾客装扮成经典蝙蝠侠反派的神秘化装舞会。",
            "经典英雄": "正义联盟晚会：宾客装扮成 DC 英雄。"}.get(
        category.strip(), "主题：未找到，试试 反派化装/经典英雄")


def budget(expr):
    """算预算(算术表达式)。"""
    return "￥" + str(eval(expr, {"__builtins__": {}}, {})) if re.fullmatch(r"[\d\s+\-*/().]+", expr) else "非法"


TOOLS = {"suggest_menu": (suggest_menu, "按场合(正式/休闲/超级英雄)推荐菜单"),
         "catering": (catering, "找餐饮供应商"),
         "party_theme": (party_theme, "按类别(反派化装/经典英雄)出主题创意"),
         "budget": (budget, "算预算(算术表达式)")}
SPEC = "\n".join(f"  - {n}: {d}" for n, (_, d) in TOOLS.items())


def llm(p, n=60):
    if "m" not in _M:
        from mlx_lm import load
        _M["m"], _M["t"] = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    from mlx_lm import generate
    t = _M["t"].apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
    return generate(_M["m"], _M["t"], prompt=t, max_tokens=n, verbose=False).strip()


def _fallback(q):
    if re.search(r"\d.*[+\-*/]|预算|多少钱|花费", q):
        return "budget", re.sub(r"[^\d+\-*/().]", "", q) or "1000"
    if re.search(r"菜单|吃|餐", q):
        return "suggest_menu", "正式" if "正式" in q else ("超级英雄" if "英雄" in q else "休闲")
    if re.search(r"主题|化装|创意", q):
        return "party_theme", "反派化装" if "反派" in q or "化装" in q else "经典英雄"
    return "catering", q


def alfred(question):
    pick = llm(f"Alfred 的工具：\n{SPEC}\n需求：{question}\n只输出：工具名|参数", n=40)
    m = re.search(r"(\w+)\s*[|｜]\s*(.+)", pick)
    tool = m.group(1) if (m and m.group(1) in TOOLS) else None
    arg = m.group(2).strip() if m else ""
    if tool is None or not arg or (tool == "budget" and not re.search(r"\d", arg)):
        tool, arg = _fallback(question)
    obs = TOOLS[tool][0](arg)
    return {"需求": question, "工具": tool, "结果": obs}


if __name__ == "__main__":
    for q in ["给正式派对推荐个菜单", "帮我出个反派化装的主题创意",
              "找个餐饮供应商", "预算：50*8 张桌子"]:
        r = alfred(q)
        print("─" * 60)
        print(f"👤 {r['需求']}\n  🎩 Alfred 用 {r['工具']} → {r['结果']}")
    assert alfred("正式派对菜单")["工具"] == "suggest_menu"
    assert alfred("预算 100*5")["工具"] == "budget"
    print("\n✅ Chapter2 案例跑通：Alfred 多工具 Agent 按需求选 菜单/供应商/主题/预算 工具。")
