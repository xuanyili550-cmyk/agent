"""
================================================================================
 Context Course · Chapter 6 · Nano Harness：读懂 agent 循环底层（学习笔记 · 多工具多步 agent 可跑）
================================================================================
 一句话：主流框架把 agent 循环藏起来;这里用纯 python 写最小 agent,把"步循环+多工具+历史+错误处理"摊开。
 本章用纯 python(规则"大脑"替代真 LLM,无需下模型)跑一个需要连调两个工具的多步任务：
   ① 系统提示 + 消息历史(messages)累积上下文,每步都读全历史再决策。
   ② 步循环(ReAct)：大脑决定 Action(调工具) 或 Final(收尾) → 执行 → 观测写回历史 → 再来。
   ③ 多工具注册表 + 工具执行 + 错误处理 + 最大步数(防死循环)。
 要点：所有 agent 框架内核都是这个循环;看懂它再用 smolagents/LangGraph 就通透。对照 实战练习/生产工程补齐/补3。
 跑：python3 chapter6_NanoHarness_学习笔记.py   （纯 python 真跑,无需模型)
================================================================================
"""
import re

PRICES = {"书": 30, "笔": 5}
TOOLS = {                                             # ③ 多工具注册表
    "price": lambda item: str(PRICES.get(item.strip(), 0)),
    "calc": lambda e: str(eval(e, {"__builtins__": {}}, {})),
}
ACTION = re.compile(r"Action:\s*(\w+)\[(.*?)\]")


def brain(history):
    """规则"大脑"(可换成调 LLM):读全历史决定下一步。任务=买 N 件某物 → 先查单价、再算总价、最后收尾。"""
    task = history[1]["content"]
    tool_results = [m["content"] for m in history if m["role"] == "tool"]
    item = next((k for k in PRICES if k in task), "")
    qty = re.search(r"(\d+)", task)
    if not tool_results:                              # 第一步:先查单价
        return f"Action: price[{item}]"
    if len(tool_results) == 1:                        # 拿到单价:算 单价×数量
        return f"Action: calc[{tool_results[0]}*{qty.group(1) if qty else 1}]"
    return f"Final: 总价是 {tool_results[-1]}"          # 拿到总价:收尾


def run(task, max_steps=6):
    history = [{"role": "system", "content": "你是带工具的 agent"},
               {"role": "user", "content": task}]
    for _ in range(max_steps):                        # ③ 步循环 + 最大步数
        thought = brain(history)
        history.append({"role": "assistant", "content": thought})
        if thought.startswith("Final:"):
            return thought[len("Final:"):].strip(), history
        m = ACTION.search(thought)
        if not m:
            history.append({"role": "tool", "content": "解析失败"}); continue
        name, arg = m.group(1), m.group(2)
        try:                                          # ③ 工具执行 + 错误处理
            obs = TOOLS[name](arg) if name in TOOLS else f"无工具 {name}"
        except Exception as e:
            obs = f"工具出错:{e}"
        history.append({"role": "tool", "content": obs})   # 观测写回,下一轮据此决策
    return "(超最大步数,兜底)", history


def main():
    ans, history = run("帮我买 3 件书,算总价")
    tool_calls = [m["content"] for m in history
                  if m["role"] == "assistant" and m["content"].startswith("Action")]
    assert ans == "总价是 90"                          # 30×3,连调 price+calc 两个工具
    assert len(tool_calls) == 2                        # 真的多步:两次工具调用
    print(f"✅ Ch6 跑通:迷你 agent 走 [查单价→算总价] 多步 → {ans}")
    print("   工具调用序列:", " → ".join(tool_calls))
    # 面试:Q agent 内核是什么? A 步循环:决策→调工具→写回观测→再决策; Q 多步怎么串? A 上一步观测进历史、下一步据此再决策; Q 防死循环? A 最大步数+错误兜底。


if __name__ == "__main__":
    main()
