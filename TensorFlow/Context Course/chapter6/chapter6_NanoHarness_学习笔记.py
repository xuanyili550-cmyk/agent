"""
================================================================================
 Context Course · Chapter 6 · Nano Harness：读懂 agent 循环底层（学习笔记 · 迷你 agent 可跑）
================================================================================
 一句话：主流框架把 agent 循环藏起来;这里用纯 python 写一个最小 agent,把"步循环+工具+历史+错误处理"摊开。
 本章讲(纯 python,用规则"大脑"替代真 LLM → 无需下模型,真跑)：
   ① 系统提示 + 消息历史(messages)累积上下文。
   ② 步循环(step loop)：大脑决定 调工具 or 收尾 → 执行 → 把观测写回历史 → 再来。
   ③ 工具执行 + 错误处理 + 最大步数(防死循环)。
 要点：所有 agent 框架内核都是这个循环;看懂它再用 smolagents/LangGraph 就通透。对照 实战练习/生产工程补齐/补3。
 跑：python3 chapter6_NanoHarness_学习笔记.py   （纯 python 真跑,无需模型)
================================================================================
"""
import re

TOOLS = {"calc": lambda e: str(eval(e, {"__builtins__": {}}, {}))}   # 仅演示


def brain(history):
    """迷你"大脑"(规则版,可换成调 LLM)：看最后一条,决定下一步文本。"""
    last = history[-1]["content"]
    if last.startswith("任务:") and any(c.isdigit() for c in last):
        expr = re.sub(r"[^0-9+\-*/().]", "", last)
        return f"Action: calc[{expr}]"
    if history[-1]["role"] == "tool":            # 拿到工具结果 → 收尾
        return f"Final: 答案是 {last}"
    return "Final: 我无法处理"


ACTION = re.compile(r"Action:\s*(\w+)\[(.*?)\]")


def run(task, max_steps=5):
    history = [{"role": "system", "content": "你是带工具的 agent"},
               {"role": "user", "content": f"任务:{task}"}]
    for _ in range(max_steps):                    # ③ 步循环 + 最大步数
        thought = brain(history)
        history.append({"role": "assistant", "content": thought})
        if thought.startswith("Final:"):
            return thought[len("Final:"):].strip()
        m = ACTION.search(thought)
        if not m:
            history.append({"role": "tool", "content": "解析失败"}); continue
        name, arg = m.group(1), m.group(2)
        try:                                      # ③ 工具执行 + 错误处理
            obs = TOOLS[name](arg) if name in TOOLS else f"无工具 {name}"
        except Exception as e:
            obs = f"工具出错:{e}"
        history.append({"role": "tool", "content": obs})
    return "(超最大步数,兜底)"


def main():
    ans = run("帮我算 12*(3+4)")
    assert ans == "答案是 84"
    print(f"✅ Ch6 跑通：迷你 agent 走完 [大脑→Action→工具→观测→收尾] → {ans}")
    # 面试：Q agent 内核是什么? A 步循环:决策→调工具→写回观测→再决策; Q 怎么防死循环? A 最大步数+错误兜底。


if __name__ == "__main__":
    main()
