"""
 Context Course · Ch6 · 基础案例：最小 agent 步循环(纯 python,机制真实,可跑)
 agent 内核 = 一个循环:大脑看历史 → 决定[调工具/收尾] → 执行 → 把观测写回历史 → 再来。
 这里用规则"大脑"替代 LLM(无需下模型),真的循环多步直到收尾,并带最大步数防死循环。
 跑：python3 本文件
"""
import re

def calc(expr):
    safe = re.sub(r"[^0-9+\-*/().]", "", expr)       # 只留算术字符,防注入
    return str(eval(safe, {"__builtins__": {}}, {}))


def step_loop(task, max_steps=5):
    history = [f"任务:{task}"]                         # 消息历史:每步都往里追加
    for _ in range(max_steps):                        # 步循环 + 最大步数(防死循环)
        last = history[-1]
        if last.startswith("工具结果:"):               # 已拿到结果 → 收尾
            return last[len("工具结果:"):], history
        if any(c.isdigit() for c in last):            # 决策:含算式 → 调 calc 工具
            history.append(f"调用 calc[{task}]")
            history.append(f"工具结果:{calc(task)}")   # 观测写回历史,下一轮据此收尾
        else:
            return "无法处理", history
    return "(超最大步数,兜底)", history


if __name__ == "__main__":
    answer, history = step_loop("算 (10+5)*2")
    print("走过的步:", " → ".join(history))
    print(f"✅ 迷你 agent 多步跑通 → 答案 {answer}")
    assert answer == "30"
