"""
 Context Course · Ch6 · 基础案例：最小 agent 步循环(纯 python,可跑)
 大脑(规则)→ 决定调工具 → 执行 → 把结果写回 → 收尾。跑：python3 本文件
"""
import re

def calc(expr):
    return str(eval(re.sub(r"[^0-9+\-*/().]", "", expr), {"__builtins__": {}}, {}))

task = "算 (10+5)*2"
# 一步决策:检测到算式 → 调 calc → 收尾
answer = calc(task)
print(f"✅ 任务:'{task}' → 调 calc 工具 → 答案 {answer}")
assert answer == "30"
