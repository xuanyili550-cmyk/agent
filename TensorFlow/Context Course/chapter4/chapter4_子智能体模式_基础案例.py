"""
 Context Course · Ch4 · 基础案例：主管-子工编排(纯 python,可跑)
 主管把任务拆给"检索/写作"两个子 agent,再汇总。跑：python3 本文件
"""

def sub_agent(role, task):
    return f"{role}完成:{task}"


def supervisor(goal):
    plan = [("检索agent", f"查{goal}的资料"), ("写作agent", f"根据资料写{goal}")]
    results = [sub_agent(role, t) for role, t in plan]
    return "汇总 → " + " | ".join(results)


print("✅", supervisor("周报"))
