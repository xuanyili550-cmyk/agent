"""
================================================================================
 Context Course · Chapter 4 · 子智能体模式(Subagent Patterns)（学习笔记 · python 可跑）
================================================================================
 一句话：多智能体工作流就那么几种"形状";子 agent 隔离上下文/职责,避免单 agent 上下文爆炸。
 本章讲(用纯 python 模拟编排,不调真 LLM,看清数据怎么流)：
   ① 主管-子工(supervisor→workers)：主管拆任务派给专职子 agent,再汇总。
   ② 流水线(pipeline)：A 的输出是 B 的输入,顺序串。
   ③ 并行分治(parallel)：同任务并行跑多个子 agent 再合并。
   ④ 反馈循环(feedback)：审核子 agent 不合格就打回重做。
 要点：子 agent 各自独立上下文 → 省 token、职责清晰、可复用;对照 实战练习/Agent进阶案例/案例2。
 跑：python3 chapter4_子智能体模式_学习笔记.py   （纯 python 模拟,无需模型)
================================================================================
"""

def worker(role, task):                       # 子 agent(这里用规则替代 LLM)
    return f"[{role}] 处理:{task}"


def supervisor(task):
    """① 主管-子工：拆解 → 分派 → 汇总。"""
    subtasks = {"检索": "查资料", "写作": "写草稿", "审核": "查漏"}
    results = [worker(r, f"{task}的{t}") for r, t in subtasks.items()]
    return results


def pipeline(task, stages=("清洗", "分析", "总结")):
    """② 流水线：上一阶段输出喂下一阶段。"""
    data = task
    for s in stages:
        data = f"{s}({data})"
    return data


def feedback_loop(task, max_round=3):
    """④ 反馈循环：写作→审核,不过就重写(这里第 2 轮通过)。"""
    for r in range(1, max_round + 1):
        draft = f"草稿v{r}"
        if r >= 2:                            # 模拟审核通过
            return draft, r
    return draft, max_round


def main():
    assert len(supervisor("报告")) == 3
    assert pipeline("原始数据") == "总结(分析(清洗(原始数据)))"
    draft, rounds = feedback_loop("文章")
    assert rounds == 2
    print(f"✅ Ch4 跑通：主管派 3 子agent · 流水线串联 · 反馈循环第 {rounds} 轮通过")
    # 面试：Q 为什么用子 agent? A 隔离上下文/职责、省 token、可复用; Q 编排形态? A 主管/流水线/并行/反馈。


if __name__ == "__main__":
    main()
