"""
================================================================================
 Context Course · Chapter 4 · 子智能体模式(Subagent Patterns)（学习笔记 · python 可跑,机制真实）
================================================================================
 一句话：多智能体工作流就那么几种"形状";子 agent 各自隔离上下文/职责,避免单 agent 上下文爆炸。
 本章用纯 python 把四种编排"做真"(不调 LLM,但并发/评估/隔离都是真的)：
   ① 主管-子工(supervisor→workers)：拆任务派给隔离上下文的子 agent,再汇总。
   ② 流水线(pipeline)：A 的输出即 B 的输入,顺序串。
   ③ 并行分治(parallel)：真·线程并发跑多个子 agent(ThreadPoolExecutor)再合并,不是假装并行。
   ④ 反馈循环(feedback)：真·打分评估,不过阈值就带批注重写,直到通过或触顶 —— 不是写死轮数。
 要点：子 agent 各自独立上下文 → 省 token、职责清晰、可复用;对照 实战练习/Agent进阶案例/案例2。
 跑：python3 chapter4_子智能体模式_学习笔记.py   （纯 python 真跑,无需模型)
================================================================================
"""
import time
from concurrent.futures import ThreadPoolExecutor


class SubAgent:
    """子 agent = 角色 + 独立上下文(自己的 memory)。隔离 = 各自一份,互不可见。"""

    def __init__(self, role):
        self.role = role
        self.memory = []

    def handle(self, task):
        self.memory.append(task)                     # 只写自己的上下文
        return f"[{self.role}] 处理:{task}"


# ① 主管-子工：拆解 → 分派给隔离子 agent → 汇总
def supervisor(task):
    plan = {"检索": f"查{task}资料", "写作": f"写{task}草稿", "审核": f"查{task}漏洞"}
    agents = {role: SubAgent(role) for role in plan}
    results = [agents[role].handle(sub) for role, sub in plan.items()]
    assert all(len(a.memory) == 1 for a in agents.values())   # 隔离:各自只记自己那条
    return results


# ② 流水线：上一阶段输出喂下一阶段
def pipeline(task, stages=("清洗", "分析", "总结")):
    data = task
    for s in stages:
        data = f"{s}({data})"
    return data


# ③ 并行分治：真·线程并发,而非假装并行
def _slow_worker(shard):
    time.sleep(0.05)                                 # 模拟每个子 agent 的 IO/推理耗时
    return f"分片[{shard}]完成"


def parallel(shards):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(shards)) as ex:
        results = list(ex.map(_slow_worker, shards))  # 并发提交,一起回来
    return results, time.perf_counter() - t0


# ④ 反馈循环：真·评估打分,不过阈值就带批注重写
def _review(draft):
    """审核子 agent:按"是否含数据 + 是否够长"打分(0~2),真实里可换成 LLM 评分。"""
    score = int("数据" in draft) + int(len(draft) >= 12)
    notes = None if score == 2 else ("补数据" if "数据" not in draft else "再写长点")
    return score, notes


def feedback_loop(topic, max_round=3):
    draft, notes = f"{topic}初稿", None
    for r in range(1, max_round + 1):
        if notes:                                    # 带上一轮批注重写
            draft = f"{topic}第{r}稿(含数据、且正文更详细)"
        score, notes = _review(draft)
        if score == 2:                               # 真的过审才停
            return draft, r, score
    return draft, max_round, score


def main():
    assert len(supervisor("报告")) == 3
    assert pipeline("原始数据") == "总结(分析(清洗(原始数据)))"

    results, elapsed = parallel(["A", "B", "C"])
    assert len(results) == 3 and elapsed < 0.12      # 真并发:远小于串行 3×0.05=0.15s

    draft, rounds, score = feedback_loop("文章")
    assert score == 2 and rounds >= 2                # 靠打分收敛,不是写死第 2 轮

    print("✅ Ch4 跑通：")
    print(f"   ① 主管派 3 个隔离子 agent")
    print(f"   ② 流水线:{pipeline('原始数据')}")
    print(f"   ③ 并行分治:3 子 agent 真并发耗时 {elapsed*1000:.0f}ms(串行需 ~150ms)")
    print(f"   ④ 反馈循环:第 {rounds} 轮打分 {score}/2 过审 → {draft}")
    # 面试:Q 为什么用子 agent? A 隔离上下文/职责、省 token、可复用; Q 编排形态? A 主管/流水线/并行/反馈。
    # Q 并行怎么真并发? A 线程池/异步; Q 反馈循环靠什么停? A 评估打分过阈值,不是写死轮数。


if __name__ == "__main__":
    main()
