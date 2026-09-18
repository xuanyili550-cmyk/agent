"""
 Context Course · Ch4 · 基础案例：主管-子工编排(纯 python,机制真实,可跑)
 主管把任务拆给"检索/写作"两个子 agent;每个子 agent 有独立上下文(隔离),
 主管只拿它们的产出做汇总,看不到彼此的中间步骤 —— 这正是子智能体模式的核心。
 跑：python3 本文件
"""


class SubAgent:
    """一个子 agent = 角色 + 独立上下文(自己的 memory)。不同子 agent 互不可见对方历史 → 上下文隔离。"""

    def __init__(self, role, handler):
        self.role = role
        self.handler = handler          # 该角色真正干活的函数(这里用规则替代 LLM)
        self.memory = []                # 独立上下文:只装自己的步骤,不与别人共享

    def run(self, task):
        self.memory.append(f"接到任务:{task}")
        output = self.handler(task, self.memory)     # 干活时读写自己的 memory
        self.memory.append(f"产出:{output}")
        return output


def retrieve(task, memory):
    """检索子 agent:攒"资料"(真实里会去查库/搜索,这里返回结构化条目)。"""
    facts = [f"要点{i}(关于「{task}」)" for i in (1, 2, 3)]
    memory.append(f"检索到 {len(facts)} 条")
    return facts


def make_writer(facts):
    """写作子 agent:只吃检索给的资料,写成一段。用闭包把资料传进独立上下文。"""
    def write(task, memory):
        memory.append(f"基于 {len(facts)} 条资料动笔")
        return f"《{task}》正文：" + "；".join(facts)
    return write


def supervisor(goal):
    """主管:① 拆解 → ② 分派给各自隔离的子 agent → ③ 汇总它们的产出。"""
    researcher = SubAgent("检索", retrieve)
    facts = researcher.run(goal)                     # 子1 在自己上下文里跑

    writer = SubAgent("写作", make_writer(facts))
    draft = writer.run(goal)                         # 子2 拿子1 的产出,但看不到子1 的 memory

    assert researcher.memory != writer.memory        # 两份上下文互相独立 → 隔离
    return draft, researcher, writer


if __name__ == "__main__":
    draft, r, w = supervisor("周报")
    print("✅ 主管汇总产出 →", draft)
    print(f"   隔离证明:检索 agent 内部走了 {len(r.memory)} 步、写作 agent 走了 {len(w.memory)} 步,主管都看不到中间过程")
