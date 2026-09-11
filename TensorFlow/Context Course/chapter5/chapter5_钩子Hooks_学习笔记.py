"""
================================================================================
 Context Course · Chapter 5 · 钩子(Hooks)（学习笔记 · 钩子系统 python 可跑）
================================================================================
 一句话：Hook 是 agent 生命周期确定性节点上的用户处理器——观测/拦截/注入,不靠模型自觉、不可绕过。
 本章讲(纯 python 实现一个迷你钩子系统,真跑)：
   ① 事件点：UserPromptSubmit(注入上下文) / PreToolUse(允许·拒绝·改参) / PostToolUse(后处理)。
   ② PreToolUse 可以"拦截"危险工具调用(如 rm -rf) → 返回 deny,工具不执行。
   ③ 为什么用 hook 而非让模型自觉：确定性护栏,模型绕不过去。
 要点：Hook = 在 agent 循环里插确定性护栏(对标 Claude Code 的 settings hooks)。
 跑：python3 chapter5_钩子Hooks_学习笔记.py   （纯 python 真跑)
================================================================================
"""

HOOKS = {"PreToolUse": [], "PostToolUse": []}


def register(event, fn):
    HOOKS[event].append(fn)


def run_tool_with_hooks(tool_name, args):
    """在"执行工具"前后跑钩子:PreToolUse 可拒绝/改参,PostToolUse 可后处理。"""
    for h in HOOKS["PreToolUse"]:
        decision = h(tool_name, args)
        if decision == "deny":
            return f"[已拦截] {tool_name} 被 PreToolUse 钩子拒绝"
    result = f"{tool_name}({args}) 执行结果"          # 模拟工具执行
    for h in HOOKS["PostToolUse"]:
        result = h(tool_name, result)
    return result


# 安全钩子：拦截危险命令
def block_dangerous(tool_name, args):
    return "deny" if tool_name == "shell" and "rm -rf" in str(args) else "allow"


# 后处理钩子：给结果加审计标记
def audit(tool_name, result):
    return result + "  [已审计]"


def main():
    register("PreToolUse", block_dangerous)
    register("PostToolUse", audit)
    ok = run_tool_with_hooks("calculator", "1+1")
    blocked = run_tool_with_hooks("shell", "rm -rf /")
    assert "已审计" in ok and "已拦截" in blocked
    print(f"✅ Ch5 跑通：正常调用→{ok}")
    print(f"           危险调用→{blocked}")
    # 面试：Q hook 解决什么? A 确定性观测/拦截/注入,不靠模型自觉; Q 典型事件? A Pre/PostToolUse、UserPromptSubmit。


if __name__ == "__main__":
    main()
