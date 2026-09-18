"""
================================================================================
 Context Course · Chapter 5 · 钩子(Hooks)（学习笔记 · 钩子系统 python 可跑,机制真实）
================================================================================
 一句话：Hook 是 agent 生命周期确定性节点上的用户处理器——观测/拦截/改写/注入,不靠模型自觉、不可绕过。
 本章用纯 python 实现一个迷你钩子系统(带事件分发 + 匹配器),真跑：
   ① 三类事件:UserPromptSubmit(往提示注入上下文) / PreToolUse(allow·deny·改 args) / PostToolUse(后处理结果)。
   ② 匹配器(matcher)：钩子只在工具名匹配时触发(对标 Claude Code settings 的 matcher)。
   ③ 为什么用 hook 而非让模型自觉：确定性护栏,模型绕不过去。
 要点：Hook = 在 agent 循环里插确定性护栏(对标 Claude Code 的 settings hooks)。
 跑：python3 chapter5_钩子Hooks_学习笔记.py   （纯 python 真跑)
================================================================================
"""
import re

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": []}


def register(event, fn, matcher=".*"):
    """注册钩子;matcher 是工具名正则,只有 Pre/PostToolUse 用得上(对标 settings 的 matcher)。"""
    HOOKS[event].append((re.compile(matcher), fn))


def emit_prompt(prompt):
    """① UserPromptSubmit:每个钩子可往用户提示追加上下文(如项目规则、时间)。"""
    for _, fn in HOOKS["UserPromptSubmit"]:
        prompt = fn(prompt)
    return prompt


def run_tool_with_hooks(tool_name, args):
    """② PreToolUse 可 deny/改 args → 执行 → PostToolUse 后处理;matcher 不中则跳过该钩子。"""
    for pat, fn in HOOKS["PreToolUse"]:
        if not pat.fullmatch(tool_name):
            continue
        out = fn(tool_name, args)
        if out.get("decision") == "deny":
            return f"[已拦截] {tool_name}:{out.get('reason', '')}"
        args = out.get("args", args)                 # 钩子改写后的参数继续往下
    result = f"{tool_name}({args}) 执行结果"          # 模拟工具执行
    for pat, fn in HOOKS["PostToolUse"]:
        if pat.fullmatch(tool_name):
            result = fn(tool_name, result)
    return result


# —— 三个钩子 ——
def inject_rules(prompt):                             # UserPromptSubmit:注入项目规则
    return prompt + "\n[上下文] 规则:命令需可回滚。"


def guard_shell(tool_name, args):                     # PreToolUse:拦危险 + 给 rm 补 --dry-run
    if "rm -rf" in str(args):
        return {"decision": "deny", "reason": "禁止 rm -rf"}
    if str(args).startswith("rm ") and "--dry-run" not in str(args):
        return {"decision": "allow", "args": args + " --dry-run"}
    return {"decision": "allow"}


def audit(tool_name, result):                         # PostToolUse:加审计标记
    return result + "  [已审计]"


def main():
    register("UserPromptSubmit", inject_rules)
    register("PreToolUse", guard_shell, matcher="shell")     # 只管 shell 工具
    register("PostToolUse", audit, matcher="shell")

    prompt = emit_prompt("帮我清理日志")
    assert "[上下文]" in prompt                        # 提示被注入

    ok = run_tool_with_hooks("shell", "rm old.log")   # 被 guard_shell 改写成带 --dry-run
    blocked = run_tool_with_hooks("shell", "rm -rf /")
    skipped = run_tool_with_hooks("calculator", "1+1") # 不匹配 shell → 两个钩子都跳过

    assert "--dry-run" in ok and "已审计" in ok
    assert "已拦截" in blocked
    assert "已审计" not in skipped                     # matcher 不中,PostToolUse 没跑

    print("✅ Ch5 跑通:")
    print(f"   ① 注入后提示 →{prompt!r}")
    print(f"   ② 改写 →{ok}")
    print(f"   ② 拦截 →{blocked}")
    print(f"   ② 不匹配跳过 →{skipped}")
    # 面试:Q hook 解决什么? A 确定性观测/拦截/改写/注入,不靠模型自觉; Q 典型事件? A UserPromptSubmit、Pre/PostToolUse; Q matcher 作用? A 按工具名过滤,钩子只在该工具触发。


if __name__ == "__main__":
    main()
