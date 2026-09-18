"""
 Context Course · Ch5 · 基础案例：PreToolUse 钩子拦截/改写工具调用(纯 python,机制真实,可跑)
 Hook = 在 agent 生命周期确定性节点上的用户处理器:能拦截(deny)、能改参(修 args),不靠模型自觉。
 这里注册两个 PreToolUse 钩子并真的串起来跑:一个拦危险命令,一个给命令自动补安全标志。
 跑：python3 本文件
"""

def block_rm(tool, args):
    """危险命令直接拦截 → 返回 deny,工具不会执行。"""
    if tool == "shell" and ("rm -rf" in args or "sudo" in args):
        return {"decision": "deny", "reason": "危险命令"}
    return {"decision": "allow"}


def add_dry_run(tool, args):
    """非拦截钩子:给删除类命令自动加 --dry-run(演示"改参"能力,不止是放行/拒绝)。"""
    if tool == "shell" and args.startswith("rm ") and "--dry-run" not in args:
        return {"decision": "allow", "args": args + " --dry-run"}
    return {"decision": "allow"}


PRE_TOOL_USE = [block_rm, add_dry_run]


def run_pre_hooks(tool, args):
    """按注册顺序跑钩子:任一 deny 就拦下;否则每个钩子都有机会改写 args。"""
    for hook in PRE_TOOL_USE:
        out = hook(tool, args)
        if out["decision"] == "deny":
            return "deny", args, out["reason"]
        args = out.get("args", args)                 # 钩子可覆盖 args,改写后继续往下
    return "allow", args, None


if __name__ == "__main__":
    cases = [("shell", "ls -l"), ("shell", "rm old.log"), ("shell", "rm -rf /"), ("read_file", "a.txt")]
    for tool, args in cases:
        decision, final_args, reason = run_pre_hooks(tool, args)
        if decision == "deny":
            print(f"🛑 拦截 {tool}({args}) —— {reason}")
        else:
            changed = "  (已被钩子改写)" if final_args != args else ""
            print(f"✅ 放行 {tool}({final_args}){changed}")
