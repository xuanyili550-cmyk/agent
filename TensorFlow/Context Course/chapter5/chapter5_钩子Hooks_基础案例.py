"""
 Context Course · Ch5 · 基础案例：PreToolUse 钩子拦截危险命令(纯 python,可跑)
 跑：python3 本文件
"""

def pre_tool_use(tool, args):
    if tool == "shell" and ("rm -rf" in args or "sudo" in args):
        return "deny"       # 危险 → 拦截,工具不会执行
    return "allow"


for tool, args in [("shell", "ls -l"), ("shell", "rm -rf /"), ("read_file", "a.txt")]:
    decision = pre_tool_use(tool, args)
    print(f"{'🛑 拦截' if decision=='deny' else '✅ 放行'}: {tool}({args})")
