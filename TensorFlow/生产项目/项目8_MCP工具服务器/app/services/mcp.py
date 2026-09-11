"""MCP 风格工具服务器:工具带契约(name/description/inputSchema),支持 list + call。"""
import inspect
from ..core.exceptions import NotFoundError, ValidationError
def add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b
def upper(text: str) -> str:
    """转大写。"""
    return text.upper()
_TOOLS = {}
def _register(fn):
    sig = inspect.signature(fn)
    _TOOLS[fn.__name__] = {"fn": fn, "name": fn.__name__,
        "description": (fn.__doc__ or "").strip(),
        "inputSchema": {p: (par.annotation.__name__ if par.annotation is not inspect._empty else "any") for p, par in sig.parameters.items()}}
for _f in (add, upper): _register(_f)
def list_tools():
    return [{"name": t["name"], "description": t["description"], "inputSchema": t["inputSchema"]} for t in _TOOLS.values()]
def call_tool(name: str, args: dict):
    if name not in _TOOLS: raise NotFoundError(f"无工具 {name}")
    try:
        return {"result": _TOOLS[name]["fn"](**(args or {}))}
    except TypeError as e:
        raise ValidationError(f"参数错误:{e}")
