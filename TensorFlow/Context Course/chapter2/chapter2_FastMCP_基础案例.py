"""
 Context Course · Ch2 · 基础案例：最小 FastMCP 服务器（🟡 需 pip install "mcp[cli]"）
 跑：python3 本文件 （会以 stdio 方式起 MCP 服务,供 Claude Desktop 等 Host 连接）
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


if __name__ == "__main__":
    mcp.run()   # 起服务;schema 由类型注解+docstring 自动生成
