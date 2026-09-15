"""workflows 包：故事引擎的 LangGraph 流水线入口。

对外只导出流水线相关的公共符号（图构建函数、状态类型、agent 容器、角色提示词），
校验逻辑（validation.py）不在这里导出，因为它是流水线内部工具，不供外部直接调用。
"""

from .pipeline import ROLE_HINTS, AgentBundle, DramaState, build_graph

__all__ = ["ROLE_HINTS", "AgentBundle", "DramaState", "build_graph"]
