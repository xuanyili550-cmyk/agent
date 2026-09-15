"""workflows 包：故事引擎的 LangGraph 流水线入口。

对外导出流水线相关的公共符号（图构建、状态类型、agent 容器、角色提示词），
以及检查点持久化（thread_id 分会话、可续跑）和流式执行（按节点吐进度）这两块能力。
校验逻辑（validation.py）不在这里导出，它是流水线内部工具，不供外部直接调用。
"""

from .checkpointing import StoreCheckpointSaver, build_checkpointer
from .pipeline import (
    ROLE_HINTS,
    AgentBundle,
    DramaState,
    build_graph,
    graph_progress,
    run_pipeline,
    stream_pipeline,
    thread_config,
)

__all__ = [
    "ROLE_HINTS",
    "AgentBundle",
    "DramaState",
    "build_graph",
    "build_checkpointer",
    "StoreCheckpointSaver",
    "graph_progress",
    "run_pipeline",
    "stream_pipeline",
    "thread_config",
]
