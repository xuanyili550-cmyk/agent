"""故事引擎的 Agent 包：LLM provider 抽象、会话记忆、工具层、工具循环 Agent、多模型降级，
以及九个业务 Agent（故事/角色/剧集/剧本/分镜/提示词/章节规划/质检官/总编审）的统一导出。

外部（demo、13_INFRA 队列任务、测试）一律 ``from agents import X``，不直接 import 子模块，
这样内部文件怎么拆分都不影响调用方。
"""

from .base import (
    AgentGenerationError,
    AnthropicProvider,
    BaseAgent,
    LLMProvider,
    LLMUsage,
    LocalTransformersProvider,
    MockLLMProvider,
    NotConfiguredError,
    OpenAIProvider,
    TransientLLMError,
    build_memory,
)
from .chapter_planner_agent import ChapterPlannerAgent
from .character_agent import CharacterAgent
from .chief_editor_agent import ChiefEditorAgent
from .drama_assistant import DramaAssistantAgent, DramaContext, build_drama_toolbox
from .episode_agent import EpisodeAgent
from .fallback_provider import FallbackEvent, FallbackProvider
from .memory import (
    ConversationLockTimeout,
    ConversationMemory,
    ConversationStore,
    FileConversationStore,
    InMemoryConversationStore,
    RedisConversationStore,
    SQLiteConversationStore,
    context_window_for,
    estimate_tokens,
)
from .prompt_agent import PromptAgent
from .provider_factory import build_provider, clear_provider_cache, resolve_provider_kind
from .qc_officer_agent import QCOfficerAgent, rule_check
from .screenplay_agent import ScreenplayAgent
from .story_agent import StoryAgent
from .storyboard_agent import DialogueBatch, StoryboardAgent
from .tool_agent import AgentLoopError, AgentRunResult, ToolAgent
from .tools import (
    AllowlistGate,
    AutoApproveGate,
    AutoDenyGate,
    CallbackGate,
    ConfirmationGate,
    ConsoleGate,
    ToolArgumentError,
    ToolDeniedError,
    ToolError,
    ToolNotFoundError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "DialogueBatch",
    "AgentGenerationError",
    "AnthropicProvider",
    "BaseAgent",
    "LLMProvider",
    "LLMUsage",
    "LocalTransformersProvider",
    "MockLLMProvider",
    "NotConfiguredError",
    "OpenAIProvider",
    "TransientLLMError",
    "build_memory",
    "build_provider",
    "clear_provider_cache",
    "resolve_provider_kind",
    "ConversationLockTimeout",
    "ConversationMemory",
    "ConversationStore",
    "FileConversationStore",
    "InMemoryConversationStore",
    "RedisConversationStore",
    "SQLiteConversationStore",
    "context_window_for",
    "estimate_tokens",
    "FallbackEvent",
    "FallbackProvider",
    "AgentLoopError",
    "AgentRunResult",
    "ToolAgent",
    "DramaAssistantAgent",
    "DramaContext",
    "build_drama_toolbox",
    "AllowlistGate",
    "AutoApproveGate",
    "AutoDenyGate",
    "CallbackGate",
    "ConfirmationGate",
    "ConsoleGate",
    "ToolArgumentError",
    "ToolDeniedError",
    "ToolError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "ChapterPlannerAgent",
    "CharacterAgent",
    "ChiefEditorAgent",
    "EpisodeAgent",
    "PromptAgent",
    "QCOfficerAgent",
    "rule_check",
    "ScreenplayAgent",
    "StoryAgent",
    "StoryboardAgent",
]
