"""队列任务共用的小工具：跨顶层模块 import、LLM provider/记忆构造、用量 & 工具调用记账。

这里是"02_STORY_ENGINE 的 Agent 抽象"和"13_INFRA 的配置/数据库/指标"之间的粘合层：
Agent 只暴露回调（usage_sink / tool_call_sink / run_sink / on_fallback），本模块把它们接到
llm_usage 表和 Prometheus。任务代码只需要 ``build_llm_provider()`` + ``make_usage_sink()`` 两行。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ..config import Settings, get_settings  # noqa: E402
from ..database import repository as repo  # noqa: E402
from ..database.session import session_scope  # noqa: E402
from ..observability import LLM_COST_USD, LLM_FALLBACK, LLM_TOKENS, get_logger, record_agent_run, record_tool_call  # noqa: E402

log = get_logger("queue.common")


def mod(name: str):
    """``mod("02_STORY_ENGINE.agents.base")``：数字开头的顶层包只能这样 import。"""
    return importlib.import_module(name)


def _on_fallback(from_model: str, to_model: str, exc: Exception) -> None:
    """FallbackProvider 的降级回调：计数 + 打警告日志，让运维知道主模型出问题了。"""
    LLM_FALLBACK.labels(from_model, to_model).inc()
    log.warning("LLM 降级", extra={"from_model": from_model, "to_model": to_model, "error": str(exc)[:200]})


def build_llm_provider(settings: Settings | None = None, model: str | None = None):
    """按 settings.llm_provider 构造 provider；"mock" 用 02 的 demo fixture（开发/测试）。

    配了 ``LLM_FALLBACK_MODELS`` 时返回 FallbackProvider（主模型 + 备用链），降级事件进 Prometheus。
    mock 不做降级：测试要的是确定性。
    """
    settings = settings or get_settings()
    kind = settings.llm_provider
    model = model or settings.llm_model
    if kind == "mock":
        return mod("02_STORY_ENGINE.demo").build_mock_provider()
    factory = mod("02_STORY_ENGINE.agents.provider_factory")
    return factory.build_provider(
        kind,
        model,
        fallback_models=settings.llm_fallback_models,
        fallback_cooldown_seconds=settings.llm_fallback_cooldown_seconds,
        on_fallback=_on_fallback,
    )


def build_conversation_store(settings: Settings | None = None):
    """历史消息存储后端：LLM_CONTEXT_REDIS_URL -> Redis；LLM_CONTEXT_SQLITE_PATH -> SQLite；否则本地 JSON 文件。"""
    settings = settings or get_settings()
    memory_mod = mod("02_STORY_ENGINE.agents.memory")
    if settings.llm_context_redis_url:
        return memory_mod.RedisConversationStore(settings.llm_context_redis_url, ttl_seconds=settings.llm_context_ttl_seconds)
    if settings.llm_context_sqlite_path:
        return memory_mod.SQLiteConversationStore(settings.llm_context_sqlite_path)
    root = settings.llm_context_dir or str(settings.artifacts_root / "contexts")
    return memory_mod.FileConversationStore(root)


def build_confirmation_gate(settings: Settings | None = None):
    """工具级确认门：AGENT_TOOL_ALLOWLIST 里的工具自动放行，其余需确认的工具一律拒绝。

    队列 worker 没有人在终端旁边，所以不能用 ConsoleGate；生产上"需要人确认"的动作应该
    做成流水线的 awaiting_review 状态（见 orchestration.py），而不是让 Agent 在后台等人回答。
    """
    settings = settings or get_settings()
    tools_mod = mod("02_STORY_ENGINE.agents.tools")
    return tools_mod.AllowlistGate(settings.agent_tool_allowlist)


def estimate_cost_usd(settings: Settings, model: str, input_tokens: int, output_tokens: int) -> float:
    """按 settings 里的每 1k token 单价表估算一次调用的美元成本；模型名前缀匹配，查不到按 0。"""
    name = (model or "").lower()

    def price(table: dict[str, float]) -> float:
        """最长前缀优先，避免 "gpt-4o" 被 "gpt-4o-mini" 的价格抢先匹配。"""
        for prefix in sorted(table, key=len, reverse=True):
            if name.startswith(prefix):
                return table[prefix]
        return 0.0

    return round(input_tokens / 1000 * price(settings.llm_price_per_1k_input) + output_tokens / 1000 * price(settings.llm_price_per_1k_output), 6)


def make_usage_sink(settings: Settings, *, context_id: str | None = None, run_id: str | None = None):
    """返回 BaseAgent.usage_sink 回调：每次 LLM 调用 -> llm_usage 表 + Prometheus（四大指标之"Token 消耗"）。"""

    def sink(agent_name: str, usage: Any) -> None:
        """记一次调用的 token 与成本。"""
        cost = estimate_cost_usd(settings, usage.model, usage.input_tokens, usage.output_tokens)
        LLM_TOKENS.labels(usage.provider, usage.model, "input").inc(usage.input_tokens)
        LLM_TOKENS.labels(usage.provider, usage.model, "output").inc(usage.output_tokens)
        LLM_COST_USD.labels(usage.provider, usage.model).inc(cost)
        try:
            with session_scope() as db:
                repo.record_llm_usage(
                    db,
                    provider=usage.provider,
                    model=usage.model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=cost,
                    context_id=context_id,
                    run_id=run_id,
                    agent=agent_name,
                )
        except Exception as exc:  # 记账失败不能拖垮主流程，但要留日志
            log.warning("LLM 用量写库失败", extra={"error": str(exc)[:200]})

    return sink


def make_tool_call_sink():
    """返回 ToolAgent.tool_call_sink 回调：每次工具调用 -> Prometheus（四大指标之"工具调用成功率"）。"""

    def sink(result: Any) -> None:
        """记一次工具调用的状态与耗时；被拒绝/参数错误的也记，成功率才真实。"""
        record_tool_call(result.tool, result.status, result.duration_seconds)
        if not result.ok:
            log.warning("工具调用未成功", extra={"tool": result.tool, "status": result.status, "error": (result.error or "")[:200]})

    return sink


def make_run_sink():
    """返回 ToolAgent.run_sink 回调：每次 run -> 轮数直方图 + 停止原因计数（无限循环被拦下会体现在这里）。"""

    def sink(agent_name: str, result: Any) -> None:
        """记一次 run。"""
        record_agent_run(agent_name, result.stopped_reason, result.tool_rounds)

    return sink


def instrument_tool_agent(agent: Any, settings: Settings, *, context_id: str | None = None, run_id: str | None = None) -> Any:
    """把一个 ToolAgent 的四个回调全部接上：token 记账、工具调用指标、run 指标、确认门。返回 agent 本身。"""
    agent.usage_sink = make_usage_sink(settings, context_id=context_id, run_id=run_id)
    agent.tool_call_sink = make_tool_call_sink()
    agent.run_sink = make_run_sink()
    agent.gate = build_confirmation_gate(settings)
    agent.max_tool_rounds = settings.agent_max_tool_rounds
    return agent
