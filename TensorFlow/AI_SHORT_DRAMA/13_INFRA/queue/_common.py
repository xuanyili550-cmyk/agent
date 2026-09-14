"""队列任务共用的小工具：跨顶层模块 import、LLM provider/记忆构造、用量记账。"""

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
from ..observability import LLM_COST_USD, LLM_TOKENS, get_logger  # noqa: E402

log = get_logger("queue.common")


def mod(name: str):
    """``mod("02_STORY_ENGINE.agents.base")``：数字开头的顶层包只能这样 import。"""
    return importlib.import_module(name)


def build_llm_provider(settings: Settings | None = None, model: str | None = None):
    """按 settings.llm_provider 构造 provider；"mock" 用 02 的 demo fixture（开发/测试）。"""
    settings = settings or get_settings()
    kind = settings.llm_provider
    model = model or settings.llm_model
    if kind == "mock":
        return mod("02_STORY_ENGINE.demo").build_mock_provider()
    factory = mod("02_STORY_ENGINE.agents.provider_factory")
    return factory.build_provider(kind, model)


def build_conversation_store(settings: Settings | None = None):
    """历史消息存储后端：LLM_CONTEXT_REDIS_URL -> Redis；否则本地 JSON 文件。"""
    settings = settings or get_settings()
    memory_mod = mod("02_STORY_ENGINE.agents.memory")
    if settings.llm_context_redis_url:
        return memory_mod.RedisConversationStore(settings.llm_context_redis_url, ttl_seconds=settings.llm_context_ttl_seconds)
    root = settings.llm_context_dir or str(settings.artifacts_root / "contexts")
    return memory_mod.FileConversationStore(root)


def estimate_cost_usd(settings: Settings, model: str, input_tokens: int, output_tokens: int) -> float:
    name = (model or "").lower()

    def price(table: dict[str, float]) -> float:
        for prefix in sorted(table, key=len, reverse=True):
            if name.startswith(prefix):
                return table[prefix]
        return 0.0

    return round(input_tokens / 1000 * price(settings.llm_price_per_1k_input) + output_tokens / 1000 * price(settings.llm_price_per_1k_output), 6)


def make_usage_sink(settings: Settings, *, context_id: str | None = None, run_id: str | None = None):
    """返回 BaseAgent.usage_sink 回调：每次 LLM 调用 -> llm_usage 表 + Prometheus。"""

    def sink(agent_name: str, usage: Any) -> None:
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
