"""通用 LLM 调用任务：一次 prompt -> 一次补全，可选带长期会话记忆。

它是 02_STORY_ENGINE 的 provider / memory 在队列里的薄封装，给 API 的 /llm 接口和调试用；
完整的故事流水线（多个 Agent 串起来跑 LangGraph）走 story_task，不经过这里。
用量记账（llm_usage 表 + Prometheus）由 _common.make_usage_sink 统一处理。
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import get_settings
from ..workers.celery_app import celery_app
from ._common import build_conversation_store, build_llm_provider, estimate_cost_usd, make_usage_sink, mod

__all__ = ["llm_task", "build_conversation_store"]


@celery_app.task(name="llm_task", bind=True)
def llm_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """LLM GPU 队列任务（剧本 / 对白 / 提示词生成都走这里调 LLM）。

    payload: {"prompt": str, "system": str | None, "model": str | None,
      "params": dict | None, "context_id": str | None}。
    返回: {"text": str, "model": str, "usage": dict | None, "context_id": str | None,
      "history_turns": int}。

    provider 由 settings.llm_provider 决定（payload["model"] 可覆盖模型名：claude* 走 Anthropic，
    gpt* 走 OpenAI，其他当作 Hugging Face 模型 id 本地跑）。本地模型在 worker 进程内缓存，
    不会每个任务重新加载权重。

    传了 ``context_id`` 就是一个长期会话：这次调用会带上该会话之前所有轮次的历史消息，
    并把本轮追加进持久化存储；只有历史逼近模型上下文上限时才裁掉最旧的轮次。
    ``params["compaction"]`` 可选 "truncate"（默认）或 "summarize"（触顶时先压成摘要）。
    不传 ``context_id`` 则退化为无状态单轮调用。
    瞬时错误（限流/网络）抛 TransientLLMError，由 BaseTask 指数退避重试。
    """
    settings = get_settings()
    base_mod = mod("02_STORY_ENGINE.agents.base")
    factory = mod("02_STORY_ENGINE.agents.provider_factory")

    # mock 模式下忽略 payload 的 model：测试环境不能因为传了个真实模型名就去联网
    model = payload.get("model")
    if model and settings.llm_provider != "mock":
        provider = factory.build_provider(factory.resolve_provider_kind(model), model)
    else:
        provider = build_llm_provider(settings)

    system = payload.get("system") or ""
    prompt = payload["prompt"]
    params = payload.get("params") or {}
    context_id = payload.get("context_id")

    def usage_dict():
        """把 provider 上一次调用的用量整理成返回值里的 usage 字段；provider 不报用量（如 mock）时为 None。"""
        usage = provider.last_usage
        if usage is None:
            return None
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": estimate_cost_usd(settings, usage.model, usage.input_tokens, usage.output_tokens),
        }

    sink = make_usage_sink(settings, context_id=context_id)

    if not context_id:
        # 无状态单轮：不读不写历史
        text = provider.complete(system, prompt)
        if provider.last_usage:
            sink("llm_task", provider.last_usage)
        return {"text": text, "model": provider.model, "usage": usage_dict(), "context_id": None, "history_turns": 0}

    memory = base_mod.build_memory(
        provider,
        context_id,
        build_conversation_store(settings),
        compaction=params.get("compaction", "truncate"),
    )
    memory.lock_timeout = settings.llm_context_lock_timeout_seconds
    # transaction() 持有该会话的锁：同一 context_id 的并发调用串行化，历史不会互相覆盖
    with memory.transaction():
        history = memory.window(system, prompt)
        text = provider.complete(system, prompt, history)
        memory.add_turn(prompt, text)
        turns = memory.turn_count
    if provider.last_usage:
        sink("llm_task", provider.last_usage)
    return {"text": text, "model": provider.model, "usage": usage_dict(), "context_id": context_id, "history_turns": turns}
