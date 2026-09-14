"""按配置构造 LLMProvider，并对本地模型做进程级缓存。

为什么要缓存：LocalTransformersProvider 第一次 complete() 会加载几 GB 权重到显存/内存，
之前每个 Celery 任务都 new 一个新实例，等于每个任务重新加载一次权重，GPU 时间全浪费在
加载上。缓存后同一个 worker 进程里同一个 model_id 只加载一次。
API 型 provider 很轻，缓存的意义在于复用 SDK 的 HTTP 连接池。
"""

from __future__ import annotations

import threading

from .base import AnthropicProvider, LLMProvider, LocalTransformersProvider, NotConfiguredError, OpenAIProvider

DEFAULT_LOCAL_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

_cache: dict[tuple[str, str], LLMProvider] = {}
_cache_lock = threading.Lock()


def resolve_provider_kind(model: str | None, default_kind: str = "local") -> str:
    """按模型名前缀推断后端：claude* -> anthropic，gpt*/o1*/o3* -> openai，其他 -> local。"""
    if not model:
        return default_kind
    name = model.lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "local"


def build_provider(kind: str | None = None, model: str | None = None, *, use_cache: bool = True) -> LLMProvider:
    """kind: "anthropic" | "openai" | "local" | "mock"；None 时按 model 名推断。

    "mock" 不在这里构造（fixture 由调用方给），传进来会抛 NotConfiguredError 提醒。
    """
    kind = kind or resolve_provider_kind(model)
    if kind == "mock":
        raise NotConfiguredError("mock provider 需要调用方自己传 MockLLMProvider(fixtures)")
    if kind == "anthropic":
        model = model or DEFAULT_ANTHROPIC_MODEL
    elif kind == "openai":
        model = model or DEFAULT_OPENAI_MODEL
    elif kind == "local":
        model = model or DEFAULT_LOCAL_MODEL
    else:
        raise ValueError(f"未知的 LLM provider 类型：{kind}")

    key = (kind, model)
    if use_cache and key in _cache:
        return _cache[key]
    with _cache_lock:
        if use_cache and key in _cache:
            return _cache[key]
        if kind == "anthropic":
            provider: LLMProvider = AnthropicProvider(model=model)
        elif kind == "openai":
            provider = OpenAIProvider(model=model)
        else:
            provider = LocalTransformersProvider(model_id=model)
        if use_cache:
            _cache[key] = provider
        return provider


def clear_provider_cache() -> None:
    with _cache_lock:
        _cache.clear()
