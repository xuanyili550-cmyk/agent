"""按配置构造 LLMProvider，并对本地模型做进程级缓存。

为什么要缓存：LocalTransformersProvider 第一次 complete() 会加载几 GB 权重到显存/内存，
之前每个 Celery 任务都 new 一个新实例，等于每个任务重新加载一次权重，GPU 时间全浪费在
加载上。缓存后同一个 worker 进程里同一个 model_id 只加载一次。
API 型 provider 很轻，缓存的意义在于复用 SDK 的 HTTP 连接池。

多模型降级：``build_provider(..., fallback_models=[...])`` 会把主模型和备用模型串成
``FallbackProvider``（主模型限流/超时/没配 key 时自动切备用，见 fallback_provider.py）。
备用模型的后端按模型名推断（claude* -> anthropic，gpt* -> openai，其他 -> 本地），
构造时就失败的备用（例如 key 没配）只打警告并跳过，不让整条链因为一个备用不可用而起不来。
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

from .base import AnthropicProvider, LLMProvider, LocalTransformersProvider, NotConfiguredError, OpenAIProvider
from .fallback_provider import FallbackHook, FallbackProvider

log = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

_cache: dict[tuple[str, str], LLMProvider] = {}
_cache_lock = threading.Lock()


def resolve_provider_kind(model: str | None, default_kind: str = "local") -> str:
    """按模型名前缀推断后端：claude* -> anthropic，gpt*/o1*/o3* -> openai，其他 -> local。

    为什么用前缀而不是配置表：Celery payload 里只带一个 model 字符串，调用方不该还要知道
    "这个模型归哪家"；前缀规则覆盖了项目里会用到的全部模型，未知名字一律当 Hugging Face id 本地跑。
    """
    if not model:
        return default_kind
    name = model.lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "local"


def build_provider(
    kind: str | None = None,
    model: str | None = None,
    *,
    use_cache: bool = True,
    fallback_models: Sequence[str] | None = None,
    fallback_cooldown_seconds: float = 60.0,
    on_fallback: FallbackHook | None = None,
) -> LLMProvider:
    """kind: "anthropic" | "openai" | "local" | "mock"；None 时按 model 名推断。

    "mock" 不在这里构造（fixture 由调用方给），传进来会抛 NotConfiguredError 提醒。
    ``fallback_models`` 非空时返回 ``FallbackProvider``（主模型在前，备用按给定顺序在后）。
    """
    primary = _build_single(kind, model, use_cache=use_cache)
    if not fallback_models:
        return primary
    chain: list[LLMProvider] = [primary]
    for backup_model in fallback_models:
        if backup_model == primary.model:
            continue  # 备用列表里写了主模型自己：跳过，不然会"降级"到同一个模型
        try:
            chain.append(_build_single(resolve_provider_kind(backup_model), backup_model, use_cache=use_cache))
        except NotConfiguredError as exc:
            log.warning("备用模型 %s 不可用，已跳过：%s", backup_model, exc)
    if len(chain) == 1:
        return primary
    return FallbackProvider(chain, cooldown_seconds=fallback_cooldown_seconds, on_fallback=on_fallback)


def _build_single(kind: str | None, model: str | None, *, use_cache: bool) -> LLMProvider:
    """构造（或从缓存取）单个 provider；build_provider 的原始逻辑。"""
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
    """清空进程级 provider 缓存（测试切换配置、或想强制重新加载本地权重时用）。"""
    with _cache_lock:
        _cache.clear()
