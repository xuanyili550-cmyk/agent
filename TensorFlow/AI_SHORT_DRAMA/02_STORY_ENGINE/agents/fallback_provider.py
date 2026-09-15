"""多模型降级：主模型不可用时自动切到备用模型（《Agent 搭建指南》坑 4"降级响应"）。

``FallbackProvider`` 本身也是一个 ``LLMProvider``，对 BaseAgent / ToolAgent / llm_task 完全透明：
调用方还是只认 ``complete()``。内部按顺序尝试 ``providers``：

    主模型（如 claude-sonnet）--瞬时错误/未配置--> 备用 1（如 gpt-4o-mini）--> 备用 2（本地模型）

设计要点：
- 只对"换个模型大概率就好"的错误降级：``TransientLLMError``（限流/超时/5xx，SDK 自带重试已耗尽）
  和 ``NotConfiguredError``（key 没配）。业务错误（JSON 不合法）不降级，那是 prompt 的问题，
  换模型只会把问题藏起来。
- 熔断冷却：某个 provider 失败后 ``cooldown_seconds`` 内直接跳过它，后续请求不用每次都
  先等主模型超时再降级；冷却结束自动恢复尝试主模型（不需要人工"切回去"）。
- 上下文预算取所有 provider 的最小值：历史窗口是按 provider.context_window 裁的，如果按主模型的
  200k 裁、降级到 8k 的本地模型就会直接超限报错。
- 每次降级都通过 ``on_fallback(from_model, to_model, exc)`` 回调上报，13_INFRA 接到 Prometheus。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .base import LLMProvider, LLMUsage, NotConfiguredError, TransientLLMError
from .memory import Message

FallbackHook = Callable[[str, str, Exception], None]


@dataclass
class FallbackEvent:
    """一次降级记录：从哪个模型切到哪个模型、因为什么错误、发生时间。"""

    from_model: str
    to_model: str
    error: str
    at: float = field(default_factory=time.time)


class FallbackProvider(LLMProvider):
    """按顺序尝试多个 provider 的组合 provider；对外表现和单个 provider 一样。"""

    provider_name = "fallback"

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        cooldown_seconds: float = 60.0,
        on_fallback: FallbackHook | None = None,
    ) -> None:
        """
        ``providers``：至少一个，顺序即优先级。
        ``cooldown_seconds``：provider 失败后被跳过的时长；0 表示每次都从头试。
        ``on_fallback``：降级回调，用来打指标/日志。
        """
        if not providers:
            raise ValueError("FallbackProvider 至少需要一个 provider")
        self.providers = list(providers)
        self.cooldown_seconds = cooldown_seconds
        self.on_fallback = on_fallback
        self.events: list[FallbackEvent] = []
        self._failed_until: dict[int, float] = {}  # provider 下标 -> 冷却截止时间
        self.active: LLMProvider = self.providers[0]  # 最近一次成功的 provider
        # 预算按最保守的 provider 算，见模块 docstring
        self.context_window = min(p.context_window for p in self.providers)
        self.max_tokens = min(p.max_tokens for p in self.providers)
        self.model = self.providers[0].model

    @property
    def primary(self) -> LLMProvider:
        """优先级最高的主模型。"""
        return self.providers[0]

    def count_tokens(self, text: str) -> int:
        """用主模型的 tokenizer 计数（各家差异不大，估算方向偏保守即可）。"""
        return self.primary.count_tokens(text)

    def _available(self, index: int) -> bool:
        """该 provider 是否不在冷却期内。"""
        return time.monotonic() >= self._failed_until.get(index, 0.0)

    def _mark_failed(self, index: int) -> None:
        """进入冷却。"""
        if self.cooldown_seconds > 0:
            self._failed_until[index] = time.monotonic() + self.cooldown_seconds

    def _record_fallback(self, index: int, provider: LLMProvider, exc: Exception) -> None:
        """把某个 provider 标记为失败并记一次降级事件（如果还有下一个可切）。"""
        self._mark_failed(index)
        next_provider = self.providers[index + 1] if index + 1 < len(self.providers) else None
        if next_provider is None:
            return
        event = FallbackEvent(provider.model, next_provider.model, f"{type(exc).__name__}: {exc}")
        self.events.append(event)
        if self.on_fallback is not None:
            try:
                self.on_fallback(event.from_model, event.to_model, exc)
            except Exception:  # 指标回调失败不影响降级本身
                pass

    def _adopt(self, provider: LLMProvider) -> None:
        """把成功干活的那个 provider 的模型名和用量暴露到外层，成本核算才记在正确的模型头上。"""
        self.active = provider
        self.last_usage = provider.last_usage or LLMUsage(provider.provider_name, provider.model)
        self.model = provider.model

    def complete(self, system_prompt: str, user_prompt: str, history: list[Message] | None = None) -> str:
        """依次尝试；全部失败时抛 TransientLLMError（让 Celery 任务层按指数退避再投递）。"""
        last_exc: Exception | None = None
        skipped_all = True
        for index, provider in enumerate(self.providers):
            if not self._available(index):
                continue
            skipped_all = False
            try:
                text = provider.complete(system_prompt, user_prompt, history)
            except (TransientLLMError, NotConfiguredError) as exc:
                self._record_fallback(index, provider, exc)
                last_exc = exc
                continue
            self._adopt(provider)
            return text
        if skipped_all:
            # 所有 provider 都在冷却期：清掉冷却立刻重试一遍，而不是干等
            self._failed_until.clear()
            return self.complete(system_prompt, user_prompt, history)
        raise TransientLLMError(f"所有 LLM provider 都不可用（共 {len(self.providers)} 个），最后错误：{last_exc}") from last_exc

    def stream(self, system_prompt: str, user_prompt: str, history: list[Message] | None = None):
        """流式版的降级：只在**还没吐出任何内容**时才切换 provider。

        为什么有这个限制：已经往前端推了半句话再换模型，用户会看到两个模型的输出被拼在一起，
        比直接失败更糟。所以一旦第一块增量发出去，中途的错误就原样抛出，由上层决定重试整轮。
        """
        last_exc: Exception | None = None
        skipped_all = True
        for index, provider in enumerate(self.providers):
            if not self._available(index):
                continue
            skipped_all = False
            emitted = False
            try:
                for piece in provider.stream(system_prompt, user_prompt, history):
                    emitted = True
                    yield piece
            except (TransientLLMError, NotConfiguredError) as exc:
                if emitted:
                    raise
                self._record_fallback(index, provider, exc)
                last_exc = exc
                continue
            self._adopt(provider)
            return
        if skipped_all:
            self._failed_until.clear()
            yield from self.stream(system_prompt, user_prompt, history)
            return
        raise TransientLLMError(f"所有 LLM provider 都不可用（共 {len(self.providers)} 个），最后错误：{last_exc}") from last_exc
