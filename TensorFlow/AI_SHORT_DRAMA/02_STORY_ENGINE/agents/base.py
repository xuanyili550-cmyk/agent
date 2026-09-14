from __future__ import annotations

import json
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .memory import (
    CompactionStrategy,
    ConversationMemory,
    ConversationStore,
    Message,
    Summarizer,
    context_window_for,
    estimate_tokens,
)

# bootstrap: 让 02_STORY_ENGINE 与 03_STRUCTURED_DATA 互相可 import，无需打包安装
for _p in Path(__file__).resolve().parents:
    if (_p / "03_STRUCTURED_DATA").is_dir():
        sys.path.insert(0, str(_p / "03_STRUCTURED_DATA"))
        sys.path.insert(0, str(_p / "02_STORY_ENGINE"))
        _ROOT = _p
        break
else:
    raise RuntimeError("找不到 AI_SHORT_DRAMA 项目根目录")

PROMPTS_DIR = _ROOT / "02_STORY_ENGINE" / "prompts"

T = TypeVar("T", bound=BaseModel)


class NotConfiguredError(RuntimeError):
    pass


class AgentGenerationError(RuntimeError):
    pass


class TransientLLMError(RuntimeError):
    """网络抖动 / 429 限流 / 5xx 这类"再试一次大概率就好"的错误。

    BaseAgent 不在进程内重试它（避免和 SDK 自带的重试叠加成指数级等待），而是原样抛给
    Celery 任务层，由 ``BaseTask.autoretry_for`` 按指数退避重新投递——这就是三级重试里的
    第一级"同样参数再试一次"。
    """


@dataclass
class LLMUsage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    # 模型上下文窗口上限（token）。ConversationMemory 用它决定历史消息什么时候该裁剪。
    context_window: int = context_window_for(None)
    # 单次输出上限；连同结构化标记一起作为 ConversationMemory 的 reserve_tokens 余量。
    max_tokens: int = 4096
    provider_name: str = "unknown"
    model: str = "unknown"
    # 最近一次 complete() 的 token 用量；没有官方数字的 provider 用估算值填充
    last_usage: LLMUsage | None = None

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, history: list[Message] | None = None) -> str:
        """``history`` 是同一会话里更早的 user/assistant 轮次（由 ConversationMemory.window() 给出），
        实现方要把它原样放在本次 user prompt 之前发给模型；None/空表示无状态单轮调用。"""
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """给 ConversationMemory 用的 token 计数；子类有真 tokenizer 就覆盖，默认离线估算。"""
        return estimate_tokens(text)

    def _estimate_usage(self, system_prompt: str, user_prompt: str, history: list[Message] | None, reply: str) -> LLMUsage:
        prompt_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_prompt)
        prompt_tokens += sum(self.count_tokens(m["content"]) for m in (history or []))
        return LLMUsage(self.provider_name, self.model, prompt_tokens, self.count_tokens(reply))


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 4096,
        timeout_seconds: float = 120.0,
        sdk_max_retries: int = 2,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise NotConfiguredError(
                f"AnthropicProvider 未配置：环境变量 {api_key_env} 为空。请设置 ANTHROPIC_API_KEY 后再使用真实 LLM，或改用 MockLLMProvider 离线运行。"
            )
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window_for(model)
        self.timeout_seconds = timeout_seconds
        self.sdk_max_retries = sdk_max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            # SDK 自带对 429/5xx/连接错误的短退避重试；超过 sdk_max_retries 后抛出，由我们转成 TransientLLMError
            self._client = anthropic.Anthropic(api_key=self._api_key, timeout=self.timeout_seconds, max_retries=self.sdk_max_retries)
        return self._client

    def complete(self, system_prompt: str, user_prompt: str, history: list[Message] | None = None) -> str:
        import anthropic

        try:
            response = self._get_client().messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[*(history or []), {"role": "user", "content": user_prompt}],
            )
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError) as exc:
            raise TransientLLMError(f"Anthropic 瞬时错误：{type(exc).__name__}: {exc}") from exc
        usage = getattr(response, "usage", None)
        self.last_usage = LLMUsage(
            self.provider_name,
            self.model,
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        max_tokens: int = 4096,
        timeout_seconds: float = 120.0,
        sdk_max_retries: int = 2,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise NotConfiguredError(
                f"OpenAIProvider 未配置：环境变量 {api_key_env} 为空。请设置 OPENAI_API_KEY 后再使用真实 LLM，或改用 MockLLMProvider 离线运行。"
            )
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window_for(model)
        self.timeout_seconds = timeout_seconds
        self.sdk_max_retries = sdk_max_retries
        self._client = None
        self._encoding = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self._api_key, timeout=self.timeout_seconds, max_retries=self.sdk_max_retries)
        return self._client

    def count_tokens(self, text: str) -> int:
        # tiktoken 首次使用要下载 BPE 词表；离线环境拿不到就退回估算，不能因为算 token 把主流程搞挂
        if self._encoding is None:
            try:
                import tiktoken

                try:
                    self._encoding = tiktoken.encoding_for_model(self.model)
                except KeyError:
                    self._encoding = tiktoken.get_encoding("o200k_base")
            except Exception:
                self._encoding = False
        if not self._encoding:
            return estimate_tokens(text)
        return len(self._encoding.encode(text))

    def complete(self, system_prompt: str, user_prompt: str, history: list[Message] | None = None) -> str:
        import openai

        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *(history or []),
                    {"role": "user", "content": user_prompt},
                ],
            )
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
            raise TransientLLMError(f"OpenAI 瞬时错误：{type(exc).__name__}: {exc}") from exc
        usage = getattr(response, "usage", None)
        self.last_usage = LLMUsage(
            self.provider_name,
            self.model,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )
        return response.choices[0].message.content or ""


class LocalTransformersProvider(LLMProvider):
    """Runs a local, free, open-weight instruct model via transformers -- no
    API key, no network calls at inference time (only the first weight
    download). Default model is ungated on Hugging Face; swap model_id for
    any other open chat model (see 06_MODELS/llm/registry.json)."""

    provider_name = "local"

    def __init__(self, model_id: str = "microsoft/Phi-3.5-mini-instruct", device_map: str = "auto", max_new_tokens: int = 2048):
        self.model_id = model_id
        self.model = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.max_tokens = max_new_tokens
        self.context_window = context_window_for(model_id)
        self._pipe = None

    def count_tokens(self, text: str) -> int:
        # 权重已加载时用模型自己的 tokenizer，精确；未加载时不为了数 token 去加载几 GB 权重
        if self._pipe is not None and getattr(self._pipe, "tokenizer", None) is not None:
            return len(self._pipe.tokenizer.encode(text, add_special_tokens=False))
        return estimate_tokens(text)

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from transformers import pipeline

        self._pipe = pipeline(
            "text-generation",
            model=self.model_id,
            device_map=self.device_map,
            dtype=torch.bfloat16,
        )
        # 权重加载后以模型 config 里声明的真实上下文长度为准，覆盖前缀表里的估计值
        max_pos = getattr(self._pipe.model.config, "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0:
            self.context_window = max_pos
        return self._pipe

    def complete(self, system_prompt: str, user_prompt: str, history: list[Message] | None = None) -> str:
        pipe = self._load()
        messages = [
            {"role": "system", "content": system_prompt},
            *(history or []),
            {"role": "user", "content": user_prompt},
        ]
        output = pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        reply = output[0]["generated_text"][-1]
        text = reply["content"] if isinstance(reply, dict) else str(reply)
        self.last_usage = self._estimate_usage(system_prompt, user_prompt, history, text)
        return text


FixtureFn = Callable[[str, str], str]


class MockLLMProvider(LLMProvider):
    provider_name = "mock"
    model = "mock"

    def __init__(self, fixtures: dict[str, str | FixtureFn]):
        self.fixtures = fixtures
        self.context_window = 128_000
        self.calls: list[dict] = []  # 记录每次调用带了多少条历史，方便测试/调试

    def complete(self, system_prompt: str, user_prompt: str, history: list[Message] | None = None) -> str:
        self.calls.append({"history_len": len(history or []), "user_prompt": user_prompt})
        match = re.search(r"\[TARGET_SCHEMA=(\w+)\]", user_prompt)
        schema_name = match.group(1) if match else None
        if schema_name not in self.fixtures:
            raise NotConfiguredError(f"MockLLMProvider 没有为 schema={schema_name} 注册 fixture")
        fixture = self.fixtures[schema_name]
        text = fixture(system_prompt, user_prompt) if callable(fixture) else fixture
        self.last_usage = self._estimate_usage(system_prompt, user_prompt, history, text)
        return text


def _extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _collect_enums(node: object, out: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Walks a pydantic model_json_schema() dict (both top-level properties and
    $defs) and collects every {"title": ..., "enum": [...]} it finds, keyed by
    title. Used to hand the LLM the exact closed set of legal values for enum
    fields (e.g. ShotSize, CameraAngle) instead of relying on it to guess a
    plausible-looking string and burning a validation-retry when it's wrong."""
    if out is None:
        out = {}
    if isinstance(node, dict):
        if isinstance(node.get("enum"), list) and node.get("title"):
            out.setdefault(node["title"], node["enum"])
        for value in node.values():
            _collect_enums(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_enums(item, out)
    return out


def _enum_cheatsheet(schema: type[BaseModel]) -> str:
    enums = _collect_enums(schema.model_json_schema())
    if not enums:
        return ""
    lines = "\n".join(f"- {name}: {', '.join(values)}" for name, values in sorted(enums.items()))
    return f"\n\n以下字段只能从给定枚举值里精确选一个（不要用同义词或自造值）：\n{lines}"


class BaseAgent(ABC):
    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str,
        max_retries: int = 3,
        memory: ConversationMemory | None = None,
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        # 会话记忆：传入后每次 generate() 都会带上同一会话的历史轮次，并把成功轮次追加进去。
        # 多个 agent 共用同一个 memory 对象，就等于整条流水线在一个连续对话里完成——
        # 生成角色时模型能看到前面生成的 Story Bible，生成分镜时能看到剧本，不再各说各话。
        self.memory = memory
        # 本 agent 每次 LLM 调用的用量流水（含失败重试），由上层汇总记账
        self.usage_log: list[LLMUsage] = []
        # 用量回调：13_INFRA 把它接到数据库 + Prometheus；不接也不影响运行
        self.usage_sink: Callable[[str, LLMUsage], None] | None = None

    def _record_usage(self) -> None:
        usage = self.provider.last_usage
        if usage is None:
            return
        self.usage_log.append(usage)
        if self.usage_sink is not None:
            self.usage_sink(self.__class__.__name__, usage)

    def generate(self, user_prompt: str, schema: type[T]) -> T:
        marker = (
            f"\n\n[TARGET_SCHEMA={schema.__name__}]\n"
            "只输出符合该 schema 字段结构的单个 JSON 对象，不要包含任何解释文字或 markdown 代码块标记。"
            f"{_enum_cheatsheet(schema)}"
        )
        prompt = user_prompt + marker
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            if self.memory is not None:
                # 加锁：同一会话同一时刻只有一个调用在"读历史 -> 调模型 -> 写历史"，
                # 多个 worker 处理同一 context_id 时不会互相覆盖对方刚写入的轮次
                with self.memory.transaction():
                    history = self.memory.window(self.system_prompt, prompt)
                    raw = self.provider.complete(self.system_prompt, prompt, history)
                    self._record_usage()
                    parsed, exc = self._try_parse(raw, schema)
                    if parsed is not None:
                        # 只记录最终通过校验的那一轮；失败的重试不进历史
                        self.memory.add_turn(prompt, raw)
                        return parsed
            else:
                raw = self.provider.complete(self.system_prompt, prompt, None)
                self._record_usage()
                parsed, exc = self._try_parse(raw, schema)
                if parsed is not None:
                    return parsed
            last_error = exc
            prompt = user_prompt + f"\n\n第 {attempt} 次输出未通过校验，错误信息：{exc}\n" + "请修正后重新只输出合法 JSON。" + marker
        raise AgentGenerationError(f"{self.__class__.__name__} 在 {self.max_retries} 次重试后仍未生成合法的 {schema.__name__} JSON：{last_error}")

    @staticmethod
    def _try_parse(raw: str, schema: type[T]) -> tuple[T | None, Exception | None]:
        try:
            return schema.model_validate_json(_extract_json(raw)), None
        except (ValidationError, json.JSONDecodeError) as exc:
            return None, exc


SUMMARY_SYSTEM_PROMPT = (
    "你是短剧创作流水线的会话压缩器。把给定的历史对话压缩成一段中文摘要，"
    "必须完整保留：已确定的故事设定、角色 id 与姓名、集数/场次/镜头 id、已做出的剧情决定。"
    "只输出摘要正文，不要解释。"
)


def build_memory(
    provider: LLMProvider,
    context_id: str,
    store: ConversationStore | None = None,
    *,
    compaction: CompactionStrategy = "truncate",
    extra_reserve_tokens: int = 2_048,
) -> ConversationMemory:
    """按 provider 的真实上下文上限构造 ConversationMemory。

    预算 = provider.context_window - (provider.max_tokens + extra_reserve_tokens)：
    留出模型输出和 ``[TARGET_SCHEMA=...]``/枚举清单标记的余量，其余全给历史消息。
    ``compaction="summarize"`` 时用同一个 provider 做摘要，不需要额外接模型。
    """
    summarizer: Summarizer | None = None
    if compaction == "summarize":

        def summarizer(messages: list[Message]) -> str:
            transcript = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
            return provider.complete(SUMMARY_SYSTEM_PROMPT, transcript).strip()

    return ConversationMemory(
        context_id,
        store,
        max_context_tokens=provider.context_window,
        reserve_tokens=provider.max_tokens + extra_reserve_tokens,
        token_counter=provider.count_tokens,
        compaction=compaction,
        summarizer=summarizer,
    )
