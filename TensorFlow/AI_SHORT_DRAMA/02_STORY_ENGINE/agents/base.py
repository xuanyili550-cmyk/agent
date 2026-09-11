from __future__ import annotations

import json
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

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


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-5", api_key_env: str = "ANTHROPIC_API_KEY", max_tokens: int = 4096):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise NotConfiguredError(
                f"AnthropicProvider 未配置：环境变量 {api_key_env} 为空。"
                "请设置 ANTHROPIC_API_KEY 后再使用真实 LLM，或改用 MockLLMProvider 离线运行。"
            )
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY", max_tokens: int = 4096):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise NotConfiguredError(
                f"OpenAIProvider 未配置：环境变量 {api_key_env} 为空。"
                "请设置 OPENAI_API_KEY 后再使用真实 LLM，或改用 MockLLMProvider 离线运行。"
            )
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


class LocalTransformersProvider(LLMProvider):
    """Runs a local, free, open-weight instruct model via transformers -- no
    API key, no network calls at inference time (only the first weight
    download). Default model is ungated on Hugging Face; swap model_id for
    any other open chat model (see 06_MODELS/llm/registry.json)."""

    def __init__(self, model_id: str = "microsoft/Phi-3.5-mini-instruct", device_map: str = "auto", max_new_tokens: int = 2048):
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self._pipe = None

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
        return self._pipe

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        pipe = self._load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        output = pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        reply = output[0]["generated_text"][-1]
        return reply["content"] if isinstance(reply, dict) else str(reply)


FixtureFn = Callable[[str, str], str]


class MockLLMProvider(LLMProvider):
    def __init__(self, fixtures: dict[str, str | FixtureFn]):
        self.fixtures = fixtures

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        match = re.search(r"\[TARGET_SCHEMA=(\w+)\]", user_prompt)
        schema_name = match.group(1) if match else None
        if schema_name not in self.fixtures:
            raise NotConfiguredError(f"MockLLMProvider 没有为 schema={schema_name} 注册 fixture")
        fixture = self.fixtures[schema_name]
        if callable(fixture):
            return fixture(system_prompt, user_prompt)
        return fixture


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
    def __init__(self, provider: LLMProvider, system_prompt: str, max_retries: int = 3):
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_retries = max_retries

    def generate(self, user_prompt: str, schema: type[T]) -> T:
        marker = (
            f"\n\n[TARGET_SCHEMA={schema.__name__}]\n"
            "只输出符合该 schema 字段结构的单个 JSON 对象，不要包含任何解释文字或 markdown 代码块标记。"
            f"{_enum_cheatsheet(schema)}"
        )
        prompt = user_prompt + marker
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            raw = self.provider.complete(self.system_prompt, prompt)
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                prompt = (
                    user_prompt
                    + f"\n\n第 {attempt} 次输出未通过校验，错误信息：{exc}\n"
                    + "请修正后重新只输出合法 JSON。"
                    + marker
                )
        raise AgentGenerationError(
            f"{self.__class__.__name__} 在 {self.max_retries} 次重试后仍未生成合法的 {schema.__name__} JSON：{last_error}"
        )
