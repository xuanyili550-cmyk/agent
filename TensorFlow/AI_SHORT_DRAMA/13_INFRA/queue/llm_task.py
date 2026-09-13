from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from ..workers.celery_app import celery_app

__all__ = ["llm_task", "build_conversation_store"]

_DEFAULT_LOCAL_MODEL = "microsoft/Phi-3.5-mini-instruct"


def build_conversation_store():
    """按环境变量选历史消息的存储后端（和 STORAGE_BACKEND 的思路一致，生产/开发只改配置）。

    - ``LLM_CONTEXT_REDIS_URL`` 非空：Redis（多 worker 共享；``LLM_CONTEXT_TTL_SECONDS`` 可选，
      不设即永不过期——会话历史"长久"保留，直到显式清除）；
    - 否则：本地 JSON 文件，目录取 ``LLM_CONTEXT_DIR``，默认系统临时目录下 ``ai_short_drama_contexts``。
    """
    memory_mod = importlib.import_module("02_STORY_ENGINE.agents.memory")
    redis_url = os.environ.get("LLM_CONTEXT_REDIS_URL")
    if redis_url:
        ttl_raw = os.environ.get("LLM_CONTEXT_TTL_SECONDS")
        ttl = int(ttl_raw) if ttl_raw else None
        return memory_mod.RedisConversationStore(redis_url, ttl_seconds=ttl)
    root = os.environ.get("LLM_CONTEXT_DIR") or str(Path(tempfile.gettempdir()) / "ai_short_drama_contexts")
    return memory_mod.FileConversationStore(root)


@celery_app.task(name="llm_task", bind=True)
def llm_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """LLM GPU 队列任务（剧本 / 对白 / 提示词生成都走这里调 LLM）。

    payload: {"prompt": str, "system": str | None, "model": str | None,
      "params": dict | None, "context_id": str | None}。
    返回: {"text": str, "model": str, "usage": dict | None, "context_id": str | None,
      "history_turns": int}。

    真正调用 02_STORY_ENGINE.agents.base 里的 provider。payload["model"] 决定后端：
    "claude" 开头走 AnthropicProvider（需要 ANTHROPIC_API_KEY），"gpt" 开头走
    OpenAIProvider（需要 OPENAI_API_KEY），其他情况（包括不传 "model"）都当作
    Hugging Face 模型 id 用 LocalTransformersProvider 本地跑，不需要 API key
    （默认免费、MIT 许可的 microsoft/Phi-3.5-mini-instruct；其他免费选项见
    06_MODELS/llm/registry.json）。

    传了 ``context_id`` 就是一个长期会话：这次调用会带上该会话之前所有轮次的历史消息，
    并把本轮追加进持久化存储；只有历史逼近模型上下文上限时才裁掉最旧的轮次。
    ``params["compaction"]`` 可选 "truncate"（默认）或 "summarize"（触顶时先压成摘要）。
    不传 ``context_id`` 则退化为无状态单轮调用。
    """
    base_mod = importlib.import_module("02_STORY_ENGINE.agents.base")

    model = payload.get("model")
    if model and model.startswith("claude"):
        provider = base_mod.AnthropicProvider(model=model)
    elif model and model.startswith("gpt"):
        provider = base_mod.OpenAIProvider(model=model)
    else:
        provider = base_mod.LocalTransformersProvider(model_id=model or _DEFAULT_LOCAL_MODEL)

    system = payload.get("system") or ""
    prompt = payload["prompt"]
    params = payload.get("params") or {}
    context_id = payload.get("context_id")

    if not context_id:
        text = provider.complete(system, prompt)
        return {"text": text, "model": model or _DEFAULT_LOCAL_MODEL, "usage": None, "context_id": None, "history_turns": 0}

    memory = base_mod.build_memory(
        provider,
        context_id,
        build_conversation_store(),
        compaction=params.get("compaction", "truncate"),
    )
    history = memory.window(system, prompt)
    text = provider.complete(system, prompt, history)
    memory.add_turn(prompt, text)
    return {
        "text": text,
        "model": model or _DEFAULT_LOCAL_MODEL,
        "usage": None,
        "context_id": context_id,
        "history_turns": memory.turn_count,
    }
