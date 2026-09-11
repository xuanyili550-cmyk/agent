from __future__ import annotations

import importlib
from typing import Any, Dict

from ..workers.celery_app import celery_app

__all__ = ["llm_task"]

_DEFAULT_LOCAL_MODEL = "microsoft/Phi-3.5-mini-instruct"


@celery_app.task(name="llm_task", bind=True)
def llm_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """LLM GPU queue task (script/dialogue/prompt generation calls into an LLM).

    Expected payload: {"prompt": str, "system": str | None, "model": str | None,
      "params": dict | None, "context_id": str | None}.
    Expected return: {"text": str, "model": str, "usage": dict | None}.

    Calls the real 02_STORY_ENGINE.agents.base providers. payload["model"] selects
    the backend: a name starting with "claude" uses AnthropicProvider (needs
    ANTHROPIC_API_KEY), one starting with "gpt" uses OpenAIProvider (needs
    OPENAI_API_KEY), anything else -- including omitting "model" entirely -- is
    treated as a Hugging Face model id and run locally via LocalTransformersProvider,
    which needs no API key (default: free, MIT-licensed microsoft/Phi-3.5-mini-instruct;
    see 06_MODELS/llm/registry.json for other free options).
    """
    base_mod = importlib.import_module("02_STORY_ENGINE.agents.base")

    model = payload.get("model")
    if model and model.startswith("claude"):
        provider = base_mod.AnthropicProvider(model=model)
    elif model and model.startswith("gpt"):
        provider = base_mod.OpenAIProvider(model=model)
    else:
        provider = base_mod.LocalTransformersProvider(model_id=model or _DEFAULT_LOCAL_MODEL)

    text = provider.complete(payload.get("system") or "", payload["prompt"])
    return {"text": text, "model": model or _DEFAULT_LOCAL_MODEL, "usage": None}
