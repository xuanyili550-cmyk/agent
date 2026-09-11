"""LLM:可切后端。stub(抽取式:据检索片段拼答案,离线) | mlx(本机) | openai(兼容服务)。"""
from ..core.config import get_settings
_M = {}


def _stub(query, context):
    return f"根据知识库资料,关于「{query}」:\n{context[:300]}" if context else "未在知识库找到相关资料。"


def _openai(query, context):
    from openai import OpenAI
    s = get_settings()
    cli = OpenAI(base_url=s.openai_base, api_key="EMPTY")
    prompt = f"只根据以下资料回答,无据说不知道。\n资料:\n{context}\n\n问题:{query}"
    r = cli.chat.completions.create(model=s.llm_model, messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


def _mlx(query, context):
    from mlx_lm import generate, load
    s = get_settings()
    if "m" not in _M:
        _M["m"], _M["t"] = load(s.llm_model)
    prompt = f"据资料回答,无据说不知道。资料:{context}\n问题:{query}\n答:"
    return generate(_M["m"], _M["t"], prompt=prompt, max_tokens=256)


def answer(query: str, context: str) -> str:
    b = get_settings().llm_backend
    return {"openai": _openai, "mlx": _mlx}.get(b, _stub)(query, context)
