"""LLM 多后端网关:stub/openai/mlx;调用失败自动降级(fallback)到 stub,保证不中断。"""
from ..core.config import get_settings
from ..core.logging import get_logger
log = get_logger("llm"); _M = {}


def _stub(model, messages):
    last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return f"[stub:{model}] 收到:{last[:100]}"


def _openai(model, messages):
    from openai import OpenAI
    s = get_settings()
    cli = OpenAI(base_url=s.openai_base, api_key="EMPTY")
    return cli.chat.completions.create(model=model, messages=messages).choices[0].message.content


def _mlx(model, messages):
    from mlx_lm import generate, load
    if "m" not in _M: _M["m"], _M["t"] = load(model)
    prompt = "\n".join(f'{m["role"]}: {m["content"]}' for m in messages) + "\nassistant:"
    return generate(_M["m"], _M["t"], prompt=prompt, max_tokens=256)


def chat(model, messages):
    """按配置后端生成;真实后端失败 → 降级 stub(服务不挂)。"""
    b = get_settings().backend
    fn = {"openai": _openai, "mlx": _mlx}.get(b)
    if fn is None:
        return _stub(model, messages)
    try:
        return fn(model, messages)
    except Exception as e:                        # 降级兜底
        log.warning("后端 %s 失败,降级 stub:%s", b, e)
        return _stub(model, messages)
