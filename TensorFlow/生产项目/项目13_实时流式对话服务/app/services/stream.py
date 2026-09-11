"""流式生成:stub(把固定回复逐字吐出,离线) | real(真 LLM 流式)。"""
from ..core.exceptions import ValidationError
def stream_reply(message: str):
    """生成器:逐 token(这里逐字)产出,模拟打字机流式。"""
    if not (message or "").strip(): raise ValidationError("message 不能为空")
    reply = f"收到:{message}。这是流式逐字返回的演示回复。"
    for ch in reply:
        yield ch
