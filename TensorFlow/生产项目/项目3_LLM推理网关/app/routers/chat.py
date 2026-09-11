from fastapi import APIRouter, Header
from ..core.config import get_settings
from ..core.exceptions import RateLimitError, ValidationError
from ..schemas.chat import ChatRequest, ChatResponse
from ..services import cache, llm, ratelimit
router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_client_id: str = Header(default="anon")):
    s = get_settings()
    if not ratelimit.check(x_client_id):                 # 限流
        raise RateLimitError("请求过于频繁,请稍后")
    msgs = [m.model_dump() for m in req.messages]
    total = sum(len(m["content"]) for m in msgs)
    if total > s.max_prompt_chars:
        raise ValidationError("prompt 过长")
    model = req.model or s.default_model
    k = cache.key(model, msgs)
    if s.cache_enabled and (hit := cache.get(k)):        # 缓存命中
        return ChatResponse(model=model, content=hit, cached=True)
    content = llm.chat(model, msgs)
    if s.cache_enabled:
        cache.put(k, content)
    return ChatResponse(model=model, content=content, cached=False)
