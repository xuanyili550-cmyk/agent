"""配置:LLM 后端可切 + 限流/缓存参数。默认 stub → 离线可跑可测。"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GW_", env_file=".env", extra="ignore")
    app_name: str = "LLM 推理网关"
    debug: bool = False
    backend: str = "stub"               # stub(离线) | openai | mlx
    openai_base: str = "http://localhost:8001/v1"
    default_model: str = "qwen"
    cache_enabled: bool = True
    rate_limit_per_min: int = 60        # 每客户端每分钟请求上限
    max_prompt_chars: int = 8000
    host: str = "127.0.0.1"; port: int = 8000; cors_origins: list[str] = ["*"]
@lru_cache
def get_settings(): return Settings()
