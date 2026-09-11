"""配置:阈值/后端可切。默认规则后端 → 离线可跑可测。"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOD_", env_file=".env", extra="ignore")
    app_name: str = "内容审核中台"
    debug: bool = False
    risk_backend: str = "rule"          # rule(离线) | model(🔴 需毒性分类模型)
    block_threshold: float = 0.7        # ≥ 拦截
    review_threshold: float = 0.4       # ≥ 人工复审,否则通过
    max_text_chars: int = 5000
    max_batch: int = 200
    max_concurrent_jobs: int = 2
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings():
    return Settings()
