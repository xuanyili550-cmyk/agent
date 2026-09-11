from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVAL_", env_file=".env", extra="ignore")
    app_name: str = "LLM 评测服务(as-judge)"
    debug: bool = False
    backend: str = "stub"
    host: str = "127.0.0.1"; port: int = 8000; cors_origins: list[str] = ["*"]
@lru_cache
def get_settings(): return Settings()
