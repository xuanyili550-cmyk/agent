from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STR_", env_file=".env", extra="ignore")
    app_name: str = "实时流式对话服务(WebSocket)"
    debug: bool = False
    backend: str = "stub"
    host: str = "127.0.0.1"; port: int = 8000; cors_origins: list[str] = ["*"]
@lru_cache
def get_settings(): return Settings()
