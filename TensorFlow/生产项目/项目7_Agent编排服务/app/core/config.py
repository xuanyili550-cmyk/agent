"""配置:后端可切,默认 stub → 离线可跑可测。"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGT_", env_file=".env", extra="ignore")
    app_name: str = "Agent 编排服务"
    debug: bool = False
    backend: str = "stub"            # stub(离线) | real(🔴/🟡 需模型/GPU)
    host: str = "127.0.0.1"; port: int = 8000; cors_origins: list[str] = ["*"]
@lru_cache
def get_settings(): return Settings()
