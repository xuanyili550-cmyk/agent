"""配置:后端可切,默认 stub → 离线可跑可测。"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPX_", env_file=".env", extra="ignore")
    app_name: str = "语音服务(ASR/TTS)"
    debug: bool = False
    backend: str = "stub"            # stub(离线) | real(🔴/🟡 需模型/GPU)
    host: str = "127.0.0.1"; port: int = 8000; cors_origins: list[str] = ["*"]
@lru_cache
def get_settings(): return Settings()
