"""配置:训练后端可切(stub 离线模拟 / trl 真实 GPU)。"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FT_", env_file=".env", extra="ignore")
    app_name: str = "微调与对齐平台"
    debug: bool = False
    train_backend: str = "stub"         # stub(离线模拟,验证流程) | trl(🔴 真实 GPU)
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    epochs: int = 3
    min_samples: int = 2
    registry_dir: str = "/tmp/ft_registry"
    max_concurrent_jobs: int = 1
    host: str = "127.0.0.1"; port: int = 8000; cors_origins: list[str] = ["*"]
@lru_cache
def get_settings(): return Settings()
