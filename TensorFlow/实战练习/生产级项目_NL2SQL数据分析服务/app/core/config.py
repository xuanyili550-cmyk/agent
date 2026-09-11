# 【构建 1/16 · 地基】无内部依赖，最先写（所有模块都读它）
"""应用配置：pydantic-settings 从环境变量/.env 读取，支持多环境、后端可切换。"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    # —— 应用 ——
    app_name: str = "NL2SQL 数据分析服务"
    env: str = "dev"                       # dev / prod
    log_level: str = "INFO"
    cors_origins: str = "*"                # 逗号分隔

    # —— 数据库(SQLAlchemy DSN)：本机 SQLite；生产改 mysql+pymysql://user:pwd@host/db ——
    database_url: str = "sqlite:///./nl2sql.db"
    db_seed: bool = True                   # 启动时灌示例数据(演示用)

    # —— LLM 后端：stub(测试,确定性) / mlx(本机 Mac) / openai(vLLM/Ollama/云端 OpenAI 兼容) ——
    llm_backend: str = "mlx"
    llm_model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    openai_base_url: str = "http://localhost:8001/v1"
    openai_api_key: str = "EMPTY"
    openai_model: str = "qwen"

    # —— NL2SQL 策略 ——
    sql_max_retries: int = 2               # SQL 生成/校验失败重试次数
    query_timeout_s: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
