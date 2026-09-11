"""配置:pydantic-settings。外部依赖都做成"可切后端",默认 stub → 测试离线可跑。"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KB_", env_file=".env", extra="ignore")

    app_name: str = "企业知识库问答平台"
    debug: bool = False

    # 嵌入后端:stub(哈希伪向量,离线/测试) | hf(AutoModel 真嵌入)
    embed_backend: str = "stub"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 64                 # stub 后端的维度

    # LLM 后端:stub(拼接片段,离线/测试) | mlx(本机) | openai(兼容服务)
    llm_backend: str = "stub"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    openai_base: str = "http://localhost:8001/v1"

    # 切块/检索
    chunk_size: int = 200
    chunk_overlap: int = 40
    top_k: int = 4
    rerank_enabled: bool = False

    # 限制/缓存/任务
    max_doc_chars: int = 200_000
    max_query_chars: int = 1000
    cache_dir: str = "/tmp/kb_cache"
    max_concurrent_jobs: int = 2

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
