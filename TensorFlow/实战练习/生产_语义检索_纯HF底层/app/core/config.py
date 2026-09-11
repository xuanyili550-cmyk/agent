"""应用配置：pydantic-settings。生产改环境变量即切模型/池化/设备，不动代码。"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMB_", env_file=".env", extra="ignore")

    app_name: str = "语义检索服务 · 纯 HF 底层"
    debug: bool = False

    # —— 嵌入模型(只用 AutoTokenizer + AutoModel，不用 pipeline / sentence-transformers) ——
    #   不同模型的"池化方式"不同：MiniLM/E5 用 mean、bge 用 cls —— 这是底层原理，必须配对。
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    pooling: str = "mean"               # mean | cls
    normalize: bool = True              # L2 归一化后，点积=余弦相似度(检索地基)
    query_prefix: str = ""              # 有的模型要前缀，如 E5 的 "query: " / bge 中文检索指令
    passage_prefix: str = ""
    max_seq_len: int = 256
    batch_size: int = 32                # 动态批处理：一批一起前向，GPU 利用率高

    # —— 重排(可选)：cross-encoder，AutoModelForSequenceClassification → 单 logit 打分 ——
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_max_seq_len: int = 512

    device: str = "auto"                # auto → mps/cuda/cpu

    # —— 输入限制 ——
    max_texts_per_request: int = 128
    max_text_chars: int = 4000
    default_top_k: int = 5

    # —— 向量库持久化 ——
    index_dir: str = "/tmp/emb_index"

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
