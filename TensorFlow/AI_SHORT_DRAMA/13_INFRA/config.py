"""统一配置：所有环境变量只在这里声明一次，其他模块通过 ``get_settings()`` 读取。

为什么：之前 DATABASE_URL / CELERY_* / STORAGE_* / LLM_CONTEXT_* 散在七八个文件里各自
``os.environ.get``，同一个变量的默认值在不同文件里还不一样，上线时根本说不清"到底要配哪些"。
pydantic-settings 把全部配置集中成一个带类型、带默认值、带校验的对象，``.env.example``
就是从这里的字段一一对应生成的。

秘钥约定（和 Docker secrets / K8s secret 挂载方式一致）：任何字段都可以用 ``<NAME>_FILE``
指向一个文件，启动时读文件内容作为值，避免把密钥明文写进环境变量或 compose 文件。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _apply_file_secrets() -> None:
    """把 ``XXX_FILE=/run/secrets/xxx`` 形式的变量读成 ``XXX=<文件内容>``。"""
    for key, path in list(os.environ.items()):
        if not key.endswith("_FILE") or key[:-5] in os.environ:
            continue
        p = Path(path)
        if p.is_file():
            os.environ[key[:-5]] = p.read_text(encoding="utf-8").strip()


class Settings(BaseSettings):
    """全部运行配置。字段名即环境变量名（大写），默认值就是本地开发能跑的值。"""

    model_config = SettingsConfigDict(env_file=os.environ.get("ENV_FILE", ".env"), env_file_encoding="utf-8", extra="ignore")

    # ---- 运行环境 ----
    app_env: Literal["dev", "test", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False  # 生产建议 true：一行一个 JSON，方便 Loki/ELK 解析
    artifacts_root: Path = Field(default=_PROJECT_ROOT / "artifacts")  # 生成产物根目录（图/视频/manifest）

    # ---- 数据库 / 队列 ----
    database_url: str = "sqlite:///./13_infra_dev.db"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str | None = None  # None 表示与 broker 相同
    celery_task_always_eager: bool = False  # 只给本地开发/测试用，生产必须 false
    task_soft_time_limit_seconds: int = 3300
    task_time_limit_seconds: int = 3600
    task_max_retries: int = 3

    # ---- 对象存储 ----
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_root: str = "./storage_data"
    storage_bucket: str | None = None
    storage_endpoint_url: str | None = None
    storage_access_key: str | None = None
    storage_secret_key: str | None = None
    storage_region: str | None = None

    # ---- LLM ----
    llm_provider: Literal["mock", "local", "anthropic", "openai"] = "local"
    llm_model: str | None = None  # None 取各 provider 默认
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    # 多模型降级（坑 4）：主模型限流/超时/没配 key 时按顺序切到这些备用模型；模型名前缀决定后端
    llm_fallback_models: Annotated[list[str], NoDecode] = Field(default_factory=list)
    llm_fallback_cooldown_seconds: float = 60.0  # 某个模型失败后多久内直接跳过它（熔断冷却）
    # 会话记忆存储：Redis（多 worker）> SQLite（单机可查询）> 本地 JSON 文件；按这个优先级取第一个配置了的
    llm_context_redis_url: str | None = None
    llm_context_sqlite_path: str | None = None
    llm_context_dir: str | None = None
    llm_context_ttl_seconds: int | None = None
    llm_context_lock_timeout_seconds: int = 300
    # ToolAgent（工具循环）：坑 1 的轮次上限；坑 3 的工具级确认门——白名单里的工具才允许自动执行
    agent_max_tool_rounds: int = 5
    agent_tool_allowlist: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # 每 1k token 的价格（美元），用来估算成本；模型名前缀匹配，查不到按 0 计
    llm_price_per_1k_input: dict[str, float] = Field(
        default={"claude-sonnet": 0.003, "claude-opus": 0.015, "claude-haiku": 0.0008, "gpt-4o-mini": 0.00015, "gpt-4o": 0.0025}
    )
    llm_price_per_1k_output: dict[str, float] = Field(
        default={"claude-sonnet": 0.015, "claude-opus": 0.075, "claude-haiku": 0.004, "gpt-4o-mini": 0.0006, "gpt-4o": 0.01}
    )

    # ---- 生成 / QC / 重试阶梯 ----
    image_backend: str = "dummy"  # "dummy" 或 diffusers 模型 id
    image_width: int = 1080
    image_height: int = 1920
    qc_backend: Literal["clip", "none"] = "clip"
    qc_max_attempts: int = 3  # 三级重试：同参数 -> 改写提示词 -> 降分辨率
    qc_character_min_similarity: float = 0.75
    retry_resolution_ladder: Annotated[list[tuple[int, int]], NoDecode] = Field(default=[(1080, 1920), (720, 1280), (540, 960)])
    require_human_review: bool = True  # 剧本阶段结束后是否等待人工/总编审批准再进入生产
    ai_editor_can_approve: bool = False  # true：总编审 Agent 通过即视为批准，不再等人

    # ---- API ----
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)  # 逗号分隔；为空时受保护接口一律 503，宁可拒绝也不裸奔
    rate_limit_per_minute: int = 120
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # ---- 可观测性 ----
    metrics_enabled: bool = True
    worker_metrics_port: int = 9100

    # ---- 发布 ----
    publish_dry_run: bool = True  # 生产显式关掉才会真的往平台推
    publish_platforms: Annotated[list[str], NoDecode] = Field(default=["youtube", "tiktok"])

    @field_validator("api_keys", "cors_origins", "publish_platforms", "llm_fallback_models", "agent_tool_allowlist", mode="before")
    @classmethod
    def _split_csv(cls, value):
        """环境变量里的逗号分隔字符串 -> 列表（去空白、去空项）。"""
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("retry_resolution_ladder", mode="before")
    @classmethod
    def _parse_ladder(cls, value):
        # 允许 "1080x1920,720x1280,540x960" 这种环境变量写法
        """把 "1080x1920,720x1280" 这种环境变量写法解析成 (宽, 高) 列表。"""
        if isinstance(value, str):
            pairs = []
            for item in value.split(","):
                w, h = item.lower().split("x")
                pairs.append((int(w), int(h)))
            return pairs
        return value

    @property
    def result_backend(self) -> str:
        """Celery 结果后端；没单独配就和 broker 一样。"""
        return self.celery_result_backend or self.celery_broker_url

    @property
    def is_prod(self) -> bool:
        """是否生产环境（决定 eager 队列禁用、create_all 禁用等）。"""
        return self.app_env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例：第一次调用时读文件密钥、构造 Settings 并做生产环境校验。"""
    _apply_file_secrets()
    settings = Settings()
    if settings.is_prod and settings.celery_task_always_eager:
        raise RuntimeError("生产环境禁止 CELERY_TASK_ALWAYS_EAGER=true")
    return settings


def reset_settings_cache() -> None:
    """测试用：改了环境变量后清掉缓存重新加载。"""
    get_settings.cache_clear()
