"""应用配置：pydantic-settings 从环境变量/.env 读取。生产改环境变量即可切参数，不动代码。"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTSAPP_", env_file=".env", extra="ignore")

    app_name: str = "双语配音 · 音频/视频制作服务"
    debug: bool = False

    # —— TTS 后端：mlx(高音质,首次下模型) | say(零下载,系统声,兜底) ——
    #   mlx 失败(模型下不动/接口不符)会自动回退到 say，服务不中断。
    tts_backend: str = "mlx"
    # mlx(Kokoro) 参数：中英各用一个音色 + 语种码(lang_code)。装 mlx-audio 后首次调用会从 HF 下模型。
    #   若报语种码错误，把 mlx_*_lang 改成 'en'/'zh' 试(不同 mlx-audio 版本约定不同)。
    mlx_model: str = "prince-canuma/Kokoro-82M"
    mlx_en_voice: str = "af_heart"      # 英文音色(备选 af_bella/am_adam/bf_emma)
    mlx_zh_voice: str = "zf_xiaobei"    # 中文音色(备选 zf_xiaoni/zm_yunjian)
    mlx_en_lang: str = "a"              # 美式英语(kokoro 语种码；或试 'en')
    mlx_zh_lang: str = "z"              # 普通话(或试 'zh')
    mlx_speed: float = 1.0

    # —— say(系统声音)偏好；启动按系统实际可用做兜底解析(见 services/system.py) ——
    zh_voice: str = "Tingting"          # 备选 Meijia / Sinji
    en_voice: str = "Samantha"          # 备选 Daniel(en_GB) / Alex
    speech_rate: int = 175              # say 语速(词/分)
    audio_sample_rate: int = 22050

    # —— 输入限制(防滥用/防打爆机器) ——
    max_text_chars: int = 5000
    max_sentences: int = 200
    max_word_len: int = 64

    # —— 视频导出 ——
    video_width: int = 1280
    video_height: int = 720
    video_fps: int = 30
    video_bg: str = "#0f172a"
    video_fg: str = "#e2e8f0"
    video_accent: str = "#38bdf8"
    max_export_sentences: int = 60

    # —— 缓存 & 任务 ——
    cache_dir: str = "/tmp/ttsapp_cache"
    cache_ttl_seconds: int = 6 * 3600
    max_concurrent_jobs: int = 2
    job_retention_seconds: int = 3600

    # —— 服务 ——
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
