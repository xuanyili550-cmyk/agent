"""文字转语音（TTS）生成：可插拔的 provider 抽象。

与 video_generator.py 相同的模式：具体 provider 是 API 客户端占位实现，从环境变量读 key，
缺失则抛 NotConfiguredError。

流水线位置：07_GENERATION 音频环节——拿 voice_cloner.py 得到的 voice_id 和剧本台词合成配音，
产物交给 lipsync_generator.py 对口型、asr_transcriber.py 反查时间戳做字幕、08_QC 做响度 / 静音检查。
为什么做 Provider 抽象：云端（ElevenLabs）、本地开源（Bark）、本地克隆音色（Coqui XTTS）三条路线
许可与依赖差异巨大，统一成 ``synthesize(text, voice_id) -> 音频路径`` 后可以按项目许可要求随时切换。
"""

from __future__ import annotations

import os
import sys
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asset_registry import AssetRecord, new_asset_id, now_iso, write_asset_record  # noqa: E402
from errors import NotConfiguredError  # noqa: E402
from http_retry import request_with_retry  # noqa: E402


class BaseTTSProvider(ABC):
    """TTS provider 的抽象接口。"""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        """返回生成音频的本地文件路径。"""


class ElevenLabsTTSProvider(BaseTTSProvider):
    """ElevenLabs 文字转语音 API：POST 文本，响应体直接就是 mp3 字节流。"""

    API_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    def __init__(self):
        """从 ELEVENLABS_API_KEY 读取凭证；缺失立即报错。"""
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise NotConfiguredError("ELEVENLABS_API_KEY is not set")

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        """调用 multilingual_v2 模型合成，把返回的 mp3 写到 output_dir 下随机文件名。"""
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {"text": text, "model_id": "eleven_multilingual_v2"}  # multilingual 才支持中文台词

        response = request_with_retry("POST", self.API_URL_TEMPLATE.format(voice_id=voice_id), json=payload, headers=headers)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4().hex}.mp3"
        out_path.write_bytes(response.content)
        return str(out_path)


class BarkLocalTTSProvider(BaseTTSProvider):
    """通过 transformers 在本地运行 Suno Bark（MIT，免费且可商用）——无需 API key。
    voice_id 对应 Bark 的说话人预设，如 "v2/en_speaker_6" 或 "v2/zh_speaker_4"
    （完整列表见 https://huggingface.co/suno/bark）。"""

    def __init__(self, model_id: str = "suno/bark-small", device: str = "cuda"):
        """只记录配置；pipeline 延迟到第一次 synthesize 时再加载。"""
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load(self):
        """懒加载并缓存 transformers text-to-speech pipeline。"""
        if self._pipe is not None:
            return self._pipe
        from transformers import pipeline

        self._pipe = pipeline("text-to-speech", model=self.model_id, device=self.device)
        return self._pipe

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        """用指定说话人预设合成，输出 wav；采样率取模型返回值而非写死。"""
        import scipy.io.wavfile

        pipe = self._load()
        preprocess_params = {"voice_preset": voice_id} if voice_id else {}
        output = pipe(text, preprocess_params=preprocess_params)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4().hex}.wav"
        # Bark 返回 (1, n) 的二维数组，squeeze 成一维才是 wavfile.write 期望的单声道格式
        scipy.io.wavfile.write(str(out_path), rate=output["sampling_rate"], data=output["audio"].squeeze())
        return str(out_path)


class CoquiXTTSLocalTTSProvider(BaseTTSProvider):
    """用 Coqui XTTS-v2 以克隆音色朗读文本：voice_id 是 CoquiXTTSLocalVoiceCloneProvider.clone_voice()
    产出的、用 "::" 拼接的参考 wav 路径。本地免费运行、无需 API key——但注意
    CoquiXTTSLocalVoiceCloneProvider 处的许可说明：默认非商用。

    在本项目主环境（`transformers` 5.5.0）下**不可用**：`pip install coqui-tts`（0.27.5）
    import 时就失败——`ImportError: cannot import name 'isin_mps_friendly' from
    'transformers.pytorch_utils'`——因为 coqui-tts 的 XTTS 代码 import 了一个 transformers 5.5.0
    已不再导出的辅助函数。在单独的 venv 里已**端到端验证可用**（真实生成 5.2s wav，并用 ffprobe 交叉核对）：

        python3 -m venv .venv_xtts && source .venv_xtts/bin/activate
        pip install "transformers<5.0" coqui-tts pydantic
        pip install torch torchaudio
        pip install "coqui-tts[codec]"   # 会带上 torchcodec

    该 venv 在 macOS 上还有两个已实际验证的坑：
    1. torchcodec 自带的原生库无法针对本机 Homebrew ffmpeg 完成 dlopen
       （`OSError: ... Library not loaded: @rpath/libavutil.57.dylib ... no LC_RPATH's found`），
       需要把加载器指向它：`export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib"`。
    2. 首次下载 XTTS-v2 会卡在一个交互式 TOS 提示（模型采用非商用的 CPML，见上面的许可说明）——
       `COQUI_TOS_AGREED=1` 可以跳过提示，但只有在你本人确实阅读并同意 https://coqui.ai/cpml
       之后才应设置；本仓库不会替你设置。

    本代码库不做补丁（这是两个第三方库之间的上游版本兼容缺口，不是本类的 bug）；主环境保持
    transformers 5.5.0 以支持 LocalTransformersProvider/BarkLocalTTSProvider/WhisperLocalASRProvider，
    与项目 README 里记录的 Python 3.14/`datasets`/`dill` 绕过方案是同一思路。"""

    def __init__(self, model_id: str = "tts_models/multilingual/multi-dataset/xtts_v2", language: str = "zh", device: str = "cuda"):
        """只记录配置；TTS 对象延迟到第一次 synthesize 时再加载。"""
        self.model_id = model_id
        self.language = language
        self.device = device
        self._tts = None

    def _load(self):
        """懒加载并缓存 Coqui TTS 对象（import TTS.api 本身就很重，务必延迟）。"""
        if self._tts is not None:
            return self._tts
        from TTS.api import TTS

        self._tts = TTS(self.model_id).to(self.device)
        return self._tts

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ) -> str:
        """把 "::" 拼接的 voice_id 拆回参考 wav 列表，零样本克隆音色合成并写 wav。"""
        tts = self._load()
        speaker_wavs = voice_id.split("::")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{uuid.uuid4().hex}.wav"
        tts.tts_to_file(text=text, speaker_wav=speaker_wavs, language=self.language, file_path=str(out_path))
        return str(out_path)


class TTSGenerator:
    """TTS 门面：调用 provider 并把产物登记到素材台账。"""

    def __init__(self, provider: BaseTTSProvider, asset_log_path: str = "./outputs/asset_records.jsonl"):
        """注入 provider 与台账路径。"""
        self.provider = provider
        self.asset_log_path = asset_log_path

    def synthesize(
        self,
        text: str,
        voice_id: str,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
        character_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        shot_id: Optional[str] = None,
        license: Optional[str] = None,
    ) -> str:
        """合成配音并写 AssetRecord（prompt 字段存台词原文）；返回产物路径。"""
        file_path = self.provider.synthesize(text=text, voice_id=voice_id, output_dir=output_dir, seed=seed)
        record = AssetRecord(
            asset_id=new_asset_id("tts"),
            file_path=file_path,
            character_id=character_id,
            episode_id=episode_id,
            shot_id=shot_id,
            model=type(self.provider).__name__,
            prompt=text,
            seed=seed,
            license=license,
            created_at=now_iso(),
        )
        write_asset_record(record, self.asset_log_path)
        return file_path
