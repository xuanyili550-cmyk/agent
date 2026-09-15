"""本地、免费、开放权重的语音转文字（ASR）：Whisper（Apache-2.0/MIT，无需 API key）。

输出适配成 09_POST/subtitle/subtitle.py 的 cues_from_whisper_result() 期望的
{"segments": [{"start","end","text"}]} 结构，这样 ASR 结果可以直接烧成字幕，
不需要经过任何付费 API。

流水线位置：07_GENERATION 音频子模块的"反向"环节——TTS 生成配音后，用 ASR 把带时间戳的文本
识别回来，供 09_POST 生成字幕、供 08_QC 做台词对齐校验。
为什么也做 Provider 抽象：与其他生成器一致，方便以后接云端 ASR 或换 faster-whisper 等实现，
调用方（后期 / QC）不必改代码。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseASRProvider(ABC):
    """ASR provider 的抽象接口：给音频路径，返回文本和分段时间戳。"""

    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """返回 {"text": str, "segments": [{"start": float, "end": float, "text": str}, ...]}。"""


class WhisperLocalASRProvider(BaseASRProvider):
    """通过 transformers 在本地运行 openai/whisper-large-v3（或任意 Whisper checkpoint）——
    无需 API key；较长音频要有 GPU 才有合理速度（CPU 能跑但很慢）。见 06_MODELS/asr/registry.json。"""

    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str = "cuda"):
        """只记录配置，不加载模型；权重延迟到第一次 transcribe 时再加载，避免 import 就占显存。"""
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load(self):
        """懒加载并缓存 transformers ASR pipeline；``return_timestamps=True`` 才会给出分段时间戳。"""
        if self._pipe is not None:
            return self._pipe
        from transformers import pipeline

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            return_timestamps=True,
        )
        return self._pipe

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """识别音频并把 transformers 的 ``chunks`` 转成字幕模块需要的 ``segments`` 结构。"""
        pipe = self._load()
        generate_kwargs = {"language": language} if language else {}
        result = pipe(audio_path, generate_kwargs=generate_kwargs)
        segments = [
            {"start": chunk["timestamp"][0], "end": chunk["timestamp"][1], "text": chunk["text"]}
            for chunk in result.get("chunks", [])
            # Whisper 对被截断的最后一段会给 end=None，这种段没法做字幕，直接丢弃
            if chunk.get("timestamp", (None, None))[1] is not None
        ]
        return {"text": result.get("text", ""), "segments": segments}
