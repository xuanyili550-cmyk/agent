"""语音:stub(ASR 确定性转写/TTS 估算,离线) | real(🟡 Whisper/TTS 模型)。"""
import hashlib
from ..core.config import get_settings
from ..core.exceptions import ValidationError
def asr(audio_b64: str):
    if not audio_b64: raise ValidationError("audio 不能为空")
    if get_settings().backend == "real":
        raise RuntimeError("real 后端需 Whisper 模型;本机用 stub")
    return {"text": f"[stub 转写] 音频长度特征 {len(audio_b64)}"}
def tts(text: str):
    if not text: raise ValidationError("text 不能为空")
    sr = 22050; dur = max(0.5, len(text) * 0.15)
    return {"sample_rate": sr, "duration_s": round(dur, 2), "bytes_est": int(sr * dur * 2)}
