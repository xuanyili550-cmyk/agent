"""
================================================================================
 服务层 · mlx 高音质 TTS 后端（Kokoro-82M · Apple MLX · 中英双语）
================================================================================
 【为什么要这个】macOS `say` 零下载但音色机械；要"高音质自然音色"就用神经 TTS。
   mlx_audio 是 Apple Silicon 上跑 TTS 的库，Kokoro-82M 是轻量高质量模型，支持中/英多音色。

 【设计要点】
   · 懒加载 + 单例：模型只在第一次真正合成时加载一次(首次会从 HF 下载几百 MB)，之后常驻复用。
     —— 绝不每次请求都重载，否则延迟爆炸。
   · 中英分开配音色：中文段用中文音色(如 zf_xiaobei)、英文段用英文音色(如 af_heart)，读音才准。
   · 只负责"把一段文本合成成一个原始 wav"；采样率/声道统一交给 tts.py 的 _to_std_wav 处理，
     好和 say 后端产物无缝拼接(可混用)。
   · 失败即抛：模型下不动/接口不符就抛异常，由上层 tts.py 自动回退到 say(服务不中断)。
 【怎么开启】配置 TTSAPP_TTS_BACKEND=mlx(默认已是)；pip install mlx-audio。
================================================================================
"""
import glob
import os
import shutil
import tempfile

from ..core.config import get_settings

_MODEL = None   # 单例：加载后的 Kokoro 模型对象


def _model():
    """懒加载模型(线程内首次调用触发下载/加载)，之后复用同一对象。"""
    global _MODEL
    if _MODEL is None:
        from mlx_audio.tts.utils import load_model     # 延迟导入：没装 mlx 也不影响 say 路径
        _MODEL = load_model(get_settings().mlx_model)  # ← 首次从 HF 拉取权重(~几百 MB)
    return _MODEL


def synth_chunk(is_cjk: bool, text: str, raw_wav: str) -> str:
    """把一段(同语种)文本用 mlx 合成成 raw_wav。中文段用中文音色、英文段用英文音色。"""
    s = get_settings()
    from mlx_audio.tts.generate import generate_audio
    voice = s.mlx_zh_voice if is_cjk else s.mlx_en_voice
    lang = s.mlx_zh_lang if is_cjk else s.mlx_en_lang
    work = tempfile.mkdtemp(prefix="mlx_")
    try:
        # generate_audio 把音频写到 output_path 下(file_prefix 为名)；join_audio 合成一条
        generate_audio(text=text, model=_model(), voice=voice, lang_code=lang, speed=s.mlx_speed,
                       audio_format="wav", output_path=work, file_prefix="seg",
                       join_audio=True, save=True, play=False, verbose=False)
        outs = sorted(glob.glob(os.path.join(work, "*.wav")))
        if not outs:   # 某些版本写到 cwd，兜底再找一次
            outs = sorted(glob.glob("seg*.wav"))
        if not outs:
            raise RuntimeError("mlx 未产出音频(检查 voice/lang_code 或 mlx-audio 版本)")
        shutil.move(outs[0], raw_wav)
        return raw_wav
    finally:
        shutil.rmtree(work, ignore_errors=True)
