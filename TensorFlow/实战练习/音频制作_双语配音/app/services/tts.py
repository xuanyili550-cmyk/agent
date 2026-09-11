"""
================================================================================
 服务层 · TTS 文本转语音（中英混排 · 可切后端：mlx 高音质 / say 零下载兜底）
================================================================================
 【这个文件做什么】把一段中英混排文本变成一条 wav/mp3 音频。

 【核心难点：中英混排怎么读准】
   一个英文音色去读中文会读成乱码般的音、中文音色读英文也别扭。所以不能整段用一个声音——
   本文件把文本【按脚本切段】：连续中文一段、连续英文一段，各配对应语种的音色，再拼成一条。

 【两个后端(可切换，配置 TTSAPP_TTS_BACKEND)】
   · mlx  ：Kokoro-82M 神经 TTS，音色自然(高音质)；首次调用下模型(~几百 MB)。见 tts_mlx.py。
   · say  ：macOS 系统语音，零下载、稳定；音色机械。作为默认兜底——mlx 失败自动回退，服务不崩。

 【两个工程细节(面试也会问)】
   ① 安全：say 用 `-f 文件` 传文本、subprocess 全列表参数 → 无命令注入/不怕特殊字符与超长。
   ② 性能：内容哈希缓存——相同文本+后端+音色只合成一次(见 cache.py)，缓存键含后端避免串音色。
 【统一规格】不管哪个后端，产物都过 _to_std_wav 统一成 22050Hz/单声道/16bit，才能被 ffmpeg 无损拼接。
================================================================================
"""
import os
import re
import shutil
import subprocess
import tempfile

from ..core.config import get_settings
from ..core.exceptions import AppError
from ..core.logging import get_logger
from . import cache
from .system import resolve_voices

log = get_logger("tts")

# 中文(含中日韩统一表意文字)字符范围——判断某字符是不是中文
_CJK = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


def _runs(text: str):
    """把混排文本切成 [(is_cjk, chunk)]：连续中文一段、连续非中文一段。
    例：'Hello 你好world' → [(False,'Hello '),(True,'你好'),(False,'world')]。"""
    out, cur, cur_cjk = [], "", None
    for ch in text:
        c = bool(_CJK.match(ch))
        if cur and c != cur_cjk:          # 脚本切换 → 收束当前段
            out.append((cur_cjk, cur))
            cur = ""
        cur += ch
        cur_cjk = c
    if cur.strip():
        out.append((cur_cjk, cur))
    return [(k, c) for k, c in out if c.strip()]


def _to_std_wav(src: str, dst: str):
    """统一成 22050Hz/单声道/16bit —— 各后端产物规格一致，才能用 concat 无损拼接。"""
    s = get_settings()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", str(s.audio_sample_rate),
                    "-ac", "1", "-c:a", "pcm_s16le", dst], check=True, capture_output=True)


def _say_segment(voice: str, text: str, wav_path: str, workdir: str):
    """say 后端：指定音色读一段 → aiff → 统一 wav。
    文本走 `say -f 文件` 而非命令行参数——防注入/防特殊字符/防超长。"""
    s = get_settings()
    txt = os.path.join(workdir, "seg.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(text)
    aiff = wav_path + ".aiff"
    subprocess.run(["say", "-v", voice, "-r", str(s.speech_rate), "-f", txt, "-o", aiff],
                   check=True, capture_output=True)
    _to_std_wav(aiff, wav_path)
    os.remove(aiff)


def _synth_chunk(is_cjk: bool, text: str, wav_path: str, workdir: str):
    """把一段(同语种)文本合成到 wav_path。按配置选后端：mlx 优先，失败回退 say。
    ★ 换 TTS 引擎(云 API 等)只要在这里加一个分支，其它代码不用动。"""
    s = get_settings()
    if s.tts_backend == "mlx":
        try:
            from . import tts_mlx
            raw = os.path.join(workdir, "raw_mlx.wav")
            tts_mlx.synth_chunk(is_cjk, text, raw)      # mlx 高音质合成
            _to_std_wav(raw, wav_path)                  # 统一规格便于拼接
            os.remove(raw)
            return
        except Exception as e:                          # noqa: BLE001 模型下不动/接口不符…
            log.warning("mlx TTS 失败，本段回退 say：%s", e)   # 回退保证服务不中断
    # say 后端(默认兜底)：需要系统中/英音色
    zh, en = resolve_voices()
    _say_segment(zh if is_cjk else en, text, wav_path, workdir)


def _voice_params() -> dict:
    """缓存键里的音色相关参数——不同后端/音色算不同音频，避免命中串了音色的旧缓存。"""
    s = get_settings()
    if s.tts_backend == "mlx":
        return {"backend": "mlx", "model": s.mlx_model, "env": s.mlx_en_voice,
                "zhv": s.mlx_zh_voice, "speed": s.mlx_speed}
    zh, en = resolve_voices()
    return {"backend": "say", "zh": zh, "en": en, "rate": s.speech_rate}


def synth_wav(text: str) -> str:
    """合成整段(中英混排)为 wav，返回缓存路径。相同文本+后端命中缓存直接返回。"""
    key = cache.key_for("wav", text, _voice_params())
    hit = cache.get(key, "wav")
    if hit:
        return hit
    out = cache.path_for(key, "wav")
    work = tempfile.mkdtemp(prefix="tts_")
    try:
        parts = []
        for i, (is_cjk, chunk) in enumerate(_runs(text)):      # 按语种分段合成
            p = os.path.join(work, f"p{i}.wav")
            _synth_chunk(is_cjk, chunk, p, work)
            parts.append(p)
        if not parts:
            raise AppError("无可合成内容。")
        if len(parts) == 1:
            shutil.copy(parts[0], out)
        else:                                            # 多段 → concat 无损拼接(规格已统一)
            listf = os.path.join(work, "list.txt")
            with open(listf, "w") as f:
                for p in parts:
                    f.write(f"file '{p}'\n")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                            "-i", listf, "-c", "copy", out], check=True, capture_output=True)
        return out
    except subprocess.CalledProcessError as e:
        raise AppError(f"TTS 合成失败：{(e.stderr or b'').decode('utf-8', 'ignore')[:200]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)          # ★ 成功失败都清理临时目录


def synth_mp3(text: str) -> str:
    """播放用 mp3(比 wav 小，前端加载快)，同样走缓存。"""
    wav = synth_wav(text)
    key = cache.key_for("mp3", text, {"src": os.path.basename(wav)})
    hit = cache.get(key, "mp3")
    if hit:
        return hit
    out = cache.path_for(key, "mp3")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-c:a", "libmp3lame",
                    "-b:a", "192k", out], check=True, capture_output=True)
    return out


def duration(path: str) -> float:
    """用 ffprobe 读时长(秒)——视频分段对齐进度条要用。"""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0
