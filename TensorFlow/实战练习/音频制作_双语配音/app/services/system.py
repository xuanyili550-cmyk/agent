"""
================================================================================
 服务层 · 系统资源探测（声音 / 字体 / ffmpeg —— 启动即检，缺了明确报错）
================================================================================
 【为什么要有这个文件】本服务依赖三样"系统级"东西：macOS 语音(say)、含中文+IPA 字形的字体、
   命令行 ffmpeg/ffprobe。它们不是 pip 装的，可能缺失。与其"跑到一半才崩、报一堆看不懂的栈"，
   不如【启动/使用前先探测】：能用就解析出实际值，缺了就抛清晰的 DependencyError(503)。

 【三个探测的要点】
   ① 声音：系统装了哪些 say 声音是不确定的(中文声音可能没下载)。所以按"偏好→同语系兜底→报错"
      解析：想要 Tingting 没有就找任意中文声音；连中文都没有就退到英文并告警(总比崩好)。
   ② 字体：中文和 IPA 音标都要能渲染，普通字体会缺字形变"豆腐块"。按候选列表挑第一个存在的。
   ③ 二进制：ffmpeg/ffprobe 缺失直接报"brew install ffmpeg"。
   探测结果用 lru_cache 缓存——只在启动/首次用时跑一次，不反复调命令行。

 【健康检查】healthcheck() 不抛异常、返回各依赖状态，供 /health 展示(ok / degraded)。
================================================================================
"""
import os
import shutil
import subprocess
from functools import lru_cache

from ..core.config import get_settings
from ..core.exceptions import DependencyError
from ..core.logging import get_logger

log = get_logger("system")

# 候选字体(按优先级)：都含中文+IPA 字形；取第一个系统里存在的
_FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


@lru_cache
def list_say_voices() -> dict:
    """跑 `say -v ?` 解析出 {声音名: 语系}，如 {'Tingting':'zh_CN','Samantha':'en_US'}。
    缓存结果——只探一次。"""
    if not shutil.which("say"):
        return {}
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {}
    voices = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and "_" in parts[1]:      # 第二列形如 zh_CN/en_US 才是有效语系
            voices[parts[0]] = parts[1]
    return voices


@lru_cache
def resolve_voices() -> tuple[str, str]:
    """解析实际可用的(中文声音, 英文声音)。策略：偏好→同语系兜底→报错。
    英文缺失=硬错误(没法读英文)；中文缺失=告警并退到英文(读音会不准，但服务能用)。"""
    s = get_settings()
    voices = list_say_voices()
    if not voices:
        raise DependencyError("未检测到 macOS `say`；say 后端依赖系统语音。")

    def pick(pref, prefixes):
        if pref in voices:                            # 首选就在 → 直接用
            return pref
        for name, loc in voices.items():              # 否则找同语系任意一个
            if any(loc.startswith(p) for p in prefixes):
                return name
        return None

    zh = pick(s.zh_voice, ["zh"])
    en = pick(s.en_voice, ["en"])
    if not en:
        raise DependencyError("系统没有任何英文语音，无法合成英文。")
    if not zh:
        log.warning("未找到中文语音，中文将回退英文语音朗读(发音不准)。")
        zh = en
    return zh, en


@lru_cache
def resolve_font() -> str:
    """挑第一个存在的候选字体；都没有则报错(视频渲染必须要字体)。"""
    for f in _FONT_CANDIDATES:
        if os.path.exists(f):
            return f
    raise DependencyError("未找到含中文+IPA 字形的字体(如 Arial Unicode)，无法渲染视频。")


def check_binaries():
    """确认 ffmpeg/ffprobe 在 PATH 里，缺了给出安装提示。"""
    missing = [b for b in ("ffmpeg", "ffprobe") if not shutil.which(b)]
    if missing:
        raise DependencyError(f"缺少命令行依赖：{', '.join(missing)}（brew install ffmpeg）。")


def healthcheck() -> dict:
    """健康检查：逐项探测但【不抛异常】，返回状态字典(供 /health 显示 ok/degraded)。"""
    status = {"say": bool(list_say_voices()), "ffmpeg": bool(shutil.which("ffmpeg")),
              "ffprobe": bool(shutil.which("ffprobe")),
              "font": any(os.path.exists(f) for f in _FONT_CANDIDATES)}
    try:
        zh, en = resolve_voices()
        status["zh_voice"], status["en_voice"] = zh, en
    except DependencyError:
        status["zh_voice"] = status["en_voice"] = None
    status["ok"] = all([status["say"], status["ffmpeg"], status["ffprobe"], status["font"]])
    return status
