"""
================================================================================
 服务层 · 内容哈希缓存（相同输入只算一次）
================================================================================
 【为什么需要】TTS 合成、视频渲染都很慢/费 CPU。同一段文本被反复请求(用户重播、多人问同一句)
   若每次都重算，既慢又浪费。用"内容哈希"当缓存键：只要文本+参数一样，就复用上次的产物文件。

 【键怎么设计(关键)】
   key = sha256(种类 | 文本 | 排序后的参数)。参数必须【全量参与】——比如 TTS 的音色/后端/语速，
   视频的分辨率/帧率。少放一个参数，就会"换了音色还命中旧音频"(串味 bug)。排序保证同参不同序也命中。

 【TTL 清理】产物只留一段时间(默认 6 小时)，过期删除，避免磁盘无限增长(见 cleanup，由启动/请求触发)。
================================================================================
"""
import hashlib
import os
import time

from ..core.config import get_settings


def _dir() -> str:
    """缓存目录(不存在就建)。生产可指到独立数据盘/挂载卷。"""
    d = get_settings().cache_dir
    os.makedirs(d, exist_ok=True)
    return d


def key_for(kind: str, text: str, params: dict) -> str:
    """生成缓存键：种类+文本+全部参数(按名排序)一起哈希。取前 32 位十六进制做文件名。"""
    raw = kind + "|" + text + "|" + "|".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def path_for(key: str, ext: str) -> str:
    return os.path.join(_dir(), f"{key}.{ext}")


def get(key: str, ext: str) -> str | None:
    """命中返回路径，否则 None。要求文件存在【且非空】(防止上次写了一半的坏文件被当命中)。"""
    p = path_for(key, ext)
    return p if os.path.exists(p) and os.path.getsize(p) > 0 else None


def cleanup():
    """删除超过 TTL 的缓存产物。启动时和(可扩展)定期触发，防磁盘涨满。"""
    s = get_settings()
    now = time.time()
    d = _dir()
    for name in os.listdir(d):
        p = os.path.join(d, name)
        try:
            if now - os.path.getmtime(p) > s.cache_ttl_seconds:
                os.remove(p)
        except OSError:                       # 并发下文件可能已被别处删——忽略即可
            pass
