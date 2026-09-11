"""内容哈希缓存(相同问题直接回);TTL 由调用方/定时清理。"""
import hashlib
_CACHE = {}


def key(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:32]


def get(k):
    return _CACHE.get(k)


def put(k, v):
    _CACHE[k] = v
    return v
