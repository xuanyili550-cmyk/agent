"""文档正文存储：id → text(检索返回原文用)。JSON 持久化，与向量库 id 对应。"""
import json
import os
import threading

from ..core.config import get_settings

_lock = threading.Lock()
_docs: dict[str, str] | None = None


def _path():
    d = get_settings().index_dir
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "docs.json")


def _load():
    global _docs
    if _docs is None:
        p = _path()
        _docs = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return _docs


def put(items: dict[str, str]):
    with _lock:
        d = _load()
        d.update(items)
        json.dump(d, open(_path(), "w", encoding="utf-8"), ensure_ascii=False)


def get(_id: str) -> str:
    return _load().get(_id, "")


def clear():
    global _docs
    with _lock:
        _docs = {}
        json.dump(_docs, open(_path(), "w", encoding="utf-8"), ensure_ascii=False)
