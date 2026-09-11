"""模型注册表:训练产出登记版本 + 指标,支持列出/取最新(模拟模型仓库)。"""
import json, os, time
from ..core.config import get_settings
def _path():
    d = get_settings().registry_dir; os.makedirs(d, exist_ok=True)
    return os.path.join(d, "registry.json")
def _load():
    p = _path(); return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
def register(metrics: dict) -> dict:
    reg = _load()
    entry = {"version": f"v{len(reg)+1}", "ts": time.strftime("%Y-%m-%d %H:%M"), "metrics": metrics}
    reg.append(entry); json.dump(reg, open(_path(), "w", encoding="utf-8"), ensure_ascii=False)
    return entry
def list_all(): return _load()
def clear():
    p = _path()
    if os.path.exists(p): os.remove(p)
