"""滑动窗口限流(每客户端每分钟)。生产用 Redis;这里内存版。"""
import time, threading
from collections import defaultdict
from ..core.config import get_settings
_hits = defaultdict(list); _lock = threading.Lock()
def check(client_id: str):
    s = get_settings(); now = time.time()
    with _lock:
        q = _hits[client_id] = [t for t in _hits[client_id] if now - t < 60]
        if len(q) >= s.rate_limit_per_min:
            return False
        q.append(now); return True
