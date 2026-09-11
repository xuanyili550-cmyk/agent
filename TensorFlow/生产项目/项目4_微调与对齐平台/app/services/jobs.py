import threading, uuid
from concurrent.futures import ThreadPoolExecutor
from ..core.config import get_settings
_pool = ThreadPoolExecutor(max_workers=get_settings().max_concurrent_jobs)
_jobs, _lock = {}, threading.Lock()
def submit(fn, *a):
    jid = uuid.uuid4().hex[:12]
    with _lock: _jobs[jid] = {"status": "running", "progress": 0.0, "result": None}
    def prog(d, t):
        with _lock: _jobs[jid]["progress"] = round(d / max(1, t), 2)
    def run():
        try:
            r = fn(prog, *a)
            with _lock: _jobs[jid].update(status="done", progress=1.0, result=r)
        except Exception as e:
            with _lock: _jobs[jid].update(status="error", result=str(e))
    _pool.submit(run); return jid
def get(jid):
    with _lock:
        j = _jobs.get(jid); return dict(j) if j else None
