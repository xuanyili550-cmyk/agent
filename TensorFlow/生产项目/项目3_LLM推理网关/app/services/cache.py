"""提示缓存:相同 messages 直接回(省算力/延迟)。"""
import hashlib
_C = {}
def key(model, messages):
    return hashlib.sha256((model + "|" + str(messages)).encode()).hexdigest()[:32]
def get(k): return _C.get(k)
def put(k, v): _C[k] = v; return v
