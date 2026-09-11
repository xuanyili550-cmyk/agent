"""
================================================================================
 服务层 · 模型单例加载（AutoTokenizer + AutoModel，只用底层，不用 pipeline）
================================================================================
 【为什么单例 + 懒加载】加载模型=读几百 MB 权重 + 上设备，是秒级重操作。绝不能每次请求都加载
   (延迟爆炸、显存爆)。所以【进程内只加载一次、常驻复用】；且懒加载——第一次真用时才加载
   (启动快、没请求不占资源)。加锁防并发首次加载时重复初始化。

 【为什么用 AutoModel 而不是 pipeline】
   pipeline 把"分词+前向+后处理"打包好了，方便但黑盒、后处理不可控(池化方式、归一化都由它定)。
   生产要精确控制池化/归一化/批处理,就得下沉到 AutoTokenizer + AutoModel 手工前向——这正是"底层原理"。

 【设备选择】auto → 优先 Apple mps / NVIDIA cuda / 否则 cpu。
================================================================================
"""
import threading

from ..core.config import get_settings
from ..core.logging import get_logger

log = get_logger("model")
_lock = threading.Lock()
_encoder = None     # (tokenizer, model, device) 单例
_reranker = None


def _pick_device(pref: str) -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_encoder():
    """懒加载嵌入模型单例 → (tokenizer, model, device)。线程安全。"""
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:                     # 双检锁：拿锁后再确认一次没被别的线程加载
                from transformers import AutoModel, AutoTokenizer
                import torch
                s = get_settings()
                dev = _pick_device(s.device)
                log.info("加载嵌入模型 %s → %s", s.model_name, dev)
                tok = AutoTokenizer.from_pretrained(s.model_name)
                mdl = AutoModel.from_pretrained(s.model_name).to(dev).eval()   # eval 关 dropout
                _encoder = (tok, mdl, dev)
    return _encoder


def get_reranker():
    """懒加载 cross-encoder 重排模型 → (tokenizer, model, device)。"""
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                s = get_settings()
                dev = _pick_device(s.device)
                log.info("加载重排模型 %s → %s", s.rerank_model, dev)
                tok = AutoTokenizer.from_pretrained(s.rerank_model)
                mdl = AutoModelForSequenceClassification.from_pretrained(s.rerank_model).to(dev).eval()
                _reranker = (tok, mdl, dev)
    return _reranker
