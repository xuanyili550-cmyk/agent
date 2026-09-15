"""CLIP 模型进程级缓存后端。

流水线位置：08_QC 中角色一致性（character/consistency_checker.py）和场景对齐（scene/scene_qc.py）
两个检查器共用的底层——它们都需要同一份 CLIP 权重与 processor。
为什么单独抽出来：CLIP 权重几百 MB，加载要几秒；编排层可能为每个镜头 new 一个检查器，
如果各自加载会把显存和时间都耗光。用类级别的字典 + 锁做进程内单例，保证每个模型名只加载一次。
"""

from __future__ import annotations

import threading

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class CLIPModelUnavailableError(RuntimeError):
    """CLIP 权重 / processor 无法加载：本地没有 HF 缓存且没有网络。"""


class CLIPBackend:
    """按模型名缓存 (model, processor) 二元组的进程级缓存。

    角色一致性 QC 和场景对齐 QC 共用它，这样无论实例化多少个检查器，
    （体积很大的）CLIP 权重在一个进程里只会加载一次。
    """

    _lock = threading.Lock()
    _cache: dict[str, tuple] = {}

    @classmethod
    def get(cls, model_name: str = DEFAULT_CLIP_MODEL):
        """返回缓存的 (model, processor)，没有则加载并缓存。

        双重检查锁：先不加锁查一次（热路径零开销），加锁后再查一次，防止两个线程同时进入加载。
        加载失败统一转成 CLIPModelUnavailableError，让调用方能区分"环境缺依赖"和"图片本身有问题"。
        """
        if model_name in cls._cache:
            return cls._cache[model_name]
        with cls._lock:
            if model_name in cls._cache:
                return cls._cache[model_name]
            try:
                from transformers import CLIPModel, CLIPProcessor
            except ImportError as exc:
                raise CLIPModelUnavailableError("transformers is not installed. Install it with `pip install -r 08_QC/requirements-qc.txt`.") from exc
            try:
                model = CLIPModel.from_pretrained(model_name)
                processor = CLIPProcessor.from_pretrained(model_name)
            except Exception as exc:
                raise CLIPModelUnavailableError(
                    f"Could not load CLIP model '{model_name}'. This checker needs either "
                    "a pre-populated Hugging Face cache (~/.cache/huggingface) or network "
                    "access to download the weights once from huggingface.co. "
                    f"Underlying error: {type(exc).__name__}: {exc}"
                ) from exc
            model.eval()  # 推理模式：关掉 dropout，保证同一张图每次得到相同的 embedding
            cls._cache[model_name] = (model, processor)
            return model, processor
