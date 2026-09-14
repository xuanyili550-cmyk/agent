from __future__ import annotations

import threading

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class CLIPModelUnavailableError(RuntimeError):
    """CLIP weights/processor could not be loaded: no local HF cache and no network access."""


class CLIPBackend:
    """Process-wide cache of (model, processor) pairs keyed by model name.

    Shared by character consistency QC and scene alignment QC so the (large) CLIP
    weights are only ever loaded once per process, regardless of how many checkers
    are instantiated.
    """

    _lock = threading.Lock()
    _cache: dict[str, tuple] = {}

    @classmethod
    def get(cls, model_name: str = DEFAULT_CLIP_MODEL):
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
            model.eval()
            cls._cache[model_name] = (model, processor)
            return model, processor
