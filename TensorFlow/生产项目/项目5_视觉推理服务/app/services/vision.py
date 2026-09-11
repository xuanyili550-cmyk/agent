"""视觉:stub(按输入确定性给标签,离线) | real(🟡 transformers pipeline,需模型)。"""
import hashlib
from ..core.config import get_settings
from ..core.exceptions import ValidationError
LABELS = ["cat", "dog", "car", "flower", "person"]
def classify(image_b64: str):
    if not image_b64: raise ValidationError("image 不能为空")
    if get_settings().backend == "real":
        from transformers import pipeline
        raise RuntimeError("real 后端需传真实图像 + 模型;本机用 stub")
    idx = int(hashlib.md5(image_b64.encode()).hexdigest(), 16) % len(LABELS)
    return {"label": LABELS[idx], "score": 0.9}
def detect(image_b64: str):
    if not image_b64: raise ValidationError("image 不能为空")
    return {"boxes": [{"label": classify(image_b64)["label"], "box": [10, 10, 100, 100], "score": 0.88}]}
