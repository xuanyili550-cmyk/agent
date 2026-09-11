"""训练数据校验:指令微调格式 [{instruction, output}]。脏数据早拦截。"""
from ..core.config import get_settings
from ..core.exceptions import ValidationError
def validate(samples: list[dict]) -> int:
    s = get_settings()
    if len(samples) < s.min_samples:
        raise ValidationError(f"样本过少(<{s.min_samples})")
    for i, x in enumerate(samples):
        if not x.get("instruction") or not x.get("output"):
            raise ValidationError(f"第{i}条缺 instruction/output")
    return len(samples)
