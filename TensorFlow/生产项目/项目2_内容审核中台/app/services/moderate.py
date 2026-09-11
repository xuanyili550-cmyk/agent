"""审核编排:检测→打分→分级(通过/人工/拦截)→PII 脱敏。可切风险后端。"""
from ..core.config import get_settings
from ..core.exceptions import ValidationError
from . import rules


def _risk_score(text, sensitive):
    """rule 后端:命中敏感词越多分越高。model 后端留钩子(🔴 毒性分类模型)。"""
    s = get_settings()
    if s.risk_backend == "model":
        from .risk_model import score   # 🔴 可选,需模型
        return score(text)
    return min(1.0, 0.4 * len(sensitive))   # 每个敏感词 +0.4


def moderate(text: str) -> dict:
    s = get_settings()
    if not (text or "").strip():
        raise ValidationError("内容不能为空")
    if len(text) > s.max_text_chars:
        raise ValidationError("内容过长")
    sensitive = rules.find_sensitive(text)
    pii = rules.find_pii(text)
    score = _risk_score(text, sensitive)
    grade = "block" if score >= s.block_threshold else ("review" if score >= s.review_threshold else "pass")
    return {"grade": grade, "risk_score": round(score, 2), "sensitive": sensitive,
            "pii": pii, "masked": rules.mask_pii(text)}
