"""Prompt A/B 评测:同一批用例,两个提示词模板各生成答案,再用 evaluator 比指标,选胜者。
stub 生成器体现『提示词质量』:含据实约束(无据/据实)的模板遇未知会拒答(不幻觉);无约束的会硬答(幻觉)。"""
from ..core.exceptions import ValidationError
from .evaluator import evaluate
_GROUNDED_HINT = ("无据", "据实", "未提及", "只根据")


def _grounded(case):
    ref, ctx = case.get("reference", ""), case.get("context", "")
    return ref if ref and ref in ctx else "资料中未提及"        # 未知→安全拒答


def _naive(case):
    ref, ctx = case.get("reference", ""), case.get("context", "")
    return ref if ref and ref in ctx else "大概是这样吧(未经核实)"  # 未知→硬答=幻觉


def generate(template: str, case: dict) -> str:
    return _grounded(case) if any(h in template for h in _GROUNDED_HINT) else _naive(case)


def ab_eval(cases, template_a, template_b):
    if not cases:
        raise ValidationError("cases 不能为空")
    def run(tpl):
        return evaluate([{**c, "answer": generate(tpl, c)} for c in cases])
    ma, mb = run(template_a), run(template_b)
    winner = "A" if ma["avg_judge_score"] >= mb["avg_judge_score"] else "B"
    return {"A": ma, "B": mb, "winner": winner,
            "note": "据实约束模板在'未知问题'上拒答→幻觉率更低→胜出"}
